use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize, Device, SampleFormat, Stream, StreamConfig, SupportedBufferSize,
    SupportedStreamConfig,
};

use super::scheduler::{OutputSample, Scheduler};
use super::sound_bank::{SoundBank, SoundPreset};
use crate::model::{normalize_buffer_frames, SharedParams};

/// Sample rate preferred over the ALSA device default. PipeWire and
/// PulseAudio servers usually run at 48 kHz while ALSA reports 44.1 kHz as
/// the device default, so requesting 48 kHz keeps the metronome out of the
/// server's resampler.
const PREFERRED_SAMPLE_RATE: u32 = 48_000;

/// Picks the output configuration for a device: the default config, unless
/// the device also supports the default format and channel count at
/// [`PREFERRED_SAMPLE_RATE`] (falling back to any format at that rate).
fn preferred_output_config(device: &Device) -> Result<SupportedStreamConfig> {
    let default_config = device
        .default_output_config()
        .context("default output configuration is unavailable")?;
    let preferred_rate = cpal::SampleRate(PREFERRED_SAMPLE_RATE);
    let default_format = default_config.sample_format();
    let default_channels = default_config.channels();

    let ranges = device
        .supported_output_configs()
        .context("supported output configurations are unavailable")?
        .collect::<Vec<_>>();
    let preferred_shape = ranges
        .iter()
        .find(|range| {
            range.sample_format() == default_format
                && range.channels() == default_channels
                && range.try_with_sample_rate(preferred_rate).is_some()
        })
        .or_else(|| {
            ranges
                .iter()
                .find(|range| range.try_with_sample_rate(preferred_rate).is_some())
        })
        .and_then(|range| range.try_with_sample_rate(preferred_rate));

    Ok(preferred_shape.unwrap_or(default_config))
}

pub struct AudioEngine {
    stream: Option<Stream>,
    device: Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    params: Arc<SharedParams>,
    sound_bank: Arc<ArcSwap<SoundBank>>,
    started: Arc<AtomicBool>,
    /// Set by the stream error callback when the output device dies; the
    /// next ensure/recover rebuilds the stream, re-resolving the default
    /// device in case it was replaced.
    stream_errored: Arc<AtomicBool>,
    pub sample_rate: u32,
}

#[allow(dead_code)]
impl AudioEngine {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no default audio output device")?;
        let supported = preferred_output_config(&device)?;
        let sample_format = supported.sample_format();
        let supported_buffer_size = *supported.buffer_size();
        let mut config: StreamConfig = supported.into();
        let sample_rate = config.sample_rate.0;
        let params = Arc::new(SharedParams::new());
        // Create live params before deriving the initial ALSA request so a
        // future change to the default only has one source of truth.
        params.sample_rate.store(sample_rate, Ordering::Relaxed);
        config.buffer_size = configured_buffer_size(supported_buffer_size, params.buffer_frames());
        // Transient initial bank only: the UI loads the persisted sample
        // before any widget (or sound) exists, and the headless smoke test
        // overrides it through settings or --preset.
        let bank = Arc::new(ArcSwap::new(Arc::new(SoundBank::preset(
            SoundPreset::Woodblock,
            sample_rate,
        ))));
        let started = Arc::new(AtomicBool::new(false));
        let stream_errored = Arc::new(AtomicBool::new(false));

        Ok(Self {
            stream: None,
            device,
            config,
            sample_format,
            params,
            sound_bank: bank,
            started,
            stream_errored,
            sample_rate,
        })
    }

    /// Re-resolves the default output device and its configuration, for
    /// example after the previous device was unplugged. Refreshing the
    /// sample rate keeps the scheduler period correct; an already decoded
    /// click bank keeps its old rate, which may shift its pitch slightly
    /// until it is reselected.
    fn refresh_device(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no default audio output device")?;
        let supported = preferred_output_config(&device)?;
        let supported_buffer_size = *supported.buffer_size();
        let sample_format = supported.sample_format();
        let mut config: StreamConfig = supported.into();
        config.buffer_size =
            configured_buffer_size(supported_buffer_size, self.params.buffer_frames());
        let sample_rate = config.sample_rate.0;

        self.device = device;
        self.config = config;
        self.sample_format = sample_format;
        if self.sample_rate != sample_rate {
            self.sample_rate = sample_rate;
            self.params
                .sample_rate
                .store(sample_rate, Ordering::Relaxed);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_stream<T>(
        device: &cpal::Device,
        config: &StreamConfig,
        params: Arc<SharedParams>,
        bank: Arc<ArcSwap<SoundBank>>,
        error: impl FnMut(cpal::StreamError) + Send + 'static,
    ) -> Result<Stream>
    where
        T: cpal::SizedSample + OutputSample + 'static,
    {
        let mut scheduler = Scheduler::new(0, params.beat_count());
        let channels = config.channels.max(1);

        let stream = device.build_output_stream(
            config,
            move |output: &mut [T], _| {
                let bank = bank.load();
                scheduler.render(&params, &bank, output, channels.into());
            },
            error,
            None,
        )?;

        Ok(stream)
    }

    fn ensure_stream(&mut self) -> Result<()> {
        if self.stream.is_some() && !self.stream_errored.swap(false, Ordering::SeqCst) {
            return Ok(());
        }

        // The previous stream is either absent or reported an error; drop it
        // and resolve the current default device afresh.
        self.stream = None;
        self.refresh_device()?;

        // The effective latency is decided by the audio server, so recording
        // what was actually negotiated makes mismatches (for example a
        // requested 1024 frames becoming a 256-frame PipeWire quantum)
        // diagnosable without external tools.
        eprintln!(
            "audio output: host={host} device=\"{device}\" format={sample_format:?} rate={rate} Hz channels={channels} buffer={buffer:?}",
            host = cpal::default_host().id().name(),
            device = self
                .device
                .name()
                .unwrap_or_else(|_| "<unknown>".to_string()),
            sample_format = self.sample_format,
            rate = self.config.sample_rate.0,
            channels = self.config.channels,
            buffer = self.config.buffer_size,
        );

        let stream_errored = self.stream_errored.clone();
        let error = move |stream_error: cpal::StreamError| {
            eprintln!("audio stream error: {stream_error}");
            stream_errored.store(true, Ordering::SeqCst);
        };

        let stream = match self.sample_format {
            SampleFormat::F32 => Self::build_stream::<f32>(
                &self.device,
                &self.config,
                self.params.clone(),
                self.sound_bank.clone(),
                error,
            )?,
            SampleFormat::I16 => Self::build_stream::<i16>(
                &self.device,
                &self.config,
                self.params.clone(),
                self.sound_bank.clone(),
                error,
            )?,
            SampleFormat::U16 => Self::build_stream::<u16>(
                &self.device,
                &self.config,
                self.params.clone(),
                self.sound_bank.clone(),
                error,
            )?,
            format => return Err(anyhow!("unsupported output sample format: {format}")),
        };

        self.stream = Some(stream);
        Ok(())
    }

    pub fn params(&self) -> &SharedParams {
        &self.params
    }

    pub fn shared_params(&self) -> Arc<SharedParams> {
        self.params.clone()
    }

    pub fn set_sound_bank(&self, bank: SoundBank) {
        self.sound_bank.store(Arc::new(bank));
    }

    pub fn set_normal_wav(&self, path: &std::path::Path) -> Result<()> {
        let mut bank = self.sound_bank.load().as_ref().clone();
        bank.normal = super::sound_bank::decode_wav(path, self.sample_rate)?;
        self.sound_bank.store(Arc::new(bank));
        Ok(())
    }

    pub fn set_accent_wav(&self, path: &std::path::Path) -> Result<()> {
        let mut bank = self.sound_bank.load().as_ref().clone();
        // The accent WAV never plays directly; it becomes the source the
        // first-beat sound is derived from.
        bank.first_source = super::sound_bank::decode_wav(path, self.sample_rate)?;
        bank.first = super::sound_bank::derive_first_sound(&bank.first_source, self.sample_rate);
        self.sound_bank.store(Arc::new(bank));
        Ok(())
    }

    pub fn play(&mut self) -> Result<()> {
        self.ensure_stream()?;

        // Only mark the stream started after play() succeeds; otherwise an
        // early failure would permanently skip the stream start on every
        // later attempt.
        if !self.started.load(Ordering::SeqCst) {
            if let Some(stream) = self.stream.as_ref() {
                stream.play()?;
            }
            self.started.store(true, Ordering::SeqCst);
        }

        self.params.running.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Plays one measure from the currently selected sound. The preview uses
    /// the current meter, tempo and per-beat pattern so a sample can be heard
    /// in its real musical context.
    pub fn preview_measure(&mut self) -> Result<Duration> {
        self.ensure_stream()?;

        if let Some(stream) = self.stream.as_ref() {
            stream.play()?;
        }
        self.started.store(true, Ordering::SeqCst);

        let bank = self.sound_bank.load();
        let sample_rate = f64::from(self.params.sample_rate.load(Ordering::Relaxed).max(1));
        let beats = f64::from(self.params.beat_count().max(1));
        let bpm = f64::from(self.params.bpm().max(1));
        let measure_duration = beats * 60.0 / bpm;
        let sound_tail = [bank.normal.len(), bank.first_source.len(), bank.first.len()]
            .into_iter()
            .max()
            .unwrap_or_default() as f64
            / sample_rate;
        self.params
            .preview_generation
            .fetch_add(1, Ordering::SeqCst);

        Ok(Duration::from_secs_f64(measure_duration + sound_tail))
    }

    pub fn stop(&mut self) {
        self.params.running.store(false, Ordering::Relaxed);
    }

    /// Plays the current sample's normal click immediately, without changing
    /// metronome playback or restarting its beat pattern.
    pub fn play_normal_one_shot(&mut self) -> Result<()> {
        self.ensure_stream()?;

        if let Some(stream) = self.stream.as_ref() {
            stream.play()?;
        }
        self.started.store(true, Ordering::SeqCst);
        self.params.one_shot_pending.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.params.running.load(Ordering::Relaxed)
    }

    /// Stores a new requested buffer size and rebuilds the output stream.
    /// Audio servers can clamp or ignore this request, so the value is a
    /// latency preference rather than an exact device buffer guarantee.
    pub fn set_buffer_frames(&mut self, requested_frames: u32) -> Result<()> {
        let requested_frames = normalize_buffer_frames(requested_frames);
        let previous_frames = self.params.buffer_frames();
        if requested_frames == previous_frames {
            return Ok(());
        }

        let was_started = self.started.load(Ordering::SeqCst);
        self.params.set_buffer_frames(requested_frames);

        match self.restart_stream(was_started) {
            Ok(()) => Ok(()),
            Err(error) => {
                // Keep the persisted choice and the loaded stream aligned.
                self.params.set_buffer_frames(previous_frames);
                Err(error)
            }
        }
    }

    /// Rebuilds the output stream after a device error. Returns whether a
    /// rebuild was attempted; playback resumes automatically if the stream
    /// had been started before.
    pub fn recover_stream_if_needed(&mut self) -> Result<bool> {
        if !self.stream_errored.load(Ordering::SeqCst) {
            return Ok(false);
        }

        self.ensure_stream()?;
        if self.started.load(Ordering::SeqCst) {
            if let Some(stream) = self.stream.as_ref() {
                stream.play()?;
            }
        }
        Ok(true)
    }

    /// Drops the current stream, recreates it from the latest config, and
    /// resumes it when the user had already enabled audio output.
    fn restart_stream(&mut self, was_started: bool) -> Result<()> {
        self.stream = None;
        self.ensure_stream()?;

        if was_started {
            if let Some(stream) = self.stream.as_ref() {
                stream.play()?;
            }
        }

        Ok(())
    }
}

fn configured_buffer_size(supported: SupportedBufferSize, requested_frames: u32) -> BufferSize {
    match supported {
        SupportedBufferSize::Range { min, max } => {
            BufferSize::Fixed(requested_frames.clamp(min, max))
        }
        SupportedBufferSize::Unknown => BufferSize::Default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DEFAULT_AUDIO_BUFFER_FRAMES;

    #[test]
    fn configured_buffer_size_stays_within_device_limits() {
        let requested = DEFAULT_AUDIO_BUFFER_FRAMES;
        assert_eq!(
            configured_buffer_size(SupportedBufferSize::Range { min: 16, max: 1024 }, requested),
            BufferSize::Fixed(512)
        );
        assert_eq!(
            configured_buffer_size(
                SupportedBufferSize::Range {
                    min: 512,
                    max: 2048
                },
                requested
            ),
            BufferSize::Fixed(512)
        );
        assert_eq!(
            configured_buffer_size(SupportedBufferSize::Range { min: 16, max: 384 }, requested),
            BufferSize::Fixed(384)
        );
        assert_eq!(
            configured_buffer_size(SupportedBufferSize::Unknown, requested),
            BufferSize::Default
        );
    }
}
