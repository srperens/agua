use gstreamer as gst;
use gstreamer_audio as gst_audio;
use gstreamer_base as gst_base;

use gst::glib;
use gst::glib::prelude::*;
use gst::prelude::{ElementExtManual, PadExtManual};
use gst_audio::subclass::prelude::*;

use agua_core::{Payload, StreamEmbedder, WatermarkConfig, WatermarkKey};

mod imp {
    use super::*;
    use gstreamer_base::subclass::base_transform::GenerateOutputSuccess;
    use once_cell::sync::Lazy;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    #[derive(Clone, Debug)]
    struct Settings {
        payload_hex: Option<String>,
        payload: Option<Payload>,
        key_passphrase: String,
        key: WatermarkKey,
        strength: f32,
        frame_size: usize,
        num_bin_pairs: usize,
        min_freq_hz: f32,
        max_freq_hz: f32,
        bin_spacing: usize,
        offset_seconds: f32,
        sample_rate: u32,
        channels: u32,
    }

    impl Default for Settings {
        fn default() -> Self {
            let key_passphrase = "agua-default-key".to_string();
            let key = WatermarkKey::from_passphrase(&key_passphrase);
            Self {
                payload_hex: None,
                payload: None,
                key_passphrase,
                key,
                strength: 0.1,
                frame_size: 1024,
                num_bin_pairs: 60,
                min_freq_hz: 860.0,
                max_freq_hz: 4300.0,
                bin_spacing: 8,
                offset_seconds: 0.0,
                sample_rate: 48000,
                channels: 1,
            }
        }
    }

    pub struct AguaEmbed {
        settings: Mutex<Settings>,
        buffer_state: Mutex<BufferState>,
    }

    impl Default for AguaEmbed {
        fn default() -> Self {
            Self {
                settings: Mutex::new(Settings::default()),
                buffer_state: Mutex::new(BufferState::default()),
            }
        }
    }

    struct BufferState {
        /// One StreamEmbedder per audio channel, created on first buffer.
        embedders: Vec<StreamEmbedder>,
        processed: VecDeque<f32>,
        pending: VecDeque<PendingBuf>,
        initialized: bool,
        offset_remaining_frames: usize,
    }

    impl Default for BufferState {
        fn default() -> Self {
            Self {
                embedders: Vec::new(),
                processed: VecDeque::new(),
                pending: VecDeque::new(),
                initialized: false,
                offset_remaining_frames: 0,
            }
        }
    }

    #[derive(Clone, Debug)]
    struct PendingBuf {
        size_samples: usize,
        pts: Option<gst::ClockTime>,
        duration: Option<gst::ClockTime>,
        flags: gst::BufferFlags,
        offset: Option<u64>,
        offset_end: Option<u64>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for AguaEmbed {
        const NAME: &'static str = "GstAguaEmbed";
        type Type = super::AguaEmbed;
        type ParentType = gst_audio::AudioFilter;
    }

    impl ObjectImpl for AguaEmbed {
        fn properties() -> &'static [glib::ParamSpec] {
            static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
                vec![
                    glib::ParamSpecString::builder("payload")
                        .nick("Payload")
                        .blurb("Hex payload (32 chars)")
                        .build(),
                    glib::ParamSpecString::builder("key")
                        .nick("Key")
                        .blurb("Key passphrase")
                        .default_value(Some("agua-default-key"))
                        .build(),
                    glib::ParamSpecFloat::builder("strength")
                        .nick("Strength")
                        .blurb("Embedding strength (power-law exponent delta)")
                        .minimum(0.0)
                        .maximum(10.0)
                        .default_value(0.1)
                        .build(),
                    glib::ParamSpecUInt::builder("frame-size")
                        .nick("Frame size")
                        .blurb("FFT frame size (power of 2)")
                        .minimum(256)
                        .maximum(8192)
                        .default_value(1024)
                        .build(),
                    glib::ParamSpecUInt::builder("num-bin-pairs")
                        .nick("Bin pairs")
                        .blurb("Number of bin pairs per frame")
                        .minimum(1)
                        .maximum(2000)
                        .default_value(60)
                        .build(),
                    glib::ParamSpecFloat::builder("min-freq")
                        .nick("Min frequency")
                        .blurb("Minimum frequency in Hz for watermark embedding")
                        .minimum(20.0)
                        .maximum(20000.0)
                        .default_value(860.0)
                        .build(),
                    glib::ParamSpecFloat::builder("max-freq")
                        .nick("Max frequency")
                        .blurb("Maximum frequency in Hz for watermark embedding")
                        .minimum(100.0)
                        .maximum(20000.0)
                        .default_value(4300.0)
                        .build(),
                    glib::ParamSpecUInt::builder("bin-spacing")
                        .nick("Bin spacing")
                        .blurb("Spacing between bins in each pair (1 = adjacent)")
                        .minimum(1)
                        .maximum(100)
                        .default_value(8)
                        .build(),
                    glib::ParamSpecFloat::builder("offset-seconds")
                        .nick("Offset seconds")
                        .blurb("Delay before embedding starts (seconds)")
                        .minimum(0.0)
                        .maximum(10_000.0)
                        .default_value(0.0)
                        .build(),
                ]
            });
            PROPERTIES.as_ref()
        }

        fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
            let mut settings = self.settings.lock().unwrap();
            match pspec.name() {
                "payload" => {
                    let payload_hex = value.get::<Option<String>>().unwrap_or(None);
                    settings.payload = payload_hex.as_ref().and_then(|p| Payload::from_hex(p).ok());
                    settings.payload_hex = payload_hex;
                }
                "key" => {
                    let key_passphrase = value
                        .get::<Option<String>>()
                        .unwrap_or(None)
                        .unwrap_or_else(|| "agua-default-key".to_string());
                    settings.key = WatermarkKey::from_passphrase(&key_passphrase);
                    settings.key_passphrase = key_passphrase;
                }
                "strength" => {
                    settings.strength = value.get::<f32>().unwrap_or(0.1);
                }
                "frame-size" => {
                    settings.frame_size = value.get::<u32>().unwrap_or(1024) as usize;
                }
                "num-bin-pairs" => {
                    settings.num_bin_pairs = value.get::<u32>().unwrap_or(60) as usize;
                }
                "min-freq" => {
                    settings.min_freq_hz = value.get::<f32>().unwrap_or(860.0);
                }
                "max-freq" => {
                    settings.max_freq_hz = value.get::<f32>().unwrap_or(4300.0);
                }
                "bin-spacing" => {
                    settings.bin_spacing = value.get::<u32>().unwrap_or(1).max(1) as usize;
                }
                "offset-seconds" => {
                    settings.offset_seconds = value.get::<f32>().unwrap_or(0.0);
                }
                _ => unimplemented!(),
            }
        }

        fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
            let settings = self.settings.lock().unwrap();
            match pspec.name() {
                "payload" => settings.payload_hex.to_value(),
                "key" => settings.key_passphrase.to_value(),
                "strength" => settings.strength.to_value(),
                "frame-size" => (settings.frame_size as u32).to_value(),
                "num-bin-pairs" => (settings.num_bin_pairs as u32).to_value(),
                "min-freq" => settings.min_freq_hz.to_value(),
                "max-freq" => settings.max_freq_hz.to_value(),
                "bin-spacing" => (settings.bin_spacing as u32).to_value(),
                "offset-seconds" => settings.offset_seconds.to_value(),
                _ => unimplemented!(),
            }
        }
    }

    impl GstObjectImpl for AguaEmbed {}

    impl ElementImpl for AguaEmbed {
        fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
            static ELEMENT_METADATA: std::sync::OnceLock<gst::subclass::ElementMetadata> =
                std::sync::OnceLock::new();
            Some(ELEMENT_METADATA.get_or_init(|| {
                gst::subclass::ElementMetadata::new(
                    "Agua embed",
                    "Filter/Effect/Audio",
                    "Embeds an agua watermark into audio",
                    "Eyevinn",
                )
            }))
        }
    }

    impl AudioFilterImpl for AguaEmbed {
        fn allowed_caps() -> &'static gst::Caps {
            static CAPS: std::sync::OnceLock<gst::Caps> = std::sync::OnceLock::new();
            CAPS.get_or_init(|| {
                gst_audio::AudioCapsBuilder::new()
                    .format(gst_audio::AudioFormat::F32le)
                    .layout(gst_audio::AudioLayout::Interleaved)
                    .build()
            })
        }

        fn setup(&self, info: &gst_audio::AudioInfo) -> Result<(), gst::LoggableError> {
            let mut settings = self.settings.lock().unwrap();
            settings.sample_rate = info.rate();
            settings.channels = info.channels();
            Ok(())
        }
    }

    impl BaseTransformImpl for AguaEmbed {
        const MODE: gst_base::subclass::BaseTransformMode =
            gst_base::subclass::BaseTransformMode::NeverInPlace;
        const PASSTHROUGH_ON_SAME_CAPS: bool = false;
        const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

        fn sink_event(&self, event: gst::Event) -> bool {
            if let gst::EventView::Eos(_) = event.view() {
                let settings = self.settings.lock().unwrap();
                let sample_rate = settings.sample_rate;
                let num_channels = settings.channels as usize;
                drop(settings);

                let mut state = self.buffer_state.lock().unwrap();
                if state.initialized && !state.embedders.is_empty() {
                    let channels = state.embedders.len();
                    let mut channel_outputs: Vec<Vec<f32>> = Vec::with_capacity(channels);
                    for ch in 0..channels {
                        channel_outputs.push(state.embedders[ch].flush());
                    }
                    let out_frames = channel_outputs.iter().map(|v| v.len()).min().unwrap_or(0);
                    if out_frames > 0 {
                        if channel_outputs.iter().any(|v| v.len() != out_frames) {
                            gst::warning!(
                                gst::CAT_DEFAULT,
                                obj = self.obj(),
                                "Channel flush length mismatch; truncating to {out_frames} frames"
                            );
                        }
                        for i in 0..out_frames {
                            for ch in 0..channels {
                                state.processed.push_back(channel_outputs[ch][i]);
                            }
                        }
                    }

                    // Push any remaining processed data as a final buffer.
                    // generate_output won't be called after EOS, so we enqueue
                    // it now so generate_output can drain it.
                    let remaining = state.processed.len();
                    if remaining > 0 {
                        let ch = num_channels.max(1);
                        let frames = remaining / ch;
                        let ns = (frames as u64).saturating_mul(gst::ClockTime::SECOND.nseconds())
                            / sample_rate as u64;
                        state.pending.push_back(PendingBuf {
                            size_samples: remaining,
                            pts: None,
                            duration: Some(gst::ClockTime::from_nseconds(ns)),
                            flags: gst::BufferFlags::empty(),
                            offset: None,
                            offset_end: None,
                        });
                    }
                }
            }
            self.parent_sink_event(event)
        }

        fn start(&self) -> Result<(), gst::ErrorMessage> {
            let mut state = self.buffer_state.lock().unwrap();
            state.embedders.clear();
            state.processed.clear();
            state.pending.clear();
            state.initialized = false;
            state.offset_remaining_frames = 0;
            Ok(())
        }

        fn query(&self, direction: gst::PadDirection, query: &mut gst::QueryRef) -> bool {
            if let gst::QueryViewMut::Latency(ref mut q) = query.view_mut() {
                let mut peer_query = gst::query::Latency::new();
                let upstream_ok = self
                    .obj()
                    .sink_pads()
                    .first()
                    .map(|p: &gst::Pad| p.peer_query(&mut peer_query))
                    .unwrap_or(false);
                if upstream_ok {
                    let (live, min, max) = peer_query.result();
                    let settings = self.settings.lock().unwrap();
                    let our_latency = gst::ClockTime::from_nseconds(
                        (settings.frame_size as u64)
                            .saturating_mul(gst::ClockTime::SECOND.nseconds())
                            / settings.sample_rate as u64,
                    );
                    q.set(live, min + our_latency, max.map(|m| m + our_latency));
                    return true;
                }
                return false;
            }
            BaseTransformImplExt::parent_query(self, direction, query)
        }

        fn submit_input_buffer(
            &self,
            is_discont: bool,
            inbuf: gst::Buffer,
        ) -> Result<gst::FlowSuccess, gst::FlowError> {
            let settings = self.settings.lock().unwrap();
            let payload = match &settings.payload {
                Some(p) => p.clone(),
                None => {
                    gst::error!(
                        gst::CAT_DEFAULT,
                        obj = self.obj(),
                        "Missing or invalid payload property"
                    );
                    return Err(gst::FlowError::Error);
                }
            };

            if !settings.frame_size.is_power_of_two() {
                gst::error!(
                    gst::CAT_DEFAULT,
                    obj = self.obj(),
                    "frame-size must be a power of two"
                );
                return Err(gst::FlowError::Error);
            }

            let map = inbuf.map_readable().map_err(|_| gst::FlowError::Error)?;
            let data = map.as_slice();
            if data.len() % 4 != 0 {
                gst::error!(
                    gst::CAT_PERFORMANCE,
                    obj = self.obj(),
                    "Buffer size not aligned to f32"
                );
                return Err(gst::FlowError::Error);
            }

            let sample_count = data.len() / 4;
            let channels = settings.channels as usize;
            if channels == 0 || sample_count % channels != 0 {
                gst::error!(
                    gst::CAT_DEFAULT,
                    obj = self.obj(),
                    "Invalid channel count or buffer size (samples={sample_count}, channels={})",
                    settings.channels
                );
                return Err(gst::FlowError::Error);
            }

            let samples: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, sample_count) };

            let mut state = self.buffer_state.lock().unwrap();
            if is_discont {
                state.embedders.clear();
                state.processed.clear();
                state.pending.clear();
                state.initialized = false;
                state.offset_remaining_frames = 0;
            }

            if !state.initialized {
                let config = WatermarkConfig {
                    sample_rate: settings.sample_rate,
                    strength: settings.strength,
                    frame_size: settings.frame_size,
                    num_bin_pairs: settings.num_bin_pairs,
                    min_freq_hz: settings.min_freq_hz,
                    max_freq_hz: settings.max_freq_hz,
                    bin_spacing: settings.bin_spacing,
                };
                let channels = settings.channels as usize;
                let mut embedders = Vec::with_capacity(channels);
                for _ in 0..channels {
                    match StreamEmbedder::new(&payload, &settings.key, &config) {
                        Ok(embedder) => embedders.push(embedder),
                        Err(err) => {
                            gst::error!(
                                gst::CAT_DEFAULT,
                                obj = self.obj(),
                                "Failed to create StreamEmbedder: {err}"
                            );
                            return Err(gst::FlowError::Error);
                        }
                    }
                }
                let offset_frames =
                    (settings.offset_seconds * settings.sample_rate as f32).round() as usize;
                state.offset_remaining_frames = offset_frames;
                state.embedders = embedders;
                state.initialized = true;
            }

            let frames_in_buf = sample_count / channels;
            let mut embed_start_frame = 0usize;
            if state.offset_remaining_frames > 0 {
                let skip_frames = state.offset_remaining_frames.min(frames_in_buf);
                let skip_samples = skip_frames * channels;
                state
                    .processed
                    .extend(samples[..skip_samples].iter().copied());
                state.offset_remaining_frames -= skip_frames;
                embed_start_frame = skip_frames;
            }

            if embed_start_frame < frames_in_buf {
                let embed_samples = &samples[embed_start_frame * channels..];
                let embed_frames = frames_in_buf - embed_start_frame;
                let mut channel_inputs: Vec<Vec<f32>> = (0..channels)
                    .map(|_| Vec::with_capacity(embed_frames))
                    .collect();
                for frame_idx in 0..embed_frames {
                    for ch in 0..channels {
                        let idx = frame_idx * channels + ch;
                        channel_inputs[ch].push(embed_samples[idx]);
                    }
                }

                let mut channel_outputs: Vec<Vec<f32>> = Vec::with_capacity(channels);
                for ch in 0..channels {
                    channel_outputs.push(state.embedders[ch].process(&channel_inputs[ch]));
                }

                let out_frames = channel_outputs.iter().map(|v| v.len()).min().unwrap_or(0);
                if out_frames > 0 {
                    if channel_outputs.iter().any(|v| v.len() != out_frames) {
                        gst::warning!(
                            gst::CAT_DEFAULT,
                            obj = self.obj(),
                            "Channel output length mismatch; truncating to {out_frames} frames"
                        );
                    }
                    for i in 0..out_frames {
                        for ch in 0..channels {
                            state.processed.push_back(channel_outputs[ch][i]);
                        }
                    }
                }
            }

            // Always create PendingBuf matching input buffer size so downstream
            // sees the same buffer rhythm as upstream (even flow).
            state.pending.push_back(PendingBuf {
                size_samples: sample_count,
                pts: inbuf.pts(),
                duration: inbuf.duration(),
                flags: inbuf.flags(),
                offset: Some(inbuf.offset()),
                offset_end: Some(inbuf.offset_end()),
            });

            gst::debug!(
                gst::CAT_DEFAULT,
                obj = self.obj(),
                "submit_input_buffer: in_samples={}, pending_bufs={}, processed_samples={}, offset_remaining_frames={}",
                sample_count,
                state.pending.len(),
                state.processed.len(),
                state.offset_remaining_frames
            );

            Ok(gst::FlowSuccess::Ok)
        }

        fn generate_output(&self) -> Result<GenerateOutputSuccess, gst::FlowError> {
            let mut state = self.buffer_state.lock().unwrap();
            let next = match state.pending.front() {
                Some(p) => p.clone(),
                None => {
                    gst::debug!(
                        gst::CAT_DEFAULT,
                        obj = self.obj(),
                        "generate_output: no pending buffers"
                    );
                    return Ok(GenerateOutputSuccess::NoOutput);
                }
            };

            if state.processed.len() < next.size_samples {
                gst::debug!(
                    gst::CAT_DEFAULT,
                    obj = self.obj(),
                    "generate_output: waiting (processed_samples={}, needed={})",
                    state.processed.len(),
                    next.size_samples
                );
                return Ok(GenerateOutputSuccess::NoOutput);
            }

            let mut outbuf =
                gst::Buffer::with_size(next.size_samples * 4).map_err(|_| gst::FlowError::Error)?;
            {
                let outbuf_mut = outbuf.get_mut().unwrap();
                let mut map = outbuf_mut
                    .map_writable()
                    .map_err(|_| gst::FlowError::Error)?;
                let out_data = map.as_mut_slice();
                let out_samples: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_data.as_mut_ptr() as *mut f32,
                        next.size_samples,
                    )
                };
                for sample in out_samples.iter_mut().take(next.size_samples) {
                    *sample = state.processed.pop_front().unwrap();
                }
                drop(map);
                outbuf_mut.set_flags(next.flags);
                outbuf_mut.set_pts(next.pts);
                outbuf_mut.set_duration(next.duration);
                if let Some(off) = next.offset {
                    outbuf_mut.set_offset(off);
                }
                if let Some(off) = next.offset_end {
                    outbuf_mut.set_offset_end(off);
                }
            }

            state.pending.pop_front();
            gst::debug!(
                gst::CAT_DEFAULT,
                obj = self.obj(),
                "generate_output: produced buffer (samples={})",
                next.size_samples
            );
            Ok(GenerateOutputSuccess::Buffer(outbuf))
        }
    }
}

glib::wrapper! {
    pub struct AguaEmbed(ObjectSubclass<imp::AguaEmbed>)
        @extends gst_audio::AudioFilter, gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "aguaembed",
        gst::Rank::NONE,
        AguaEmbed::static_type(),
    )
}
