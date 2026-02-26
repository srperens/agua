use crate::codec;
use crate::config::WatermarkConfig;
use crate::embed::DetectionResult;
use crate::fft::FftProcessor;
use crate::frame::hann_window;
use crate::key::WatermarkKey;
use crate::patchwork;
use crate::payload::Payload;
use crate::sync::{self, generate_sync_pattern};

/// Streaming watermark embedder.
///
/// Accepts arbitrary-length input chunks and produces watermarked output.
/// Internally buffers to maintain frame alignment with 50% overlap
/// using Hann-windowed overlap-add. The Hann analysis window at 50%
/// overlap satisfies the COLA condition (sum = 1.0) for perfect
/// reconstruction.
pub struct StreamEmbedder {
    config: WatermarkConfig,
    key: WatermarkKey,
    fft: FftProcessor,
    block_bits: Vec<bool>,
    window: Vec<f32>,
    input_buf: Vec<f32>,
    /// Overlap buffer holding the tail of the previous frame's output
    /// (frame_size - hop_size samples that overlap with the next frame).
    overlap_buf: Vec<f32>,
    frame_counter: usize,
    /// Whether we have processed at least one frame (needed for output logic).
    first_frame_done: bool,
}

impl StreamEmbedder {
    /// Create a new streaming embedder.
    pub fn new(
        payload: &Payload,
        key: &WatermarkKey,
        config: &WatermarkConfig,
    ) -> crate::error::Result<Self> {
        let sync_pattern = generate_sync_pattern(key);
        let data_bits = payload.encode_with_crc();
        let coded_bits = codec::encode(&data_bits);

        let block_bits: Vec<bool> = sync_pattern
            .iter()
            .chain(coded_bits.iter())
            .copied()
            .collect();

        let fft = FftProcessor::new(config.frame_size)?;
        let window = hann_window(config.frame_size);
        let hop_size = config.hop_size();

        Ok(Self {
            config: config.clone(),
            key: key.clone(),
            fft,
            block_bits,
            window,
            input_buf: Vec::new(),
            overlap_buf: vec![0.0f32; config.frame_size - hop_size],
            frame_counter: 0,
            first_frame_done: false,
        })
    }

    /// Process input samples and return watermarked output.
    ///
    /// Uses Hann-windowed overlap-add with 50% overlap. Each call may
    /// produce fewer samples than the input if buffering is needed.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.input_buf.extend_from_slice(input);
        let frame_size = self.config.frame_size;
        let hop_size = self.config.hop_size();
        let mut result = Vec::new();

        while self.input_buf.len() >= frame_size {
            // Extract frame_size samples from input buffer (advance by hop_size later)
            let mut buf = self.input_buf[..frame_size].to_vec();

            // Apply Hann analysis window
            for (i, s) in buf.iter_mut().enumerate() {
                *s *= self.window[i];
            }

            // FFT -> embed -> IFFT -> normalize
            self.fft
                .forward(&mut buf)
                .expect("buffer size matches frame_size");

            let bit_idx = self.frame_counter % self.block_bits.len();
            let bit = self.block_bits[bit_idx];
            let bin_pair_seed = sync::bin_pair_seed(bit_idx);
            patchwork::embed_frame(
                self.fft.freq_bins_mut(),
                bit,
                &self.key,
                bin_pair_seed,
                &self.config,
            );

            self.fft
                .inverse(&mut buf)
                .expect("buffer size matches frame_size");
            self.fft.normalize(&mut buf);

            // Overlap-add WITHOUT synthesis window (analysis-only COLA = 1.0).
            let mut frame_output = vec![0.0f32; frame_size];

            // Add overlap from previous frame
            for (i, &ov) in self.overlap_buf.iter().enumerate() {
                frame_output[i] = ov;
            }

            // Add current frame
            for (i, &s) in buf.iter().enumerate() {
                frame_output[i] += s;
            }

            // Output the first hop_size samples (they are complete)
            if self.first_frame_done {
                result.extend_from_slice(&frame_output[..hop_size]);
            } else {
                result.extend_from_slice(&frame_output[..hop_size]);
                self.first_frame_done = true;
            }

            // Save the remaining samples as overlap for the next frame
            self.overlap_buf
                .copy_from_slice(&frame_output[hop_size..frame_size]);

            // Advance input by hop_size (not frame_size)
            self.input_buf.drain(..hop_size);
            self.frame_counter += 1;
        }

        result
    }

    /// Flush remaining buffered samples.
    ///
    /// Returns the overlap buffer (which contains the tail of the last
    /// processed frame) plus any unprocessed input samples.
    pub fn flush(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        // Output the remaining overlap
        out.extend_from_slice(&self.overlap_buf);
        // Output any remaining input that didn't form a full frame
        out.append(&mut self.input_buf);
        // Reset overlap
        for s in self.overlap_buf.iter_mut() {
            *s = 0.0;
        }
        out
    }
}

/// Streaming watermark detector with sliding-window sync search.
///
/// Accepts arbitrary-length input chunks and reports detected watermarks.
/// Internally buffers raw audio samples and periodically runs batch
/// detection, delegating to the one-shot `detect()` function which handles:
/// 1. Arbitrary block boundary (sync search via sliding window)
/// 2. Arbitrary sub-frame sample alignment (multi-offset search)
///
/// This enables detection when the mic starts recording at any point
/// in the watermarked stream (e.g. speaker → air → mic path).
pub struct StreamDetector {
    config: WatermarkConfig,
    key: WatermarkKey,
    frames_per_block: usize,
    /// Raw audio sample buffer.
    audio_buf: Vec<f32>,
    /// Maximum audio buffer size in samples.
    max_buf_samples: usize,
    /// Samples needed for one full block of frames.
    block_samples: usize,
    /// Total samples consumed (for offset reporting).
    total_samples_consumed: usize,
    /// Diagnostics: number of detection attempts so far.
    diag_detect_attempts: u32,
    /// Diagnostics: best sync correlation seen in last detection attempt.
    diag_best_sync_corr: f32,
    /// Diagnostics: number of sync candidates above threshold in last attempt.
    diag_sync_candidates: u32,
}

impl StreamDetector {
    /// Create a new streaming detector.
    pub fn new(key: &WatermarkKey, config: &WatermarkConfig) -> crate::error::Result<Self> {
        // Validate config by constructing an FftProcessor (checks frame_size).
        let _ = FftProcessor::new(config.frame_size)?;
        let frames_per_block = sync::frames_per_block();
        let hop_size = config.hop_size();
        // Samples needed to produce frames_per_block frames.
        let block_samples = (frames_per_block - 1) * hop_size + config.frame_size;
        // Buffer up to 2 blocks to allow sync search.
        let max_buf_samples = 2 * block_samples;

        Ok(Self {
            config: config.clone(),
            key: key.clone(),
            frames_per_block,
            audio_buf: Vec::new(),
            max_buf_samples,
            block_samples,
            total_samples_consumed: 0,
            diag_detect_attempts: 0,
            diag_best_sync_corr: 0.0,
            diag_sync_candidates: 0,
        })
    }

    /// Process input samples and return any detected watermarks.
    pub fn process(&mut self, input: &[f32]) -> Vec<DetectionResult> {
        self.audio_buf.extend_from_slice(input);

        let mut results = Vec::new();
        // Try detection when we have at least 1.25 blocks of audio.
        let scan_threshold = self.block_samples + self.block_samples / 4;
        while self.audio_buf.len() >= scan_threshold {
            if let Some(result) = self.try_detect_batch() {
                results.push(result);
            } else {
                // No detection — advance by the scan window size so the
                // next attempt covers new frame positions. The scan window
                // is (num_frames - frames_per_block) frames.
                let hop_size = self.config.hop_size();
                let num_frames =
                    (self.audio_buf.len() - self.config.frame_size) / hop_size + 1;
                let scanned_frames = num_frames.saturating_sub(self.frames_per_block);
                let advance = (scanned_frames * hop_size).max(hop_size);
                let drain = advance.min(self.audio_buf.len());
                self.audio_buf.drain(..drain);
                self.total_samples_consumed += drain;
                break;
            }
        }

        // Trim buffer if it exceeds max size
        if self.audio_buf.len() > self.max_buf_samples {
            let excess = self.audio_buf.len() - self.max_buf_samples;
            self.audio_buf.drain(..excess);
            self.total_samples_consumed += excess;
        }

        results
    }

    /// Run one-shot detection on the audio buffer. The `detect()` function
    /// handles both sync search and sub-frame alignment internally.
    fn try_detect_batch(&mut self) -> Option<DetectionResult> {
        let hop_size = self.config.hop_size();
        self.diag_detect_attempts += 1;

        // Use single-offset detection (offset 0) for speed. The streaming
        // drain mechanism provides offset diversity over time, so we don't
        // need the expensive multi-offset search that detect() does.
        // This avoids 16× FFT passes and hundreds of Viterbi decodes that
        // would freeze WASM for minutes.
        let (result, diag) = crate::detect::detect_single_offset_with_diagnostics(
            &self.audio_buf,
            &self.key,
            &self.config,
        );
        self.diag_best_sync_corr = diag.best_sync_corr;
        self.diag_sync_candidates = diag.sync_candidates;

        if let Ok(det_results) = result
            && let Some(result) = det_results.into_iter().next()
        {
            // Drain buffer past the detected block
            let block_end_frames = result.offset + self.frames_per_block;
            let block_end_samples = (block_end_frames - 1) * hop_size + self.config.frame_size;
            let drain = block_end_samples.min(self.audio_buf.len());
            self.audio_buf.drain(..drain);
            self.total_samples_consumed += drain;

            return Some(DetectionResult {
                payload: result.payload,
                confidence: result.confidence,
                offset: self.total_samples_consumed,
            });
        }
        None
    }

    /// Finalize detection with any remaining buffered data.
    pub fn finalize(&mut self) -> Option<DetectionResult> {
        if self.audio_buf.len() >= self.block_samples {
            self.try_detect_batch()
        } else {
            None
        }
    }

    /// How full the detection buffer is, as a fraction (0.0 to 1.0).
    ///
    /// Returns the ratio of buffered audio to the samples needed for one
    /// complete watermark block.
    pub fn buffer_fill(&self) -> f32 {
        (self.audio_buf.len() as f32 / self.block_samples as f32).min(1.0)
    }

    /// Number of detection attempts so far.
    pub fn diag_detect_attempts(&self) -> u32 {
        self.diag_detect_attempts
    }

    /// Best sync correlation found in the last detection attempt.
    pub fn diag_best_sync_corr(&self) -> f32 {
        self.diag_best_sync_corr
    }

    /// Number of sync candidates above threshold in the last detection attempt.
    pub fn diag_sync_candidates(&self) -> u32 {
        self.diag_sync_candidates
    }

    /// Number of samples currently buffered.
    pub fn diag_buffer_samples(&self) -> usize {
        self.audio_buf.len()
    }

    /// Reset the detector state, clearing all buffered data.
    pub fn reset(&mut self) {
        self.audio_buf.clear();
        self.total_samples_consumed = 0;
        self.diag_detect_attempts = 0;
        self.diag_best_sync_corr = 0.0;
        self.diag_sync_candidates = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_audio(num_samples: usize, sample_rate: u32) -> Vec<f32> {
        let mut samples = vec![0.0f32; num_samples];
        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f32 / sample_rate as f32;
            for k in 1u32..80 {
                let freq = k as f32 * 60.0;
                let amp = 1.0 / (k as f32).sqrt();
                *sample += amp * (2.0 * std::f32::consts::PI * freq * t + k as f32).sin();
            }
        }
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 0.0 {
            for s in samples.iter_mut() {
                *s *= 0.5 / peak;
            }
        }
        samples
    }

    #[test]
    fn stream_embedder_produces_output() {
        let config = WatermarkConfig::default();
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([0xAA; 16]);

        let mut embedder = StreamEmbedder::new(&payload, &key, &config).unwrap();
        let audio = make_test_audio(48000, config.sample_rate);

        let mut output = Vec::new();
        for chunk in audio.chunks(4096) {
            output.extend(embedder.process(chunk));
        }
        output.extend(embedder.flush());

        assert!(!output.is_empty(), "streaming embedder produced no output");
    }

    /// Helper: embed a watermark using one-shot API and return the watermarked audio.
    fn embed_test_audio(
        config: &WatermarkConfig,
        key: &WatermarkKey,
        payload: &Payload,
        duration_secs: usize,
    ) -> Vec<f32> {
        let num_samples = config.sample_rate as usize * duration_secs;
        let mut audio = make_test_audio(num_samples, config.sample_rate);
        crate::embed::embed(&mut audio, payload, key, config).unwrap();
        audio
    }

    #[test]
    fn stream_detector_with_offset() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC,
            0xBA, 0x98,
        ]);

        // Embed into 30s of audio (enough for 2+ blocks)
        let watermarked = embed_test_audio(&config, &key, &payload, 30);

        // Start detection from ~3 seconds into the watermarked audio.
        // This simulates a mic starting to capture an already-playing
        // watermarked stream at an arbitrary sample offset.
        let offset = 48000 * 3; // 144000 samples — NOT hop-aligned
        let offset_audio = &watermarked[offset..];

        // Feed in small 128-sample chunks to StreamDetector
        let mut detector = StreamDetector::new(&key, &config).unwrap();
        let mut results = Vec::new();
        for chunk in offset_audio.chunks(128) {
            results.extend(detector.process(chunk));
        }
        if let Some(r) = detector.finalize() {
            results.push(r);
        }

        assert!(
            !results.is_empty(),
            "StreamDetector failed to detect watermark with offset"
        );
        assert_eq!(results[0].payload, payload, "payload mismatch");
    }

    #[test]
    fn stream_detector_continuous_blocks() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22,
            0x33, 0x44,
        ]);

        // 30s should contain ~2 full blocks
        let watermarked = embed_test_audio(&config, &key, &payload, 30);

        let mut detector = StreamDetector::new(&key, &config).unwrap();
        let mut results = Vec::new();
        for chunk in watermarked.chunks(4096) {
            results.extend(detector.process(chunk));
        }
        if let Some(r) = detector.finalize() {
            results.push(r);
        }

        assert!(
            !results.is_empty(),
            "StreamDetector failed to detect in 30s"
        );
        for r in &results {
            assert_eq!(
                r.payload, payload,
                "payload mismatch in continuous detection"
            );
        }
    }

    #[test]
    fn stream_detector_bounded_memory() {
        let config = WatermarkConfig::default();
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();

        // Feed unwatermarked audio for ~40s (several block durations)
        let audio = make_test_audio(48000 * 40, config.sample_rate);

        let mut detector = StreamDetector::new(&key, &config).unwrap();
        for chunk in audio.chunks(4096) {
            let _ = detector.process(chunk);
        }

        // Buffer should be bounded to max_buf_samples
        assert!(
            detector.audio_buf.len() <= detector.max_buf_samples,
            "buffer grew unbounded: {} > {}",
            detector.audio_buf.len(),
            detector.max_buf_samples
        );
    }

    #[test]
    fn stream_detector_matches_batch() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([0x55; 16]);

        let watermarked = embed_test_audio(&config, &key, &payload, 20);

        // One-shot detection
        let batch_results = crate::detect::detect(&watermarked, &key, &config).unwrap();
        assert!(!batch_results.is_empty());

        // Streaming detection
        let mut detector = StreamDetector::new(&key, &config).unwrap();
        let mut stream_results = Vec::new();
        for chunk in watermarked.chunks(4096) {
            stream_results.extend(detector.process(chunk));
        }
        if let Some(r) = detector.finalize() {
            stream_results.push(r);
        }

        assert!(
            !stream_results.is_empty(),
            "streaming detector found nothing"
        );
        assert_eq!(
            batch_results[0].payload, stream_results[0].payload,
            "batch and streaming payloads differ"
        );
    }
}
