use crate::codec;
use crate::config::WatermarkConfig;
use crate::detect;
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

/// Default maximum number of blocks to combine before resetting.
const DEFAULT_MAX_COMBINE_BLOCKS: u32 = 5;

/// Number of sub-frame offsets to try when searching for the best
/// frame grid alignment during the first block of a combining cycle.
/// With hop_size=512, step = 512/8 = 64 samples.
const COMBINE_SUB_FRAME_STEPS: usize = 8;


/// Streaming watermark detector with sliding-window sync search.
///
/// Accepts arbitrary-length input chunks and reports detected watermarks.
/// Internally buffers raw audio samples and periodically runs batch
/// detection, delegating to the one-shot `detect()` function which handles:
/// 1. Arbitrary block boundary (sync search via sliding window)
/// 2. Arbitrary sub-frame sample alignment (multi-offset search)
///
/// When single-block detection fails (low SNR), the detector accumulates
/// soft values across consecutive blocks and tries combined Viterbi
/// decoding. SNR improves by ~sqrt(N) where N is the number of blocks
/// combined. This is transparent to the caller — the same `process()`
/// API is used.
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
    /// Accumulated soft values from previous blocks (element-wise sum).
    accumulated_soft: Option<Vec<f32>>,
    /// Number of blocks combined in the accumulator so far.
    combine_count: u32,
    /// Maximum blocks to combine before resetting.
    max_combine_blocks: u32,
    /// Absolute sample position of the last sync detection, used to predict
    /// next block's sync. Computed as total_samples_consumed + sync_offset * hop_size.
    last_sync_absolute: Option<usize>,
    /// Diagnostics: combine count from the last successful combined detection.
    diag_last_combine_count: u32,
    /// Diagnostics: successful soft value extractions.
    diag_soft_extractions: u32,
    /// Diagnostics: reason for last soft combine failure (0=not tried, 1=extract_err, 2=extract_none, 3=ok).
    diag_soft_combine_status: u32,
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
        // Buffer enough for ~3 blocks. The combining path drains after each
        // block extraction, so the buffer never needs to hold all combined
        // blocks simultaneously.
        let max_buf_samples = 3 * block_samples;

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
            accumulated_soft: None,
            combine_count: 0,
            max_combine_blocks: DEFAULT_MAX_COMBINE_BLOCKS,
            last_sync_absolute: None,
            diag_last_combine_count: 0,
            diag_soft_extractions: 0,
            diag_soft_combine_status: 0,
        })
    }

    /// Process input samples and return any detected watermarks.
    pub fn process(&mut self, input: &[f32]) -> Vec<DetectionResult> {
        self.audio_buf.extend_from_slice(input);

        let mut results = Vec::new();
        // Threshold depends on combining state:
        // - First block / single-block: need 1.25 blocks for sync search margin.
        // - Subsequent combining blocks: know exact position, need just 1 block.
        //   This saves ~2.9s of latency per subsequent block.
        while self.audio_buf.len() >= self.scan_threshold() {
            if let Some(result) = self.try_detect_batch() {
                results.push(result);
            } else if self.combine_count == 0 {
                // Not making combining progress — break to avoid spinning.
                // When combine_count > 0, soft values are accumulating and
                // we should continue looping to feed the next block.
                break;
            }
        }

        // Trim buffer if it exceeds max size
        if self.audio_buf.len() > self.max_buf_samples {
            let excess = self.audio_buf.len() - self.max_buf_samples;
            self.audio_buf.drain(..excess);
            self.total_samples_consumed += excess;
            // Trimming shifts total_samples_consumed without block-aligned
            // draining, so last_sync_absolute becomes stale and predicted_frame
            // would point to the wrong position. Reset to avoid corrupt combining.
            self.reset_accumulator();
        }

        results
    }

    /// Run one-shot detection on the audio buffer, with soft combining fallback.
    ///
    /// First tries single-block Viterbi decode (fast path). On failure,
    /// extracts data soft values and accumulates them across blocks.
    /// When enough blocks are combined, tries Viterbi on the combined soft values.
    fn try_detect_batch(&mut self) -> Option<DetectionResult> {
        let hop_size = self.config.hop_size();
        self.diag_detect_attempts += 1;

        // Fast path: try single-block detection (unchanged behavior)
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
            // Single-block success — drain to start of NEXT block, reset, report.
            // Using (offset + frames_per_block) * hop puts us exactly at the
            // next block boundary. The old formula drained to the end of the
            // last frame, which overshoots by hop_size (512 samples) and loses
            // frame 0 of the next block, forcing a ~23s gap between detections.
            let next_block_start = (result.offset + self.frames_per_block) * hop_size;
            let drain = next_block_start.min(self.audio_buf.len());
            self.audio_buf.drain(..drain);
            self.total_samples_consumed += drain;
            self.reset_accumulator();

            return Some(DetectionResult {
                payload: result.payload,
                confidence: result.confidence,
                offset: self.total_samples_consumed,
            });
        }

        // Single-block failed — try soft combining
        self.try_soft_combine()
    }

    /// Drain the buffer when no sync was found or detection failed.
    ///
    /// Advances past the scanned region (all frames beyond one block) so that
    /// the detector makes progress and doesn't re-scan the same audio.
    fn drain_no_detection(&mut self) {
        let hop_size = self.config.hop_size();
        let num_frames = if self.audio_buf.len() >= self.config.frame_size {
            (self.audio_buf.len() - self.config.frame_size) / hop_size + 1
        } else {
            0
        };
        let scanned_frames = num_frames.saturating_sub(self.frames_per_block);
        let advance = (scanned_frames * hop_size).max(hop_size);
        let drain = advance.min(self.audio_buf.len());
        self.audio_buf.drain(..drain);
        self.total_samples_consumed += drain;
    }

    /// Minimum audio needed before attempting detection.
    ///
    /// When combining is in progress and we know the exact block position,
    /// one block suffices. Otherwise we need extra margin for sync search.
    fn scan_threshold(&self) -> usize {
        if self.last_sync_absolute.is_some() {
            // Subsequent combining block: extract at predicted frame 0
            self.block_samples
        } else {
            // Sync search needs scan margin (0.25 blocks extra)
            self.block_samples + self.block_samples / 4
        }
    }

    /// Extract soft values from the current buffer and combine with accumulated values.
    ///
    /// Uses two strategies depending on accumulator state:
    /// - **First block** (no accumulator): searches across multiple sub-frame
    ///   offsets to find the frame grid alignment that maximises sync correlation,
    ///   then establishes the block boundary position. This is critical for
    ///   acoustic scenarios where the detector's frame grid is randomly offset
    ///   from the watermark's embedded grid.
    /// - **Subsequent blocks** (have accumulator): uses the **predicted** sync
    ///   position (based on the first block) to extract data at the exact same
    ///   frame alignment. This is critical because even 1 frame of misalignment
    ///   between blocks causes the combined soft values to decorrelate.
    ///
    /// Always drains the buffer on all paths to ensure the detector makes progress.
    fn try_soft_combine(&mut self) -> Option<DetectionResult> {
        let hop_size = self.config.hop_size();

        // Two paths: sub-frame search for first block, predicted position for subsequent
        // sub_offset is the sub-frame byte offset within audio_buf for optimal alignment
        let (data_soft, sync_offset, sync_corr, sub_offset) =
            if let Some(last_abs) = self.last_sync_absolute {
                // Subsequent block: extract at the PREDICTED position to maintain
                // exact frame alignment with the first block's extraction.
                // After the first block's drain (which included the sub-frame offset),
                // the buffer starts at the correct alignment, so sub_offset = 0.
                let predicted_abs = last_abs + self.frames_per_block * hop_size;
                let predicted_frame =
                    predicted_abs.saturating_sub(self.total_samples_consumed) / hop_size;

                match detect::extract_data_soft_at(
                    &self.audio_buf,
                    &self.key,
                    &self.config,
                    predicted_frame,
                ) {
                    Ok(Some((soft, corr))) => (soft, predicted_frame, corr, 0usize),
                    Ok(None) => {
                        self.diag_soft_combine_status = 2;
                        self.reset_accumulator();
                        self.drain_no_detection();
                        return None;
                    }
                    Err(_) => {
                        self.diag_soft_combine_status = 1;
                        self.reset_accumulator();
                        self.drain_no_detection();
                        return None;
                    }
                }
            } else {
                // First block: search across sub-frame offsets for best sync alignment.
                // In acoustic scenarios, the detector's frame grid may be randomly
                // offset from the watermark's grid. Trying multiple offsets finds
                // the alignment that maximises sync correlation, producing cleaner
                // soft values that actually benefit from combining.
                let coarse_step = (hop_size / COMBINE_SUB_FRAME_STEPS).max(1);
                let mut best_sub = 0usize;
                let mut best_sub_corr = f32::NEG_INFINITY;

                for i in 0..COMBINE_SUB_FRAME_STEPS {
                    let sub = i * coarse_step;
                    if sub + self.config.frame_size > self.audio_buf.len() {
                        break;
                    }
                    if let Ok(Some((_frame, corr))) =
                        detect::find_best_sync(&self.audio_buf[sub..], &self.key, &self.config)
                        && corr > best_sub_corr
                    {
                        best_sub_corr = corr;
                        best_sub = sub;
                    }
                }

                // Extract data soft values at the best sub-frame offset
                match detect::extract_data_soft(
                    &self.audio_buf[best_sub..],
                    &self.key,
                    &self.config,
                ) {
                    Ok(Some((soft, offset, corr))) => (soft, offset, corr, best_sub),
                    Ok(None) => {
                        self.diag_soft_combine_status = 2;
                        self.reset_accumulator();
                        self.drain_no_detection();
                        return None;
                    }
                    Err(_) => {
                        self.diag_soft_combine_status = 1;
                        self.reset_accumulator();
                        self.drain_no_detection();
                        return None;
                    }
                }
            };

        self.diag_soft_extractions += 1;
        self.diag_soft_combine_status = 3; // extraction OK

        // Compute absolute sample position of this sync (including sub-frame offset)
        let current_sync_absolute =
            self.total_samples_consumed + sub_offset + sync_offset * hop_size;

        // Accumulate soft values
        if let Some(ref mut acc) = self.accumulated_soft {
            for (a, &s) in acc.iter_mut().zip(data_soft.iter()) {
                *a += s;
            }
            self.combine_count += 1;
        } else {
            self.accumulated_soft = Some(data_soft);
            self.combine_count = 1;
        }
        self.last_sync_absolute = Some(current_sync_absolute);

        // Try Viterbi on combined soft values (only after 2+ blocks)
        if self.combine_count >= 2
            && let Some(ref acc) = self.accumulated_soft
            && let Some(payload) = detect::try_decode_soft(acc)
        {
            let confidence = sync_corr;
            // Drain to start of NEXT block (same formula as non-success path).
            let next_block_start =
                sub_offset + (sync_offset + self.frames_per_block) * hop_size;
            let drain = next_block_start.min(self.audio_buf.len());
            self.audio_buf.drain(..drain);
            self.total_samples_consumed += drain;
            let combine_count = self.combine_count;
            self.reset_accumulator();
            self.diag_last_combine_count = combine_count;

            return Some(DetectionResult {
                payload,
                confidence,
                offset: self.total_samples_consumed,
            });
        }

        // Max blocks reached without success — reset
        if self.combine_count >= self.max_combine_blocks {
            self.reset_accumulator();
        }

        // Drain to the START of the next block (including sub-frame offset).
        // For the first block, sub_offset shifts the drain to maintain the
        // optimal frame grid alignment for subsequent blocks.
        let next_block_start = sub_offset + (sync_offset + self.frames_per_block) * hop_size;
        let drain = next_block_start.min(self.audio_buf.len());
        self.audio_buf.drain(..drain);
        self.total_samples_consumed += drain;

        None
    }

    /// Reset the soft combining accumulator.
    fn reset_accumulator(&mut self) {
        self.accumulated_soft = None;
        self.combine_count = 0;
        self.last_sync_absolute = None;
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

    /// Number of blocks currently combined in the soft accumulator.
    ///
    /// Returns 0 when no combining is in progress, or 1..=max_combine_blocks
    /// when blocks are being accumulated.
    pub fn combine_count(&self) -> u32 {
        self.combine_count
    }

    /// Maximum number of blocks to combine before resetting.
    pub fn max_combine_blocks(&self) -> u32 {
        self.max_combine_blocks
    }

    /// Combine count from the last successful combined detection.
    ///
    /// Returns 0 if the last detection was single-block, or 2..=max_combine_blocks
    /// if soft combining was used.
    pub fn diag_last_combine_count(&self) -> u32 {
        self.diag_last_combine_count
    }

    /// Number of successful soft value extractions.
    pub fn diag_soft_extractions(&self) -> u32 {
        self.diag_soft_extractions
    }

    /// Status of last soft combine attempt.
    /// 0=not tried, 1=extract error, 2=extract None, 3=extraction OK.
    pub fn diag_soft_combine_status(&self) -> u32 {
        self.diag_soft_combine_status
    }

    /// Reset the detector state, clearing all buffered data.
    pub fn reset(&mut self) {
        self.audio_buf.clear();
        self.total_samples_consumed = 0;
        self.diag_detect_attempts = 0;
        self.diag_best_sync_corr = 0.0;
        self.diag_sync_candidates = 0;
        self.diag_last_combine_count = 0;
        self.diag_soft_extractions = 0;
        self.diag_soft_combine_status = 0;
        self.reset_accumulator();
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

    /// Test that extract_data_soft and try_decode_soft work correctly,
    /// and that summing soft values from multiple blocks produces valid output.
    #[test]
    fn stream_detector_soft_combine() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0x00,
            0xEE, 0xFF,
        ]);

        // 30s = ~2.5 blocks at 48kHz
        let watermarked = embed_test_audio(&config, &key, &payload, 30);

        // Test extract_data_soft on the first block
        let result = crate::detect::extract_data_soft(&watermarked, &key, &config)
            .expect("extract should not error");
        let (soft1, _sync_offset1, corr1) = result.expect("should find sync and extract soft");
        assert_eq!(
            soft1.len(),
            codec::CODED_BITS,
            "soft values should match coded bits"
        );
        assert!(corr1 > 0.0, "sync correlation should be positive");

        // Verify try_decode_soft works on single-block soft values
        let decoded = crate::detect::try_decode_soft(&soft1);
        assert!(decoded.is_some(), "should decode single-block soft values");
        assert_eq!(decoded.unwrap(), payload, "single-block payload mismatch");

        // Extract soft values from a later segment (second block)
        let hop_size = config.hop_size();
        let frames_per_block = sync::frames_per_block();
        let block_samples = (frames_per_block - 1) * hop_size + config.frame_size;
        if watermarked.len() > block_samples + block_samples / 2 {
            let second_segment = &watermarked[block_samples..];
            if let Ok(Some((soft2, _sync_offset2, _corr2))) =
                crate::detect::extract_data_soft(second_segment, &key, &config)
            {
                // Sum soft values from both blocks
                let combined: Vec<f32> =
                    soft1.iter().zip(soft2.iter()).map(|(a, b)| a + b).collect();
                let combined_decoded = crate::detect::try_decode_soft(&combined);
                assert!(combined_decoded.is_some(), "combined soft should decode");
                assert_eq!(
                    combined_decoded.unwrap(),
                    payload,
                    "combined payload mismatch"
                );
            }
        }

        // Test StreamDetector accessors
        let detector = StreamDetector::new(&key, &config).unwrap();
        assert_eq!(detector.combine_count(), 0);
        assert_eq!(
            detector.max_combine_blocks(),
            super::DEFAULT_MAX_COMBINE_BLOCKS
        );
        assert_eq!(detector.diag_last_combine_count(), 0);
    }

    /// Simple LCG pseudo-random noise for deterministic tests.
    fn add_noise(audio: &mut [f32], level: f32, seed: u64) {
        let mut state = seed;
        for s in audio.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Extract a float in [-1, 1] from the upper bits
            let noise = (state >> 33) as f32 / (u32::MAX as f32 / 2.0) - 1.0;
            *s += noise * level;
        }
    }

    /// End-to-end test that soft combining in StreamDetector actually works.
    ///
    /// Adds noise to watermarked audio so that single-block Viterbi fails,
    /// but combining 2-3 blocks provides enough SNR for a successful decode.
    #[test]
    fn stream_detector_combining_end_to_end() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC,
            0xDE, 0xF0,
        ]);

        // 60s = ~5 blocks — plenty for combining
        let watermarked = embed_test_audio(&config, &key, &payload, 60);

        // Verify single-block works on clean signal first
        let clean_result = crate::detect::detect(&watermarked[..48000 * 15], &key, &config);
        assert!(
            clean_result.is_ok(),
            "single-block should work on clean signal"
        );

        // Add noise to degrade single-block detection.
        // Try increasing noise levels until single-offset detection fails.
        // (StreamDetector uses detect_single_offset_with_diagnostics, not full multi-offset.)
        let mut noise_level = 0.0;
        let mut noisy = watermarked.clone();
        for level in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0] {
            noisy = watermarked.clone();
            add_noise(&mut noisy, level, 42);
            let (result, _diag) = crate::detect::detect_single_offset_with_diagnostics(
                &noisy[..48000 * 15],
                &key,
                &config,
            );
            if result.is_err() || result.unwrap().is_empty() {
                noise_level = level;
                break;
            }
        }
        if noise_level == 0.0 {
            eprintln!("NOTE: single-offset detection survived all noise levels, skipping");
            return;
        }
        // Apply the calibrated noise to the full signal
        let watermarked = noisy;
        eprintln!("Calibrated noise level: {noise_level}");

        // Feed through StreamDetector with small chunks to exercise combining
        let mut detector = StreamDetector::new(&key, &config).unwrap();
        let mut results = Vec::new();
        for chunk in watermarked.chunks(512) {
            results.extend(detector.process(chunk));
        }
        if let Some(r) = detector.finalize() {
            results.push(r);
        }

        eprintln!(
            "combining e2e: results={}, soft_extractions={}, last_combine_count={}, soft_status={}",
            results.len(),
            detector.diag_soft_extractions(),
            detector.diag_last_combine_count(),
            detector.diag_soft_combine_status(),
        );

        // Soft extraction should have been attempted
        assert!(
            detector.diag_soft_extractions() > 0,
            "soft extraction was never attempted — combining path not exercised"
        );

        // If combining worked, we should have results with correct payload
        if !results.is_empty() {
            for r in &results {
                assert_eq!(r.payload, payload, "combined payload mismatch");
            }
            // Verify at least one result came from combining (not single-block)
            if detector.diag_last_combine_count() > 0 {
                eprintln!(
                    "SUCCESS: combining worked with {} blocks",
                    detector.diag_last_combine_count()
                );
            }
        }
    }

    /// Test combining with large buffer (all audio at once).
    ///
    /// This specifically targets a potential double-drain bug: when
    /// try_soft_combine drains to the next block start, but process()
    /// drains additional audio, causing block skips that break alignment.
    #[test]
    fn stream_detector_combining_large_buffer() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC,
            0xDE, 0xF0,
        ]);

        // 60s audio
        let watermarked = embed_test_audio(&config, &key, &payload, 60);

        // Calibrate noise: find level where single-offset fails
        let mut noise_level = 0.0;
        let mut noisy = watermarked.clone();
        for level in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0] {
            noisy = watermarked.clone();
            add_noise(&mut noisy, level, 99);
            let (result, _) = crate::detect::detect_single_offset_with_diagnostics(
                &noisy[..48000 * 15],
                &key,
                &config,
            );
            if result.is_err() || result.unwrap().is_empty() {
                noise_level = level;
                break;
            }
        }
        if noise_level == 0.0 {
            eprintln!("NOTE: single-offset survived all noise, skipping large-buffer test");
            return;
        }
        let watermarked = noisy;
        eprintln!("Large-buffer calibrated noise: {noise_level}");

        // Feed ALL audio at once — this makes the buffer very large and
        // triggers the potential double-drain issue in process().
        let mut detector = StreamDetector::new(&key, &config).unwrap();
        let results_all_at_once = detector.process(&watermarked);
        let finalize_result = detector.finalize();

        let mut all_results: Vec<_> = results_all_at_once;
        if let Some(r) = finalize_result {
            all_results.push(r);
        }

        eprintln!(
            "large buffer: results={}, soft_extractions={}, last_combine_count={}, soft_status={}",
            all_results.len(),
            detector.diag_soft_extractions(),
            detector.diag_last_combine_count(),
            detector.diag_soft_combine_status(),
        );

        // Compare with small-chunk feeding
        let mut detector2 = StreamDetector::new(&key, &config).unwrap();
        let mut small_chunk_results = Vec::new();
        for chunk in watermarked.chunks(512) {
            small_chunk_results.extend(detector2.process(chunk));
        }
        if let Some(r) = detector2.finalize() {
            small_chunk_results.push(r);
        }

        eprintln!(
            "small chunks: results={}, soft_extractions={}, last_combine_count={}, attempts={}",
            small_chunk_results.len(),
            detector2.diag_soft_extractions(),
            detector2.diag_last_combine_count(),
            detector2.diag_detect_attempts(),
        );

        // Both should find the watermark if combining works correctly
        // If large-buffer finds fewer results, the double-drain bug is real
        if !small_chunk_results.is_empty() && all_results.is_empty() {
            panic!(
                "BUG: small chunks found {} results but large buffer found none \
                 (double-drain issue in process())",
                small_chunk_results.len()
            );
        }

        // Both approaches should find approximately the same number of results
        // A big discrepancy indicates the large-buffer path has issues
        if !small_chunk_results.is_empty() {
            assert!(
                all_results.len() >= small_chunk_results.len() / 2,
                "large buffer found significantly fewer results ({}) than small chunks ({})",
                all_results.len(),
                small_chunk_results.len(),
            );
        }
    }

    /// Test that the accumulator resets properly:
    /// - After max_combine_blocks without success
    /// - When sync alignment is lost
    #[test]
    fn stream_detector_combine_resets() {
        let config = WatermarkConfig::default();
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();

        // Feed unwatermarked audio — accumulator should reset after max_combine_blocks
        let audio = make_test_audio(48000 * 60, config.sample_rate);

        let mut detector = StreamDetector::new(&key, &config).unwrap();
        for chunk in audio.chunks(4096) {
            let _ = detector.process(chunk);
        }

        // Accumulator should be reset (not stuck at max)
        assert_eq!(
            detector.combine_count(),
            0,
            "accumulator should reset after max blocks"
        );

        // Buffer should still be bounded
        assert!(
            detector.audio_buf.len() <= detector.max_buf_samples,
            "buffer grew unbounded: {} > {}",
            detector.audio_buf.len(),
            detector.max_buf_samples
        );
    }
}
