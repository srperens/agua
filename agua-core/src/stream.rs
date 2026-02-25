use crate::codec;
use crate::config::WatermarkConfig;
use crate::embed::DetectionResult;
use crate::fft::FftProcessor;
use crate::frame::hann_window;
use crate::key::WatermarkKey;
use crate::patchwork;
use crate::payload::Payload;
use crate::sync::{self, SYNC_PATTERN_BITS, correlate_sync, generate_sync_pattern};

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
            patchwork::embed_frame(
                self.fft.freq_bins_mut(),
                bit,
                &self.key,
                self.frame_counter as u32,
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

/// Streaming watermark detector.
///
/// Accepts arbitrary-length input chunks and reports detected watermarks.
/// Uses Hann-windowed overlapping frames (50% overlap) matching the
/// embedding scheme.
pub struct StreamDetector {
    config: WatermarkConfig,
    key: WatermarkKey,
    fft: FftProcessor,
    window: Vec<f32>,
    sync_pattern: Vec<bool>,
    frames_per_block: usize,
    coded_bits_count: usize,
    input_buf: Vec<f32>,
    soft_values: Vec<f32>,
    frame_counter: usize,
}

impl StreamDetector {
    /// Create a new streaming detector.
    pub fn new(key: &WatermarkKey, config: &WatermarkConfig) -> crate::error::Result<Self> {
        let fft = FftProcessor::new(config.frame_size)?;
        let window = hann_window(config.frame_size);
        let sync_pattern = generate_sync_pattern(key);
        let frames_per_block = sync::frames_per_block();
        let coded_bits_count = codec::CODED_BITS;

        Ok(Self {
            config: config.clone(),
            key: key.clone(),
            fft,
            window,
            sync_pattern,
            frames_per_block,
            coded_bits_count,
            input_buf: Vec::new(),
            soft_values: Vec::new(),
            frame_counter: 0,
        })
    }

    /// Process input samples and return any detected watermarks.
    pub fn process(&mut self, input: &[f32]) -> Vec<DetectionResult> {
        self.input_buf.extend_from_slice(input);
        let frame_size = self.config.frame_size;
        let hop_size = self.config.hop_size();
        let mut results = Vec::new();

        while self.input_buf.len() >= frame_size {
            // Extract frame_size samples and apply Hann analysis window
            let mut buf = self.input_buf[..frame_size].to_vec();
            for (i, s) in buf.iter_mut().enumerate() {
                *s *= self.window[i];
            }

            self.fft
                .forward(&mut buf)
                .expect("buffer size matches frame_size");
            let soft = patchwork::detect_frame(
                self.fft.freq_bins(),
                &self.key,
                self.frame_counter as u32,
                &self.config,
            );
            self.soft_values.push(soft);
            self.frame_counter += 1;

            // Check if we have enough soft values for a full block
            if self.soft_values.len() >= self.frames_per_block
                && let Some(result) = self.try_detect()
            {
                results.push(result);
                self.soft_values.clear();
            }

            // Advance input by hop_size (not frame_size)
            self.input_buf.drain(..hop_size);
        }

        results
    }

    fn try_detect(&self) -> Option<DetectionResult> {
        let n = self.soft_values.len();
        if n < self.frames_per_block {
            return None;
        }

        // Try detection at offset 0
        let sync_soft = &self.soft_values[0..SYNC_PATTERN_BITS];
        let corr = correlate_sync(sync_soft, &self.sync_pattern);

        let data_start = SYNC_PATTERN_BITS;
        let data_end = data_start + self.coded_bits_count;
        if data_end > n {
            return None;
        }

        let data_soft = &self.soft_values[data_start..data_end];
        let decoded_bits = codec::decode(data_soft);

        if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
            Some(DetectionResult {
                payload,
                confidence: corr,
                offset: self.frame_counter - n,
            })
        } else {
            None
        }
    }

    /// Finalize detection with any remaining buffered data.
    pub fn finalize(&self) -> Option<DetectionResult> {
        self.try_detect()
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
}
