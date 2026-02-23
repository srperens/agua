use crate::codec;
use crate::config::WatermarkConfig;
use crate::embed::DetectionResult;
use crate::fft::FftProcessor;
use crate::key::WatermarkKey;
use crate::patchwork;
use crate::payload::Payload;
use crate::sync::{self, SYNC_PATTERN_BITS, correlate_sync, generate_sync_pattern};

/// Streaming watermark embedder.
///
/// Accepts arbitrary-length input chunks and produces watermarked output.
/// Internally buffers to maintain frame alignment.
pub struct StreamEmbedder {
    config: WatermarkConfig,
    key: WatermarkKey,
    fft: FftProcessor,
    block_bits: Vec<bool>,
    input_buf: Vec<f32>,
    frame_counter: usize,
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

        Ok(Self {
            config: config.clone(),
            key: key.clone(),
            fft,
            block_bits,
            input_buf: Vec::new(),
            frame_counter: 0,
        })
    }

    /// Process input samples and return watermarked output.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.input_buf.extend_from_slice(input);
        let frame_size = self.config.frame_size;
        let mut result = Vec::new();

        while self.input_buf.len() >= frame_size {
            let mut buf: Vec<f32> = self.input_buf.drain(..frame_size).collect();

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

            result.extend_from_slice(&buf);
            self.frame_counter += 1;
        }

        result
    }

    /// Flush remaining buffered samples (unmodified, since they don't form a full frame).
    pub fn flush(&mut self) -> Vec<f32> {
        self.input_buf.drain(..).collect()
    }
}

/// Streaming watermark detector.
///
/// Accepts arbitrary-length input chunks and reports detected watermarks.
pub struct StreamDetector {
    config: WatermarkConfig,
    key: WatermarkKey,
    fft: FftProcessor,
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
        let sync_pattern = generate_sync_pattern(key);
        let frames_per_block = sync::frames_per_block();
        let coded_bits_count = codec::CODED_BITS;

        Ok(Self {
            config: config.clone(),
            key: key.clone(),
            fft,
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
        let mut results = Vec::new();

        while self.input_buf.len() >= frame_size {
            let mut buf: Vec<f32> = self.input_buf.drain(..frame_size).collect();

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
