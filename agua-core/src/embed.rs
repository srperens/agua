use crate::codec;
use crate::config::WatermarkConfig;
use crate::error::{Error, Result};
use crate::fft::FftProcessor;
use crate::key::WatermarkKey;
use crate::patchwork;
use crate::payload::Payload;
use crate::sync::generate_sync_pattern;

/// Embed a watermark into audio samples (in-place).
///
/// Processes audio in non-overlapping frames using direct FFT replacement.
/// Each frame is transformed to frequency domain, patchwork-modified, and
/// transformed back. Non-overlapping frames ensure the detector sees the
/// exact modified spectrum.
pub fn embed(
    samples: &mut [f32],
    payload: &Payload,
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> Result<()> {
    let frame_size = config.frame_size;
    if samples.len() < frame_size {
        return Err(Error::AudioTooShort {
            needed: frame_size,
            got: samples.len(),
        });
    }
    let num_frames = samples.len() / frame_size;

    // Prepare the bit sequence: sync pattern + convolutionally-encoded payload+CRC
    let sync_pattern = generate_sync_pattern(key);
    let data_bits = payload.encode_with_crc();
    let coded_bits = codec::encode(&data_bits);

    let block_bits: Vec<bool> = sync_pattern
        .iter()
        .chain(coded_bits.iter())
        .copied()
        .collect();
    let block_len = block_bits.len();

    let mut fft = FftProcessor::new(frame_size)?;

    for frame_idx in 0..num_frames {
        let offset = frame_idx * frame_size;
        let bit_idx = frame_idx % block_len;
        let bit = block_bits[bit_idx];

        // Copy frame into buffer
        let mut buf = samples[offset..offset + frame_size].to_vec();

        // FFT → modify bins → IFFT → replace frame
        fft.forward(&mut buf)?;
        patchwork::embed_frame(fft.freq_bins_mut(), bit, key, frame_idx as u32, config);
        fft.inverse(&mut buf)?;
        fft.normalize(&mut buf);

        samples[offset..offset + frame_size].copy_from_slice(&buf);
    }

    Ok(())
}

/// Detection result containing the recovered payload and confidence.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub payload: Payload,
    pub confidence: f32,
    /// Frame offset where the watermark was found.
    pub offset: usize,
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
    fn embed_does_not_destroy_signal() {
        let config = WatermarkConfig::default();
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let original = make_test_audio(config.sample_rate as usize * 2, config.sample_rate);
        let mut watermarked = original.clone();

        embed(&mut watermarked, &payload, &key, &config).unwrap();

        let max_diff: f32 = original
            .iter()
            .zip(watermarked.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 0.1, "watermark distortion too high: {max_diff}");

        let total_diff: f32 = original
            .iter()
            .zip(watermarked.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(total_diff > 0.0, "watermark had no effect");
    }

    #[test]
    fn embed_too_short() {
        let config = WatermarkConfig::default();
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([0; 16]);
        let mut samples = vec![0.0f32; 100];

        assert!(embed(&mut samples, &payload, &key, &config).is_err());
    }

    #[test]
    fn perfect_reconstruction_without_modification() {
        let config = WatermarkConfig {
            strength: 0.0,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([0; 16]);

        let original = make_test_audio(48000, config.sample_rate);
        let mut reconstructed = original.clone();

        embed(&mut reconstructed, &payload, &key, &config).unwrap();

        let max_diff: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "reconstruction error too high: {max_diff}");
    }
}
