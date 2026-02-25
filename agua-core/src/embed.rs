use crate::codec;
use crate::config::WatermarkConfig;
use crate::error::{Error, Result};
use crate::fft::FftProcessor;
use crate::frame::hann_window;
use crate::key::WatermarkKey;
use crate::patchwork;
use crate::payload::Payload;
use crate::sync::generate_sync_pattern;

/// Embed a watermark into audio samples (in-place).
///
/// Processes audio using Hann-windowed overlap-add (WOLA) with 50% overlap.
/// Each frame is analysis-windowed (Hann), transformed to frequency domain,
/// patchwork-modified, transformed back, and overlap-added into the output
/// buffer. The Hann window at 50% overlap satisfies the COLA condition
/// (constant overlap-add sum = 1.0), ensuring near-perfect reconstruction
/// when no modification is applied.
pub fn embed(
    samples: &mut [f32],
    payload: &Payload,
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> Result<()> {
    embed_with_offset(samples, payload, key, config, 0)
}

/// Embed a watermark into audio samples (in-place) starting at a frame offset.
///
/// `frame_offset` specifies the absolute frame index of `samples[0]` in the
/// original stream, ensuring PRNG bin selection aligns across segments.
pub fn embed_with_offset(
    samples: &mut [f32],
    payload: &Payload,
    key: &WatermarkKey,
    config: &WatermarkConfig,
    frame_offset: u32,
) -> Result<()> {
    let frame_size = config.frame_size;
    if samples.len() < frame_size {
        return Err(Error::AudioTooShort {
            needed: frame_size,
            got: samples.len(),
        });
    }

    let hop_size = config.hop_size();
    let num_frames = (samples.len() - frame_size) / hop_size + 1;

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

    let window = hann_window(frame_size);
    let mut fft = FftProcessor::new(frame_size)?;

    // Output buffer for overlap-add reconstruction
    let mut output = vec![0.0f32; samples.len()];

    for frame_idx in 0..num_frames {
        let offset = frame_idx * hop_size;
        let global_frame = frame_offset as usize + frame_idx;
        let bit_idx = global_frame % block_len;
        let bit = block_bits[bit_idx];

        // Extract frame and apply Hann analysis window
        let mut buf = vec![0.0f32; frame_size];
        let end = (offset + frame_size).min(samples.len());
        buf[..end - offset].copy_from_slice(&samples[offset..end]);
        for (i, s) in buf.iter_mut().enumerate() {
            *s *= window[i];
        }

        // FFT -> modify bins -> IFFT -> normalize
        fft.forward(&mut buf)?;
        patchwork::embed_frame(fft.freq_bins_mut(), bit, key, global_frame as u32, config);
        fft.inverse(&mut buf)?;
        fft.normalize(&mut buf);

        // Overlap-add WITHOUT synthesis window. With analysis-only Hann at
        // 50% overlap the COLA condition holds exactly (sum = 1.0), giving
        // perfect reconstruction when strength = 0. Omitting the synthesis
        // window also preserves the full watermark modification amplitude,
        // which is critical for detection through the second Hann analysis
        // window applied by the detector.
        for (i, &sample) in buf.iter().enumerate() {
            let pos = offset + i;
            if pos < output.len() {
                output[pos] += sample;
            }
        }
    }

    // Copy output back into samples
    samples.copy_from_slice(&output);

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

        // Check distortion in steady-state region (skip boundary frames where
        // the Hann window taper means incomplete overlap-add coverage).
        let frame_size = config.frame_size;
        let max_diff: f32 = original[frame_size..original.len() - frame_size]
            .iter()
            .zip(watermarked[frame_size..watermarked.len() - frame_size].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 0.1, "watermark distortion too high: {max_diff}");

        let total_diff: f32 = original[frame_size..original.len() - frame_size]
            .iter()
            .zip(watermarked[frame_size..watermarked.len() - frame_size].iter())
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

        // With Hann analysis window at 50% overlap, the COLA condition holds
        // (window sum = 1.0 in steady state). Reconstruction should be near-perfect.
        // Boundary regions (first/last frame_size samples) may differ slightly.
        let frame_size = config.frame_size;
        let max_diff: f32 = original[frame_size..original.len() - frame_size]
            .iter()
            .zip(reconstructed[frame_size..reconstructed.len() - frame_size].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "reconstruction error too high: {max_diff}");
    }
}
