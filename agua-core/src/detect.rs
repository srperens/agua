use crate::codec;
use crate::config::WatermarkConfig;
use crate::embed::DetectionResult;
use crate::error::{Error, Result};
use crate::fft::FftProcessor;
use crate::key::WatermarkKey;
use crate::patchwork;
use crate::payload::Payload;
use crate::sync::{self, SYNC_PATTERN_BITS, correlate_sync, generate_sync_pattern};

/// Minimum sync correlation threshold to consider a sync pattern detected.
const SYNC_THRESHOLD: f32 = 0.15;

/// Detect watermarks in audio samples.
///
/// Processes audio in non-overlapping frames (matching the embedding scheme).
/// Returns all successfully detected watermark payloads.
pub fn detect(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> Result<Vec<DetectionResult>> {
    detect_with_offset(samples, key, config, 0)
}

/// Detect watermarks in audio samples starting at a frame offset.
///
/// `frame_offset` specifies the absolute frame index of `samples[0]` in the
/// original stream, ensuring PRNG bin selection aligns across segments.
pub fn detect_with_offset(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
    frame_offset: u32,
) -> Result<Vec<DetectionResult>> {
    let frame_size = config.frame_size;
    if samples.len() < frame_size {
        return Err(Error::AudioTooShort {
            needed: frame_size,
            got: samples.len(),
        });
    }
    let num_frames = samples.len() / frame_size;

    let sync_pattern = generate_sync_pattern(key);
    let frames_per_block = sync::frames_per_block();
    let coded_bits_count = codec::CODED_BITS;

    // Extract soft values for all frames (no windowing, matching embed)
    let mut fft = FftProcessor::new(frame_size)?;
    let mut soft_values = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let offset = frame_idx * frame_size;
        let mut buf = samples[offset..offset + frame_size].to_vec();
        fft.forward(&mut buf)?;
        let global_frame = frame_offset as usize + frame_idx;
        let soft = patchwork::detect_frame(fft.freq_bins(), key, global_frame as u32, config);
        soft_values.push(soft);
    }

    let mut results = Vec::new();

    if num_frames >= frames_per_block {
        // Strategy 1: Scan for sync patterns
        for start in 0..=(num_frames - frames_per_block) {
            let sync_soft = &soft_values[start..start + SYNC_PATTERN_BITS];
            let corr = correlate_sync(sync_soft, &sync_pattern);

            if corr > SYNC_THRESHOLD {
                let data_start = start + SYNC_PATTERN_BITS;
                let data_end = data_start + coded_bits_count;
                if data_end > num_frames {
                    continue;
                }

                let data_soft = &soft_values[data_start..data_end];
                let decoded_bits = codec::decode(data_soft);

                if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
                    results.push(DetectionResult {
                        payload,
                        confidence: corr,
                        offset: start + frame_offset as usize,
                    });
                }
            }
        }

        // Strategy 2: Try offset 0 directly (no sync threshold)
        if results.is_empty() {
            let data_start = SYNC_PATTERN_BITS;
            let data_end = data_start + coded_bits_count;
            if data_end <= num_frames {
                let data_soft = &soft_values[data_start..data_end];
                let decoded_bits = codec::decode(data_soft);
                if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
                    let sync_soft = &soft_values[0..SYNC_PATTERN_BITS];
                    let corr = correlate_sync(sync_soft, &sync_pattern);
                    results.push(DetectionResult {
                        payload,
                        confidence: corr,
                        offset: frame_offset as usize,
                    });
                }
            }
        }
    }

    if results.is_empty() {
        Err(Error::NotDetected)
    } else {
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::embed;

    /// Broadband test audio with energy across many frequencies.
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
    fn embed_detect_round_trip() {
        let config = WatermarkConfig {
            strength: 0.03,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC,
            0xBA, 0x98,
        ]);

        // frames_per_block = 64 + 960 = 1024 frames
        // Non-overlapping: need 1024 * 1024 = ~1M samples ≈ 21s at 48kHz
        let num_samples = 48000 * 22;
        let mut audio = make_test_audio(num_samples, config.sample_rate);

        embed(&mut audio, &payload, &key, &config).unwrap();

        let results = detect(&audio, &key, &config).unwrap();
        assert!(!results.is_empty(), "no watermark detected");
        assert_eq!(results[0].payload, payload);
    }

    #[test]
    fn detect_no_watermark() {
        let config = WatermarkConfig::default();
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let audio = make_test_audio(48000 * 22, config.sample_rate);

        let result = detect(&audio, &key, &config);
        assert!(result.is_err());
    }

    #[test]
    fn detect_wrong_key() {
        let config = WatermarkConfig {
            strength: 0.03,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let wrong_key = WatermarkKey::new(&[99u8; 16]).unwrap();
        let payload = Payload::new([0xFF; 16]);

        let num_samples = 48000 * 22;
        let mut audio = make_test_audio(num_samples, config.sample_rate);
        embed(&mut audio, &payload, &key, &config).unwrap();

        let result = detect(&audio, &wrong_key, &config);
        assert!(result.is_err(), "should not detect with wrong key");
    }
}
