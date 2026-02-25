use crate::codec;
use crate::config::WatermarkConfig;
use crate::embed::DetectionResult;
use crate::error::{Error, Result};
use crate::fft::FftProcessor;
use crate::frame::hann_window;
use crate::key::WatermarkKey;
use crate::patchwork;
use crate::payload::Payload;
use crate::sync::{self, SYNC_PATTERN_BITS, correlate_sync, generate_sync_pattern};

/// Minimum sync correlation threshold to consider a sync pattern detected.
///
/// Kept low because the CRC-32 check after Viterbi decoding provides the
/// real false-positive rejection (probability ~2^-32). The threshold only
/// serves to limit the number of Viterbi decode attempts.
const SYNC_THRESHOLD: f32 = 0.01;

/// Detect watermarks in audio samples.
///
/// Processes audio in Hann-windowed overlapping frames (50% overlap),
/// matching the embedding scheme. Returns all successfully detected
/// watermark payloads.
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

    let hop_size = config.hop_size();
    let num_frames = (samples.len() - frame_size) / hop_size + 1;

    let sync_pattern = generate_sync_pattern(key);
    let frames_per_block = sync::frames_per_block();
    let coded_bits_count = codec::CODED_BITS;

    let window = hann_window(frame_size);
    let mut fft = FftProcessor::new(frame_size)?;
    let mut soft_values = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let offset = frame_idx * hop_size;

        // Extract frame and apply Hann analysis window. While this
        // attenuates the soft values ~3x vs rectangular, the Hann window
        // provides superior frequency selectivity that improves
        // signal-to-noise ratio for the Viterbi decoder.
        let mut buf = vec![0.0f32; frame_size];
        let end = (offset + frame_size).min(samples.len());
        buf[..end - offset].copy_from_slice(&samples[offset..end]);
        for (i, s) in buf.iter_mut().enumerate() {
            *s *= window[i];
        }

        fft.forward(&mut buf)?;
        let global_frame = frame_offset as usize + frame_idx;
        let soft = patchwork::detect_frame(fft.freq_bins(), key, global_frame as u32, config);
        soft_values.push(soft);
    }

    let mut results = Vec::new();

    // Maximum Viterbi decode attempts to limit cost (K=15 decode is expensive).
    const MAX_DECODE_ATTEMPTS: usize = 5;

    if num_frames >= frames_per_block {
        // Compute sync correlation at every valid position, then try
        // Viterbi decoding at the best candidates (best-first search).
        let scan_end = num_frames - frames_per_block;
        let mut candidates: Vec<(usize, f32)> = (0..=scan_end)
            .map(|start| {
                let sync_soft = &soft_values[start..start + SYNC_PATTERN_BITS];
                let corr = correlate_sync(sync_soft, &sync_pattern);
                (start, corr)
            })
            .filter(|&(_, corr)| corr > SYNC_THRESHOLD)
            .collect();

        // Sort by correlation descending (best candidates first)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(start, corr) in candidates.iter().take(MAX_DECODE_ATTEMPTS) {
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
                break;
            }
        }

        // Fallback: try offset 0 directly (no sync threshold)
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
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC,
            0xBA, 0x98,
        ]);

        // frames_per_block = 128 sync + 960 data = 1088 frames
        // With 50% overlap (hop_size=512): 1088 * 512 + 1024 = ~558k samples ~= 11.6s
        // Use 20s to provide comfortable margin for boundary effects.
        let num_samples = 48000 * 20;
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
        let audio = make_test_audio(48000 * 13, config.sample_rate);

        let result = detect(&audio, &key, &config);
        assert!(result.is_err());
    }

    #[test]
    fn detect_wrong_key() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let wrong_key = WatermarkKey::new(&[99u8; 16]).unwrap();
        let payload = Payload::new([0xFF; 16]);

        let num_samples = 48000 * 13;
        let mut audio = make_test_audio(num_samples, config.sample_rate);
        embed(&mut audio, &payload, &key, &config).unwrap();

        let result = detect(&audio, &wrong_key, &config);
        assert!(result.is_err(), "should not detect with wrong key");
    }
}
