//! Optional parallel processing using rayon.
//!
//! Enable with the `parallel` feature flag. Provides `embed_parallel` and
//! `detect_parallel` which use rayon to process FFT frames across multiple
//! threads. Both use Hann-windowed overlap-add (50% overlap) matching the
//! sequential implementation.

use rayon::prelude::*;

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

/// Number of frames processed per rayon task.
const BATCH_SIZE: usize = 64;

/// Minimum sync correlation threshold (matches detect.rs).
const SYNC_THRESHOLD: f32 = 0.01;

/// Maximum Viterbi decode attempts to limit cost (matches detect.rs).
const MAX_DECODE_ATTEMPTS: usize = 5;

/// Embed a watermark into audio samples using parallel processing.
///
/// Functionally identical to [`crate::embed::embed`] but uses rayon to
/// parallelize the FFT processing. The overlap-add is performed sequentially
/// since overlapping frames write to shared output positions.
pub fn embed_parallel(
    samples: &mut [f32],
    payload: &Payload,
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> Result<()> {
    embed_parallel_with_offset(samples, payload, key, config, 0)
}

/// Embed a watermark into audio samples in parallel, starting at a frame offset.
pub fn embed_parallel_with_offset(
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

    // Phase 1: Parallel FFT processing. Each frame is independently windowed,
    // FFT'd, modified, IFFT'd, and normalized. Results stored per-frame.
    let frame_indices: Vec<usize> = (0..num_frames).collect();
    let frame_buffers: Vec<Vec<f32>> = frame_indices
        .par_chunks(BATCH_SIZE)
        .flat_map(|batch| {
            let mut fft = FftProcessor::new(frame_size).expect("FFT init should not fail");
            batch
                .iter()
                .map(|&frame_idx| {
                    let global_frame = frame_offset as usize + frame_idx;
                    let bit_idx = global_frame % block_len;
                    let bit = block_bits[bit_idx];
                    let bin_pair_seed = sync::bin_pair_seed(bit_idx);

                    let offset = frame_idx * hop_size;
                    let mut buf = vec![0.0f32; frame_size];
                    let end = (offset + frame_size).min(samples.len());
                    buf[..end - offset].copy_from_slice(&samples[offset..end]);

                    // Hann analysis window
                    for (i, s) in buf.iter_mut().enumerate() {
                        *s *= window[i];
                    }

                    fft.forward(&mut buf).expect("FFT forward should not fail");
                    patchwork::embed_frame(fft.freq_bins_mut(), bit, key, bin_pair_seed, config);
                    fft.inverse(&mut buf).expect("FFT inverse should not fail");
                    fft.normalize(&mut buf);

                    buf
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Phase 2: Sequential overlap-add into output buffer.
    let mut output = vec![0.0f32; samples.len()];
    for (frame_idx, buf) in frame_buffers.iter().enumerate() {
        let offset = frame_idx * hop_size;
        for (i, &sample) in buf.iter().enumerate() {
            let pos = offset + i;
            if pos < output.len() {
                output[pos] += sample;
            }
        }
    }

    samples.copy_from_slice(&output);
    Ok(())
}

/// Detect watermarks in audio samples using parallel processing.
///
/// The FFT and soft-value extraction is parallelized across frames.
/// Sync search and Viterbi decoding remain sequential.
pub fn detect_parallel(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> Result<Vec<DetectionResult>> {
    detect_parallel_with_offset(samples, key, config, 0)
}

/// Detect watermarks in audio samples in parallel, starting at a frame offset.
///
/// Two-pass approach matching the sequential detect:
/// - Pass 1: Compute soft values with constant seed 0 (parallel FFT) for sync search
/// - Pass 2: Recompute data soft values with correct block-relative seeds for Viterbi
pub fn detect_parallel_with_offset(
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

    // Pass 1: Parallel FFT + soft-value extraction with constant seed 0.
    // Accurate for sync frames, approximate for data frames (used only for sync search).
    let frame_indices: Vec<usize> = (0..num_frames).collect();
    let sync_soft_values: Vec<f32> = frame_indices
        .par_chunks(BATCH_SIZE)
        .flat_map(|batch| {
            let mut fft = FftProcessor::new(frame_size).expect("FFT init should not fail");
            batch
                .iter()
                .map(|&frame_idx| {
                    let offset = frame_idx * hop_size;

                    let mut buf = vec![0.0f32; frame_size];
                    let end = (offset + frame_size).min(samples.len());
                    buf[..end - offset].copy_from_slice(&samples[offset..end]);

                    for (i, s) in buf.iter_mut().enumerate() {
                        *s *= window[i];
                    }

                    fft.forward(&mut buf).expect("FFT forward should not fail");
                    patchwork::detect_frame(fft.freq_bins(), key, 0, config)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Sync search + two-pass Viterbi decode
    let mut results = Vec::new();

    if num_frames >= frames_per_block {
        let scan_end = num_frames - frames_per_block;
        let mut candidates: Vec<(usize, f32)> = (0..=scan_end)
            .map(|start| {
                let sync_soft = &sync_soft_values[start..start + SYNC_PATTERN_BITS];
                let corr = correlate_sync(sync_soft, &sync_pattern);
                (start, corr)
            })
            .filter(|&(_, corr)| corr > SYNC_THRESHOLD)
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(start, corr) in candidates.iter().take(MAX_DECODE_ATTEMPTS) {
            let data_start = start + SYNC_PATTERN_BITS;
            let data_end = data_start + coded_bits_count;
            if data_end > num_frames {
                continue;
            }

            // Pass 2: Recompute data soft values with correct block-relative seeds
            let data_frame_indices: Vec<usize> = (data_start..data_end).collect();
            let data_soft: Vec<f32> = data_frame_indices
                .par_chunks(BATCH_SIZE)
                .flat_map(|batch| {
                    let mut fft = FftProcessor::new(frame_size).expect("FFT init should not fail");
                    batch
                        .iter()
                        .map(|&frame_idx| {
                            let data_pos = frame_idx - data_start;
                            let seed = sync::bin_pair_seed(SYNC_PATTERN_BITS + data_pos);
                            let offset = frame_idx * hop_size;

                            let mut buf = vec![0.0f32; frame_size];
                            let end = (offset + frame_size).min(samples.len());
                            buf[..end - offset].copy_from_slice(&samples[offset..end]);

                            for (i, s) in buf.iter_mut().enumerate() {
                                *s *= window[i];
                            }

                            fft.forward(&mut buf).expect("FFT forward should not fail");
                            patchwork::detect_frame(fft.freq_bins(), key, seed, config)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            let decoded_bits = codec::decode(&data_soft);

            if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
                results.push(DetectionResult {
                    payload,
                    confidence: corr,
                    offset: start + frame_offset as usize,
                });
                break;
            }
        }

        // Fallback: try offset 0 directly
        if results.is_empty() {
            let data_start = SYNC_PATTERN_BITS;
            let data_end = data_start + coded_bits_count;
            if data_end <= num_frames {
                let data_frame_indices: Vec<usize> = (data_start..data_end).collect();
                let data_soft: Vec<f32> = data_frame_indices
                    .par_chunks(BATCH_SIZE)
                    .flat_map(|batch| {
                        let mut fft =
                            FftProcessor::new(frame_size).expect("FFT init should not fail");
                        batch
                            .iter()
                            .map(|&frame_idx| {
                                let data_pos = frame_idx - data_start;
                                let seed = sync::bin_pair_seed(SYNC_PATTERN_BITS + data_pos);
                                let offset = frame_idx * hop_size;

                                let mut buf = vec![0.0f32; frame_size];
                                let end = (offset + frame_size).min(samples.len());
                                buf[..end - offset].copy_from_slice(&samples[offset..end]);

                                for (i, s) in buf.iter_mut().enumerate() {
                                    *s *= window[i];
                                }

                                fft.forward(&mut buf).expect("FFT forward should not fail");
                                patchwork::detect_frame(fft.freq_bins(), key, seed, config)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();

                let decoded_bits = codec::decode(&data_soft);
                if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
                    let sync_soft = &sync_soft_values[0..SYNC_PATTERN_BITS];
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
    fn parallel_embed_matches_sequential() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let audio = make_test_audio(48000 * 2, config.sample_rate);

        let mut seq = audio.clone();
        crate::embed::embed(&mut seq, &payload, &key, &config).unwrap();

        let mut par = audio.clone();
        embed_parallel(&mut par, &payload, &key, &config).unwrap();

        assert_eq!(seq.len(), par.len());
        for (i, (a, b)) in seq.iter().zip(par.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "mismatch at sample {i}: seq={a}, par={b}"
            );
        }
    }

    #[test]
    fn parallel_detect_matches_sequential() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC,
            0xBA, 0x98,
        ]);

        let num_samples = 48000 * 20;
        let mut audio = make_test_audio(num_samples, config.sample_rate);
        crate::embed::embed(&mut audio, &payload, &key, &config).unwrap();

        let seq_results = crate::detect::detect(&audio, &key, &config).unwrap();
        let par_results = detect_parallel(&audio, &key, &config).unwrap();

        assert_eq!(seq_results.len(), par_results.len());
        for (s, p) in seq_results.iter().zip(par_results.iter()) {
            assert_eq!(s.payload, p.payload);
            assert_eq!(s.offset, p.offset);
            assert!((s.confidence - p.confidence).abs() < 1e-6);
        }
    }

    #[test]
    fn parallel_embed_detect_round_trip() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22,
            0x33, 0x44,
        ]);

        let num_samples = 48000 * 20;
        let mut audio = make_test_audio(num_samples, config.sample_rate);

        embed_parallel(&mut audio, &payload, &key, &config).unwrap();
        let results = detect_parallel(&audio, &key, &config).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].payload, payload);
    }
}
