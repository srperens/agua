//! Optional parallel processing using rayon.
//!
//! Enable with the `parallel` feature flag. Provides `embed_parallel` and
//! `detect_parallel` which use rayon to process FFT frames across multiple
//! threads.

use rayon::prelude::*;

use crate::codec;
use crate::config::WatermarkConfig;
use crate::embed::DetectionResult;
use crate::error::{Error, Result};
use crate::fft::FftProcessor;
use crate::key::WatermarkKey;
use crate::patchwork;
use crate::payload::Payload;
use crate::sync::{self, SYNC_PATTERN_BITS, correlate_sync, generate_sync_pattern};

/// Number of frames processed per rayon task.
const BATCH_SIZE: usize = 64;

/// Embed a watermark into audio samples using parallel processing.
///
/// Functionally identical to [`crate::embed`] but uses rayon to process
/// batches of frames in parallel. Each batch creates its own [`FftProcessor`].
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

    // Trim samples to exact frame boundary for par_chunks_mut
    let usable = num_frames * frame_size;
    let sample_slice = &mut samples[..usable];

    // Process in batches of BATCH_SIZE frames.
    // Each chunk of (BATCH_SIZE * frame_size) samples is one rayon task.
    let chunk_size = BATCH_SIZE * frame_size;

    sample_slice
        .par_chunks_mut(chunk_size)
        .enumerate()
        .try_for_each(|(chunk_idx, chunk)| -> Result<()> {
            let mut fft = FftProcessor::new(frame_size)?;
            let base_frame = chunk_idx * BATCH_SIZE;
            let frames_in_chunk = chunk.len() / frame_size;

            for local_frame in 0..frames_in_chunk {
                let frame_idx = base_frame + local_frame;
                let global_frame = frame_offset as usize + frame_idx;
                let bit_idx = global_frame % block_len;
                let bit = block_bits[bit_idx];

                let offset = local_frame * frame_size;
                let mut buf = chunk[offset..offset + frame_size].to_vec();

                fft.forward(&mut buf)?;
                patchwork::embed_frame(fft.freq_bins_mut(), bit, key, global_frame as u32, config);
                fft.inverse(&mut buf)?;
                fft.normalize(&mut buf);

                chunk[offset..offset + frame_size].copy_from_slice(&buf);
            }

            Ok(())
        })?;

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
    let num_frames = samples.len() / frame_size;

    let sync_pattern = generate_sync_pattern(key);
    let frames_per_block = sync::frames_per_block();
    let coded_bits_count = codec::CODED_BITS;

    // Parallel FFT + soft-value extraction in batches
    let chunk_size = BATCH_SIZE * frame_size;
    let usable = num_frames * frame_size;
    let sample_slice = &samples[..usable];

    let batch_results: Vec<Vec<(usize, f32)>> = sample_slice
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut fft = FftProcessor::new(frame_size).expect("FFT init should not fail");
            let base_frame = chunk_idx * BATCH_SIZE;
            let frames_in_chunk = chunk.len() / frame_size;

            let mut local_softs = Vec::with_capacity(frames_in_chunk);
            for local_frame in 0..frames_in_chunk {
                let frame_idx = base_frame + local_frame;
                let global_frame = frame_offset as usize + frame_idx;
                let offset = local_frame * frame_size;
                let mut buf = chunk[offset..offset + frame_size].to_vec();
                fft.forward(&mut buf).expect("FFT forward should not fail");
                let soft =
                    patchwork::detect_frame(fft.freq_bins(), key, global_frame as u32, config);
                local_softs.push((frame_idx, soft));
            }
            local_softs
        })
        .collect();

    // Flatten into ordered soft_values
    let mut soft_values = vec![0.0f32; num_frames];
    for batch in &batch_results {
        for &(idx, val) in batch {
            soft_values[idx] = val;
        }
    }

    // Sequential sync search + Viterbi decode (same as detect.rs)
    let mut results = Vec::new();

    if num_frames >= frames_per_block {
        // Strategy 1: Scan for sync patterns
        for start in 0..=(num_frames - frames_per_block) {
            let sync_soft = &soft_values[start..start + SYNC_PATTERN_BITS];
            let corr = correlate_sync(sync_soft, &sync_pattern);

            if corr > 0.15 {
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
            strength: 0.03,
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
            strength: 0.03,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC,
            0xBA, 0x98,
        ]);

        let num_samples = 48000 * 22;
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
            strength: 0.03,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = Payload::new([
            0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22,
            0x33, 0x44,
        ]);

        let num_samples = 48000 * 22;
        let mut audio = make_test_audio(num_samples, config.sample_rate);

        embed_parallel(&mut audio, &payload, &key, &config).unwrap();
        let results = detect_parallel(&audio, &key, &config).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].payload, payload);
    }
}
