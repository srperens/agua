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
const SYNC_THRESHOLD: f32 = 0.02;

/// Maximum Viterbi decode attempts to limit cost (K=15 decode is expensive).
const MAX_DECODE_ATTEMPTS: usize = 5;

/// Coarse sample-offset search step divides hop_size into this many steps.
/// With hop_size=512, step = 512/16 = 32 samples.
const COARSE_SEARCH_DIVISOR: usize = 16;

/// Fine sample-offset search step divides hop_size into this many steps.
/// With hop_size=512, step = 512/64 = 8 samples.
const FINE_SEARCH_DIVISOR: usize = 64;

/// Diagnostics from a detection attempt.
#[derive(Debug, Clone, Default)]
pub struct DetectionDiagnostics {
    /// Best sync correlation found across all offsets/candidates.
    pub best_sync_corr: f32,
    /// Total sync candidates above threshold across all offsets.
    pub sync_candidates: u32,
    /// Number of offsets searched.
    pub offsets_searched: u32,
    /// Number of Viterbi decode attempts.
    pub viterbi_attempts: u32,
    /// Whether the fast path (offset 0) was tried.
    pub fast_path_tried: bool,
}

/// Detect watermarks in audio samples.
///
/// Searches for the watermark across multiple sub-frame sample offsets
/// to handle arbitrary alignment between embedder and detector frame
/// grids. Uses a two-phase approach:
///
/// 1. **Fast path**: try offset 0 (covers the common file-based case)
/// 2. **Coarse search**: try offsets at `hop/16` steps, running two-pass
///    detection (sync finding + data decoding) at each
/// 3. **Fine search**: refine around the best coarse candidate at `hop/64`
pub fn detect(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> Result<Vec<DetectionResult>> {
    // Fast path: try aligned detection first (covers the common case)
    if let Ok(results) = detect_with_offset(samples, key, config, 0) {
        return Ok(results);
    }

    let hop_size = config.hop_size();
    let frame_size = config.frame_size;
    let coarse_step = (hop_size / COARSE_SEARCH_DIVISOR).max(1);
    let fine_step = (hop_size / FINE_SEARCH_DIVISOR).max(1);

    let sync_pattern = generate_sync_pattern(key);
    let frames_per_block = sync::frames_per_block();
    let window = hann_window(frame_size);

    // --- Phase 1: Coarse search ---
    // At each coarse offset, run two-pass detection: constant-seed sync
    // finding + block-position-aware data decoding.
    let mut best_coarse_offset = coarse_step;
    let mut best_coarse_corr = f32::NEG_INFINITY;

    for i in 1..COARSE_SEARCH_DIVISOR {
        let sample_offset = i * coarse_step;
        if sample_offset + frame_size > samples.len() {
            break;
        }
        let buf = &samples[sample_offset..];
        let num_frames = (buf.len() - frame_size) / hop_size + 1;
        if num_frames < frames_per_block {
            continue;
        }

        // Pass 1: sync soft values with constant seed 0
        let sync_soft = compute_soft_values_const_seed(buf, key, config, &window, num_frames)?;

        if let Some(result) = try_two_pass_decode(
            buf,
            key,
            config,
            &window,
            &sync_soft,
            num_frames,
            &sync_pattern,
            sample_offset,
        )? {
            return Ok(vec![result]);
        }

        // Track best coarse offset for fine search
        let scan_end = num_frames - frames_per_block;
        let max_corr = (0..=scan_end)
            .map(|start| {
                correlate_sync(&sync_soft[start..start + SYNC_PATTERN_BITS], &sync_pattern)
            })
            .fold(f32::NEG_INFINITY, f32::max);
        if max_corr > best_coarse_corr {
            best_coarse_corr = max_corr;
            best_coarse_offset = sample_offset;
        }
    }

    // --- Phase 2: Fine search around the best coarse candidate ---
    let search_start = best_coarse_offset.saturating_sub(coarse_step);
    let search_end = (best_coarse_offset + coarse_step).min(hop_size);

    let mut off = search_start;
    while off <= search_end {
        if !off.is_multiple_of(coarse_step) && off + frame_size <= samples.len() {
            let buf = &samples[off..];
            let num_frames = (buf.len() - frame_size) / hop_size + 1;
            if num_frames >= frames_per_block {
                let sync_soft =
                    compute_soft_values_const_seed(buf, key, config, &window, num_frames)?;

                if let Some(result) = try_two_pass_decode(
                    buf,
                    key,
                    config,
                    &window,
                    &sync_soft,
                    num_frames,
                    &sync_pattern,
                    off,
                )? {
                    return Ok(vec![result]);
                }
            }
        }
        off += fine_step;
    }

    Err(Error::NotDetected)
}

/// Like `detect()` but also returns diagnostics about the search.
pub fn detect_with_diagnostics(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> (
    core::result::Result<Vec<DetectionResult>, Error>,
    DetectionDiagnostics,
) {
    let mut diag = DetectionDiagnostics {
        fast_path_tried: true,
        ..DetectionDiagnostics::default()
    };

    // Fast path: try aligned detection first
    if let Ok(results) = detect_with_offset(samples, key, config, 0) {
        if let Some(r) = results.first() {
            diag.best_sync_corr = r.confidence;
            diag.sync_candidates = 1;
            diag.viterbi_attempts = 1;
        }
        return (Ok(results), diag);
    }

    let hop_size = config.hop_size();
    let frame_size = config.frame_size;
    let coarse_step = (hop_size / COARSE_SEARCH_DIVISOR).max(1);
    let fine_step = (hop_size / FINE_SEARCH_DIVISOR).max(1);

    let sync_pattern = generate_sync_pattern(key);
    let frames_per_block = sync::frames_per_block();
    let window = hann_window(frame_size);

    let mut best_coarse_offset = coarse_step;
    let mut best_coarse_corr = f32::NEG_INFINITY;

    for i in 1..COARSE_SEARCH_DIVISOR {
        let sample_offset = i * coarse_step;
        if sample_offset + frame_size > samples.len() {
            break;
        }
        let buf = &samples[sample_offset..];
        let num_frames = (buf.len() - frame_size) / hop_size + 1;
        if num_frames < frames_per_block {
            continue;
        }
        diag.offsets_searched += 1;

        let sync_soft = match compute_soft_values_const_seed(buf, key, config, &window, num_frames)
        {
            Ok(s) => s,
            Err(e) => return (Err(e), diag),
        };

        // Count candidates at this offset
        let scan_end = num_frames - frames_per_block;
        let candidates_here: Vec<(usize, f32)> = (0..=scan_end)
            .map(|start| {
                let corr =
                    correlate_sync(&sync_soft[start..start + SYNC_PATTERN_BITS], &sync_pattern);
                (start, corr)
            })
            .filter(|&(_, corr)| corr > SYNC_THRESHOLD)
            .collect();

        for &(_, corr) in &candidates_here {
            if corr > diag.best_sync_corr {
                diag.best_sync_corr = corr;
            }
        }
        diag.sync_candidates += candidates_here.len() as u32;
        diag.viterbi_attempts += candidates_here.len().min(MAX_DECODE_ATTEMPTS) as u32;

        if let Ok(Some(result)) = try_two_pass_decode(
            buf,
            key,
            config,
            &window,
            &sync_soft,
            num_frames,
            &sync_pattern,
            sample_offset,
        ) {
            return (Ok(vec![result]), diag);
        }

        let max_corr = (0..=scan_end)
            .map(|start| {
                correlate_sync(&sync_soft[start..start + SYNC_PATTERN_BITS], &sync_pattern)
            })
            .fold(f32::NEG_INFINITY, f32::max);
        if max_corr > best_coarse_corr {
            best_coarse_corr = max_corr;
            best_coarse_offset = sample_offset;
        }
    }

    // Fine search
    let search_start = best_coarse_offset.saturating_sub(coarse_step);
    let search_end = (best_coarse_offset + coarse_step).min(hop_size);
    let mut off = search_start;
    while off <= search_end {
        if !off.is_multiple_of(coarse_step) && off + frame_size <= samples.len() {
            let buf = &samples[off..];
            let num_frames = (buf.len() - frame_size) / hop_size + 1;
            if num_frames >= frames_per_block {
                diag.offsets_searched += 1;
                let sync_soft =
                    match compute_soft_values_const_seed(buf, key, config, &window, num_frames) {
                        Ok(s) => s,
                        Err(e) => return (Err(e), diag),
                    };

                let scan_end = num_frames - frames_per_block;
                let candidates_here: Vec<(usize, f32)> = (0..=scan_end)
                    .map(|start| {
                        let corr = correlate_sync(
                            &sync_soft[start..start + SYNC_PATTERN_BITS],
                            &sync_pattern,
                        );
                        (start, corr)
                    })
                    .filter(|&(_, corr)| corr > SYNC_THRESHOLD)
                    .collect();

                for &(_, corr) in &candidates_here {
                    if corr > diag.best_sync_corr {
                        diag.best_sync_corr = corr;
                    }
                }
                diag.sync_candidates += candidates_here.len() as u32;
                diag.viterbi_attempts += candidates_here.len().min(MAX_DECODE_ATTEMPTS) as u32;

                if let Ok(Some(result)) = try_two_pass_decode(
                    buf,
                    key,
                    config,
                    &window,
                    &sync_soft,
                    num_frames,
                    &sync_pattern,
                    off,
                ) {
                    return (Ok(vec![result]), diag);
                }
            }
        }
        off += fine_step;
    }

    (Err(Error::NotDetected), diag)
}

/// Fast single-offset detection with diagnostics for streaming use.
///
/// Only checks offset 0 (no multi-offset search). Much faster than
/// `detect_with_diagnostics` — suitable for WASM/real-time where the
/// caller provides offset diversity over time via buffer sliding.
pub fn detect_single_offset_with_diagnostics(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> (
    core::result::Result<Vec<DetectionResult>, Error>,
    DetectionDiagnostics,
) {
    let mut diag = DetectionDiagnostics {
        fast_path_tried: true,
        offsets_searched: 1,
        ..DetectionDiagnostics::default()
    };

    let frame_size = config.frame_size;
    if samples.len() < frame_size {
        return (
            Err(Error::AudioTooShort {
                needed: frame_size,
                got: samples.len(),
            }),
            diag,
        );
    }

    let hop_size = config.hop_size();
    let num_frames = (samples.len() - frame_size) / hop_size + 1;
    let sync_pattern = generate_sync_pattern(key);
    let frames_per_block = sync::frames_per_block();
    let coded_bits_count = codec::CODED_BITS;

    if num_frames < frames_per_block {
        return (Err(Error::NotDetected), diag);
    }

    let window = hann_window(frame_size);
    let sync_soft = match compute_soft_values_const_seed(samples, key, config, &window, num_frames)
    {
        Ok(s) => s,
        Err(e) => return (Err(e), diag),
    };

    // Find sync candidates
    let scan_end = num_frames - frames_per_block;
    let mut candidates: Vec<(usize, f32)> = (0..=scan_end)
        .map(|start| {
            let corr = correlate_sync(&sync_soft[start..start + SYNC_PATTERN_BITS], &sync_pattern);
            (start, corr)
        })
        .filter(|&(_, corr)| corr > SYNC_THRESHOLD)
        .collect();

    for &(_, corr) in &candidates {
        if corr > diag.best_sync_corr {
            diag.best_sync_corr = corr;
        }
    }
    diag.sync_candidates = candidates.len() as u32;

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    // Limit Viterbi attempts (expensive with K=15)
    let max_attempts = MAX_DECODE_ATTEMPTS;
    diag.viterbi_attempts = candidates.len().min(max_attempts) as u32;

    for &(start, corr) in candidates.iter().take(max_attempts) {
        let data_frame_start = start + SYNC_PATTERN_BITS;
        let data_frame_end = data_frame_start + coded_bits_count;
        if data_frame_end > num_frames {
            continue;
        }

        let data_soft = match compute_data_soft_values(
            samples,
            key,
            config,
            &window,
            data_frame_start,
            coded_bits_count,
        ) {
            Ok(s) => s,
            Err(e) => return (Err(e), diag),
        };

        let decoded_bits = codec::decode(&data_soft);
        if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
            return (
                Ok(vec![DetectionResult {
                    payload,
                    confidence: corr,
                    offset: start,
                }]),
                diag,
            );
        }
    }

    (Err(Error::NotDetected), diag)
}

/// Two-pass decode: find sync using constant-seed soft values, then
/// recompute data soft values with correct block positions for Viterbi.
#[allow(clippy::too_many_arguments)]
fn try_two_pass_decode(
    buf: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
    window: &[f32],
    sync_soft: &[f32],
    num_frames: usize,
    sync_pattern: &[bool],
    sample_offset: usize,
) -> Result<Option<DetectionResult>> {
    let frames_per_block = sync::frames_per_block();
    let coded_bits_count = codec::CODED_BITS;
    let scan_end = num_frames - frames_per_block;
    let mut candidates: Vec<(usize, f32)> = (0..=scan_end)
        .map(|start| {
            let corr = correlate_sync(&sync_soft[start..start + SYNC_PATTERN_BITS], sync_pattern);
            (start, corr)
        })
        .filter(|&(_, corr)| corr > SYNC_THRESHOLD)
        .collect();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    for &(start, corr) in candidates.iter().take(MAX_DECODE_ATTEMPTS) {
        let data_frame_start = start + SYNC_PATTERN_BITS;
        let data_frame_end = data_frame_start + coded_bits_count;
        if data_frame_end > num_frames {
            continue;
        }

        // Pass 2: recompute data soft values with correct block positions
        let data_soft =
            compute_data_soft_values(buf, key, config, window, data_frame_start, coded_bits_count)?;

        let decoded_bits = codec::decode(&data_soft);
        if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
            return Ok(Some(DetectionResult {
                payload,
                confidence: corr,
                offset: start + sample_offset,
            }));
        }
    }

    // Fallback: try block starting at position 0 (no sync threshold)
    let data_frame_start = SYNC_PATTERN_BITS;
    let data_frame_end = data_frame_start + coded_bits_count;
    if data_frame_end <= num_frames {
        let data_soft =
            compute_data_soft_values(buf, key, config, window, data_frame_start, coded_bits_count)?;

        let decoded_bits = codec::decode(&data_soft);
        if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
            let corr = correlate_sync(&sync_soft[0..SYNC_PATTERN_BITS], sync_pattern);
            return Ok(Some(DetectionResult {
                payload,
                confidence: corr,
                offset: sample_offset,
            }));
        }
    }

    Ok(None)
}

/// Extract data soft values for the best sync candidate without running Viterbi.
///
/// Finds the sync pattern, then computes block-position-aware data soft values
/// for the best candidate. Returns `(data_soft, sync_frame_offset, sync_correlation)`
/// or `None` if no sync candidate exceeds the threshold.
///
/// Used by `StreamDetector` for soft combining across blocks.
pub fn extract_data_soft(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> Result<Option<(Vec<f32>, usize, f32)>> {
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

    if num_frames < frames_per_block {
        return Ok(None);
    }

    let window = hann_window(frame_size);
    let sync_soft = compute_soft_values_const_seed(samples, key, config, &window, num_frames)?;

    // Find best sync candidate
    let scan_end = num_frames - frames_per_block;
    let mut candidates: Vec<(usize, f32)> = (0..=scan_end)
        .map(|start| {
            let corr = correlate_sync(&sync_soft[start..start + SYNC_PATTERN_BITS], &sync_pattern);
            (start, corr)
        })
        .filter(|&(_, corr)| corr > SYNC_THRESHOLD)
        .collect();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    if let Some(&(start, corr)) = candidates.first() {
        let data_frame_start = start + SYNC_PATTERN_BITS;
        let data_frame_end = data_frame_start + coded_bits_count;
        if data_frame_end > num_frames {
            return Ok(None);
        }

        let data_soft = compute_data_soft_values(
            samples,
            key,
            config,
            &window,
            data_frame_start,
            coded_bits_count,
        )?;

        Ok(Some((data_soft, start, corr)))
    } else {
        Ok(None)
    }
}

/// Run Viterbi decode + CRC check on pre-computed soft values.
///
/// Returns the decoded `Payload` if successful, or `None` if Viterbi
/// decoding fails CRC validation. Used by `StreamDetector` to decode
/// combined soft values from multiple blocks.
pub fn try_decode_soft(soft: &[f32]) -> Option<Payload> {
    let decoded_bits = codec::decode(soft);
    Payload::decode_with_crc(&decoded_bits).ok()
}

/// Compute soft values using constant seed 0 for all frames.
///
/// Used for sync pattern finding. Sync frames were embedded with seed 0,
/// so these soft values are accurate for sync frames and noisy for data frames.
fn compute_soft_values_const_seed(
    buf: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
    window: &[f32],
    num_frames: usize,
) -> Result<Vec<f32>> {
    let hop_size = config.hop_size();
    let frame_size = config.frame_size;
    let mut fft = FftProcessor::new(frame_size)?;
    let mut soft_values = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = (start + frame_size).min(buf.len());
        let mut frame = vec![0.0f32; frame_size];
        frame[..end - start].copy_from_slice(&buf[start..end]);
        for (j, s) in frame.iter_mut().enumerate() {
            *s *= window[j];
        }
        fft.forward(&mut frame)?;
        // Constant seed 0 matches the sync frame embedding seed
        let soft = patchwork::detect_frame(fft.freq_bins(), key, 0, config);
        soft_values.push(soft);
    }

    Ok(soft_values)
}

/// Compute soft values for data frames using correct block positions.
///
/// `data_frame_start` is the local frame index where data begins.
/// Each data frame `j` uses block position `SYNC_PATTERN_BITS + j` as
/// the bin pair seed, matching the embedder's seed selection.
fn compute_data_soft_values(
    buf: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
    window: &[f32],
    data_frame_start: usize,
    num_data_frames: usize,
) -> Result<Vec<f32>> {
    let hop_size = config.hop_size();
    let frame_size = config.frame_size;
    let mut fft = FftProcessor::new(frame_size)?;
    let mut soft_values = Vec::with_capacity(num_data_frames);

    for j in 0..num_data_frames {
        let frame_idx = data_frame_start + j;
        let start = frame_idx * hop_size;
        let end = (start + frame_size).min(buf.len());
        let mut frame = vec![0.0f32; frame_size];
        frame[..end - start].copy_from_slice(&buf[start..end]);
        for (i, s) in frame.iter_mut().enumerate() {
            *s *= window[i];
        }
        fft.forward(&mut frame)?;
        // Block position for data frame j
        let block_pos = sync::bin_pair_seed(SYNC_PATTERN_BITS + j);
        let soft = patchwork::detect_frame(fft.freq_bins(), key, block_pos, config);
        soft_values.push(soft);
    }

    Ok(soft_values)
}

/// Detect watermarks in audio samples starting at a frame offset.
///
/// Uses a two-pass approach:
/// 1. Compute soft values with constant seed 0 (accurate for sync frames)
/// 2. Find sync pattern candidates
/// 3. Recompute data soft values with correct block-relative seeds
/// 4. Viterbi decode + CRC validation
///
/// `frame_offset` specifies the block-relative position of `samples[0]`
/// (used when the caller knows the alignment, e.g. from a previous detection).
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

    // Pass 1: compute soft values with constant seed 0 for sync finding.
    // Sync frames were embedded with seed 0, so these values are accurate
    // for sync frames and noisy for data frames.
    let sync_soft = compute_soft_values_const_seed(samples, key, config, &window, num_frames)?;

    if num_frames < frames_per_block {
        return Err(Error::NotDetected);
    }

    // Find sync candidates
    let scan_end = num_frames - frames_per_block;
    let mut candidates: Vec<(usize, f32)> = (0..=scan_end)
        .map(|start| {
            let corr = correlate_sync(&sync_soft[start..start + SYNC_PATTERN_BITS], &sync_pattern);
            (start, corr)
        })
        .filter(|&(_, corr)| corr > SYNC_THRESHOLD)
        .collect();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Pass 2: recompute data soft values at the best sync candidates
    for &(start, corr) in candidates.iter().take(MAX_DECODE_ATTEMPTS) {
        let data_frame_start = start + SYNC_PATTERN_BITS;
        let data_frame_end = data_frame_start + coded_bits_count;
        if data_frame_end > num_frames {
            continue;
        }

        let data_soft = compute_data_soft_values(
            samples,
            key,
            config,
            &window,
            data_frame_start,
            coded_bits_count,
        )?;

        let decoded_bits = codec::decode(&data_soft);
        if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
            return Ok(vec![DetectionResult {
                payload,
                confidence: corr,
                offset: start + frame_offset as usize,
            }]);
        }
    }

    // Fallback: try block starting at position 0 directly (no sync threshold)
    let data_frame_start = SYNC_PATTERN_BITS;
    let data_frame_end = data_frame_start + coded_bits_count;
    if data_frame_end <= num_frames {
        let data_soft = compute_data_soft_values(
            samples,
            key,
            config,
            &window,
            data_frame_start,
            coded_bits_count,
        )?;

        let decoded_bits = codec::decode(&data_soft);
        if let Ok(payload) = Payload::decode_with_crc(&decoded_bits) {
            let corr = correlate_sync(&sync_soft[0..SYNC_PATTERN_BITS], &sync_pattern);
            return Ok(vec![DetectionResult {
                payload,
                confidence: corr,
                offset: frame_offset as usize,
            }]);
        }
    }

    Err(Error::NotDetected)
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
