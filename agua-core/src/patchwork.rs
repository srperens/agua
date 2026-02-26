use realfft::num_complex::Complex32;

use crate::config::WatermarkConfig;
use crate::key::WatermarkKey;

/// Embed a single bit into a frequency-domain frame using power-law patchwork.
///
/// For each bin pair (a, b), the magnitude is scaled via `mag^(1 +/- delta)`:
/// - bit=true:  boost `a` (`mag_a^(1+delta)`), attenuate `b` (`mag_b^(1-delta)`)
/// - bit=false: attenuate `a` (`mag_a^(1-delta)`), boost `b` (`mag_b^(1+delta)`)
///
/// Complex phase is preserved; only magnitudes change.
pub fn embed_frame(
    freq_bins: &mut [Complex32],
    bit: bool,
    key: &WatermarkKey,
    frame_index: u32,
    config: &WatermarkConfig,
) {
    let (min_bin, max_bin) = config.effective_bin_range();
    // Per-frame bin pairs give frequency diversity across the block, which
    // is critical for robust detection across all keys. The frame_index
    // should be the block-relative position (0..block_len-1) so that the
    // detector can reconstruct the correct seed after finding the sync.
    let pairs = key.generate_bin_pairs(
        frame_index,
        config.num_bin_pairs,
        min_bin,
        max_bin,
        config.bin_spacing,
    );
    let delta = config.strength;

    for (a, b) in pairs {
        if a >= freq_bins.len() || b >= freq_bins.len() {
            continue;
        }

        let mag_a = freq_bins[a].norm();
        let mag_b = freq_bins[b].norm();

        if mag_a < 1e-10 || mag_b < 1e-10 {
            continue;
        }

        // Power-law: use |ln(mag)| so the multiplier always pushes magnitude
        // in the correct direction regardless of whether mag > 1 or mag < 1.
        // For boost (exp > 0): multiplier = exp(delta * |ln(mag)|) > 1 always.
        // For attenuate (exp < 0): multiplier = exp(-delta * |ln(mag)|) < 1 always.
        let (exp_a, exp_b) = if bit {
            (delta, -delta)
        } else {
            (-delta, delta)
        };

        freq_bins[a] *= (exp_a * mag_a.ln().abs()).exp();
        freq_bins[b] *= (exp_b * mag_b.ln().abs()).exp();
    }
}

/// Detect a single bit from a frequency-domain frame using log-ratio patchwork.
///
/// Computes the average `ln(mag_a / mag_b)` across all bin pairs.
/// Returns a soft value: positive suggests bit=1, negative suggests bit=0.
pub fn detect_frame(
    freq_bins: &[Complex32],
    key: &WatermarkKey,
    frame_index: u32,
    config: &WatermarkConfig,
) -> f32 {
    let (min_bin, max_bin) = config.effective_bin_range();
    // frame_index selects which bin pairs to probe. When matching the
    // embedder's block-relative position, this gives accurate soft values.
    // When using constant seed 0 (for sync finding), soft values are
    // noisier but still usable for sync correlation.
    let pairs = key.generate_bin_pairs(
        frame_index,
        config.num_bin_pairs,
        min_bin,
        max_bin,
        config.bin_spacing,
    );

    let mut sum = 0.0f32;
    let mut count = 0usize;

    for (a, b) in pairs {
        if a >= freq_bins.len() || b >= freq_bins.len() {
            continue;
        }
        let mag_a = freq_bins[a].norm();
        let mag_b = freq_bins[b].norm();

        if mag_a < 1e-10 || mag_b < 1e-10 {
            continue;
        }

        sum += (mag_a / mag_b).ln();
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    sum / count as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::FftProcessor;

    /// Create a broadband test signal with energy in the target frequency range.
    /// Harmonics at FFT bins 10..100 cover the default 860-4300 Hz band.
    fn make_test_signal(size: usize) -> Vec<f32> {
        let mut signal = vec![0.0f32; size];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f32 / size as f32;
            for k in 10..100 {
                let amp = 1.0 / (k as f32).sqrt();
                *sample += amp * (2.0 * std::f32::consts::PI * k as f32 * t).sin();
            }
        }
        signal
    }

    #[test]
    fn embed_detect_single_frame_bit_true() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let mut fft = FftProcessor::new(config.frame_size).unwrap();

        let mut signal = make_test_signal(config.frame_size);

        // Get baseline stat without watermark
        let mut baseline_signal = signal.clone();
        fft.forward(&mut baseline_signal).unwrap();
        let baseline_stat = detect_frame(fft.freq_bins(), &key, 0, &config);

        fft.forward(&mut signal).unwrap();
        embed_frame(fft.freq_bins_mut(), true, &key, 0, &config);
        let stat = detect_frame(fft.freq_bins(), &key, 0, &config);

        // Embedded stat should be more positive than baseline
        assert!(
            stat > baseline_stat,
            "expected stat ({stat}) > baseline ({baseline_stat}) for bit=true"
        );
    }

    #[test]
    fn embed_detect_single_frame_bit_false() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let mut fft = FftProcessor::new(config.frame_size).unwrap();

        let mut signal = make_test_signal(config.frame_size);

        // Get baseline stat
        let mut baseline_signal = signal.clone();
        fft.forward(&mut baseline_signal).unwrap();
        let baseline_stat = detect_frame(fft.freq_bins(), &key, 0, &config);

        fft.forward(&mut signal).unwrap();
        embed_frame(fft.freq_bins_mut(), false, &key, 0, &config);
        let stat = detect_frame(fft.freq_bins(), &key, 0, &config);

        // Embedded stat should be more negative than baseline
        assert!(
            stat < baseline_stat,
            "expected stat ({stat}) < baseline ({baseline_stat}) for bit=false"
        );
    }

    #[test]
    fn embed_shifts_stat_symmetrically() {
        let config = WatermarkConfig {
            strength: 0.05,
            ..WatermarkConfig::default()
        };
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();

        let signal = make_test_signal(config.frame_size);

        // Embed bit=true
        let mut fft = FftProcessor::new(config.frame_size).unwrap();
        let mut sig_true = signal.clone();
        fft.forward(&mut sig_true).unwrap();
        let baseline = detect_frame(fft.freq_bins(), &key, 0, &config);
        embed_frame(fft.freq_bins_mut(), true, &key, 0, &config);
        let stat_true = detect_frame(fft.freq_bins(), &key, 0, &config);

        // Embed bit=false
        let mut sig_false = signal.clone();
        fft.forward(&mut sig_false).unwrap();
        embed_frame(fft.freq_bins_mut(), false, &key, 0, &config);
        let stat_false = detect_frame(fft.freq_bins(), &key, 0, &config);

        let shift_true = stat_true - baseline;
        let shift_false = stat_false - baseline;

        // Shifts should be in opposite directions
        assert!(
            shift_true > 0.0,
            "bit=true should shift stat positive: {shift_true}"
        );
        assert!(
            shift_false < 0.0,
            "bit=false should shift stat negative: {shift_false}"
        );
    }

    #[test]
    fn round_trip_preserves_signal() {
        let config = WatermarkConfig::default();
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let mut fft = FftProcessor::new(config.frame_size).unwrap();

        let original = make_test_signal(config.frame_size);
        let mut signal = original.clone();

        fft.forward(&mut signal).unwrap();
        embed_frame(fft.freq_bins_mut(), true, &key, 0, &config);
        fft.inverse(&mut signal).unwrap();
        fft.normalize(&mut signal);

        // Signal should be close to original (small perturbation).
        // Power-law encoding perturbs more than linear for large FFT magnitudes.
        // With default strength 0.1 the peak diff can reach ~3.0 on synthetic signals
        // with high bin magnitudes. Real audio at 0.5 peak level stays well under 5.0.
        let max_diff: f32 = original
            .iter()
            .zip(signal.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 5.0,
            "watermark perturbation too large: {max_diff}"
        );
    }
}
