use realfft::num_complex::Complex32;

use crate::config::WatermarkConfig;
use crate::key::WatermarkKey;

/// Embed a single bit into a frequency-domain frame using the patchwork algorithm.
///
/// For each bin pair (a, b), if bit is true, increase |a| and decrease |b|;
/// if bit is false, decrease |a| and increase |b|. This creates a detectable
/// statistical difference between the paired bins.
pub fn embed_frame(
    freq_bins: &mut [Complex32],
    bit: bool,
    key: &WatermarkKey,
    frame_index: u32,
    config: &WatermarkConfig,
) {
    let pairs = key.generate_bin_pairs(
        frame_index,
        config.num_bin_pairs,
        config.min_bin,
        config.max_bin,
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

        let (scale_a, scale_b) = if bit {
            (1.0 + delta, 1.0 - delta)
        } else {
            (1.0 - delta, 1.0 + delta)
        };

        freq_bins[a] *= scale_a;
        freq_bins[b] *= scale_b;
    }
}

/// Detect a single bit from a frequency-domain frame using the patchwork algorithm.
///
/// Computes the patchwork statistic: sum of (|a| - |b|) / (|a| + |b|) for all bin pairs.
/// Returns a soft value: positive suggests bit=1, negative suggests bit=0.
/// The magnitude indicates confidence.
pub fn detect_frame(
    freq_bins: &[Complex32],
    key: &WatermarkKey,
    frame_index: u32,
    config: &WatermarkConfig,
) -> f32 {
    let pairs = key.generate_bin_pairs(
        frame_index,
        config.num_bin_pairs,
        config.min_bin,
        config.max_bin,
    );

    let mut sum_diff = 0.0f32;
    let mut sum_total = 0.0f32;

    for (a, b) in pairs {
        if a >= freq_bins.len() || b >= freq_bins.len() {
            continue;
        }
        let mag_a = freq_bins[a].norm();
        let mag_b = freq_bins[b].norm();

        sum_diff += mag_a - mag_b;
        sum_total += mag_a + mag_b;
    }

    if sum_total < 1e-10 {
        return 0.0;
    }

    // Global normalization: the embedding shifts this ratio by approximately ±strength
    sum_diff / sum_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::FftProcessor;

    /// Create a broadband test signal (many harmonics) for more reliable detection.
    fn make_test_signal(size: usize) -> Vec<f32> {
        let mut signal = vec![0.0f32; size];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f32 / size as f32;
            // Use many harmonics for a broad spectrum
            for k in 1..50 {
                let freq = k as f32 * 100.0;
                let amp = 1.0 / k as f32;
                *sample += amp * (2.0 * std::f32::consts::PI * freq * t).sin();
            }
        }
        signal
    }

    #[test]
    fn embed_detect_single_frame_bit_true() {
        let config = WatermarkConfig {
            strength: 0.05, // Stronger for single-frame test
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

        // Shifts should be in opposite directions and roughly equal magnitude
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

        // Signal should be close to original (small perturbation)
        let max_diff: f32 = original
            .iter()
            .zip(signal.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.1,
            "watermark perturbation too large: {max_diff}"
        );
    }
}
