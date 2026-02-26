//! Audio pre-processing for microphone input before watermark detection.
//!
//! Applies bandpass filtering (removes DC/rumble and high-frequency noise)
//! and RMS normalization to handle AGC level variations. All processing
//! is WASM-compatible (pure math, no std I/O or system calls).

/// Second-order IIR (biquad) filter coefficients.
#[derive(Debug, Clone)]
struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

/// Biquad filter state (Direct Form I).
#[derive(Debug, Clone)]
struct BiquadState {
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadState {
    fn new() -> Self {
        Self {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, coeffs: &BiquadCoeffs, x: f32) -> f32 {
        let y = coeffs.b0 * x + coeffs.b1 * self.x1 + coeffs.b2 * self.x2
            - coeffs.a1 * self.y1
            - coeffs.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Compute Butterworth highpass biquad coefficients.
fn butterworth_highpass(sample_rate: f32, cutoff: f32) -> BiquadCoeffs {
    let w0 = 2.0 * core::f32::consts::PI * cutoff / sample_rate;
    let cos_w0 = w0.cos();
    let alpha = w0.sin() / (2.0_f32.sqrt());

    let a0 = 1.0 + alpha;
    BiquadCoeffs {
        b0: ((1.0 + cos_w0) / 2.0) / a0,
        b1: (-(1.0 + cos_w0)) / a0,
        b2: ((1.0 + cos_w0) / 2.0) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

/// Compute Butterworth lowpass biquad coefficients.
fn butterworth_lowpass(sample_rate: f32, cutoff: f32) -> BiquadCoeffs {
    let w0 = 2.0 * core::f32::consts::PI * cutoff / sample_rate;
    let cos_w0 = w0.cos();
    let alpha = w0.sin() / (2.0_f32.sqrt());

    let a0 = 1.0 + alpha;
    BiquadCoeffs {
        b0: ((1.0 - cos_w0) / 2.0) / a0,
        b1: (1.0 - cos_w0) / a0,
        b2: ((1.0 - cos_w0) / 2.0) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

/// Streaming audio pre-processor for mic input.
///
/// Applies cascaded highpass + lowpass biquad filters and EMA-based RMS
/// normalization. Designed for the detector pipeline: removes DC offset,
/// low-frequency rumble, high-frequency noise, and normalizes level to
/// compensate for AGC variations.
pub struct PreProcessor {
    hp_coeffs: BiquadCoeffs,
    hp_state: BiquadState,
    lp_coeffs: BiquadCoeffs,
    lp_state: BiquadState,
    /// Exponential moving average of squared samples for RMS estimation.
    ema_sq: f32,
    /// EMA smoothing coefficient (per sample).
    ema_alpha: f32,
    /// Target RMS level.
    target_rms: f32,
    /// Maximum gain to prevent noise amplification.
    max_gain: f32,
}

impl PreProcessor {
    /// Create a new pre-processor for the given sample rate.
    ///
    /// - Highpass at 200 Hz (removes DC and low-frequency rumble)
    /// - Lowpass at 8000 Hz (removes high-frequency noise above watermark band)
    /// - RMS normalization: target 0.05, 0.5s time constant, max gain 20x
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate as f32;
        let hp_coeffs = butterworth_highpass(sr, 200.0);
        let lp_coeffs = butterworth_lowpass(sr, 8000.0);

        // EMA time constant: 0.5 seconds
        let ema_alpha = 1.0 - (-1.0 / (0.5 * sr)).exp();

        Self {
            hp_coeffs,
            hp_state: BiquadState::new(),
            lp_coeffs,
            lp_state: BiquadState::new(),
            ema_sq: 0.0025, // Initial RMS estimate = 0.05
            ema_alpha,
            target_rms: 0.05,
            max_gain: 20.0,
        }
    }

    /// Process audio samples in-place.
    ///
    /// Applies bandpass filter and RMS normalization to each sample.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            // Cascaded biquad filters: HP → LP
            let filtered = self.hp_state.process(&self.hp_coeffs, *s);
            let filtered = self.lp_state.process(&self.lp_coeffs, filtered);

            // Update EMA of squared signal
            self.ema_sq += self.ema_alpha * (filtered * filtered - self.ema_sq);

            // Compute gain from RMS
            let rms = self.ema_sq.sqrt().max(1e-6);
            let gain = (self.target_rms / rms).min(self.max_gain);

            *s = filtered * gain;
        }
    }

    /// Reset the pre-processor state.
    pub fn reset(&mut self) {
        self.hp_state = BiquadState::new();
        self.lp_state = BiquadState::new();
        self.ema_sq = 0.0025;
    }
}

/// One-shot convenience function: pre-process audio samples.
///
/// Creates a `PreProcessor`, runs it over the entire buffer, and returns
/// the processed audio. For streaming use, create a `PreProcessor` directly.
pub fn preprocess(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let mut output = samples.to_vec();
    let mut proc = PreProcessor::new(sample_rate);
    proc.process(&mut output);
    output
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
                *sample += amp * (2.0 * core::f32::consts::PI * freq * t + k as f32).sin();
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
    fn preprocess_preserves_watermark() {
        let config = crate::config::WatermarkConfig {
            strength: 0.05,
            ..crate::config::WatermarkConfig::default()
        };
        let key = crate::key::WatermarkKey::new(&[42u8; 16]).unwrap();
        let payload = crate::payload::Payload::new([0xDE; 16]);

        let mut audio = make_test_audio(48000 * 20, config.sample_rate);
        crate::embed::embed(&mut audio, &payload, &key, &config).unwrap();

        // Pre-process
        let processed = preprocess(&audio, config.sample_rate);

        // Detect in preprocessed audio
        let results = crate::detect::detect(&processed, &key, &config);
        assert!(
            results.is_ok(),
            "watermark not detected after preprocessing"
        );
        assert_eq!(results.unwrap()[0].payload, payload);
    }

    #[test]
    fn preprocess_removes_dc_offset() {
        let sample_rate = 48000u32;
        let mut audio = make_test_audio(48000, sample_rate);
        // Add DC offset
        for s in audio.iter_mut() {
            *s += 0.5;
        }

        let processed = preprocess(&audio, sample_rate);

        // Check that the mean of the second half (after filter settles) is near zero
        let n = processed.len();
        let mean: f32 = processed[n / 2..].iter().sum::<f32>() / (n / 2) as f32;
        assert!(mean.abs() < 0.05, "DC offset not removed: mean={mean:.4}");
    }

    #[test]
    fn preprocess_normalizes_level() {
        let sample_rate = 48000u32;
        // Use 5 seconds of audio so the EMA has time to converge
        let mut audio = make_test_audio(48000 * 5, sample_rate);
        // Scale to low level (simulates quiet mic capture at ~-30 dB FS)
        for s in audio.iter_mut() {
            *s *= 0.03;
        }

        let processed = preprocess(&audio, sample_rate);

        // Check RMS of the last second (after EMA has converged)
        let last_sec = &processed[processed.len() - sample_rate as usize..];
        let rms: f32 = (last_sec.iter().map(|s| s * s).sum::<f32>() / last_sec.len() as f32).sqrt();
        assert!(
            rms > 0.02 && rms < 0.15,
            "normalization failed: rms={rms:.4}"
        );
    }
}
