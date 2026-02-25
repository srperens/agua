//! Acoustic channel simulation tests.
//!
//! Simulates degradations that occur in a speaker → air → microphone path:
//! low-pass filtering, additive noise, and sample-rate jitter.
//! These tests are marked `#[ignore]` since they exercise robustness limits
//! and may be sensitive to parameter tuning.

use agua_core::{Payload, WatermarkConfig, WatermarkKey};

/// Generate broadband test audio with energy across many frequencies.
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

/// Simple low-pass filter using a windowed-sinc FIR.
///
/// `cutoff_hz` is the -3dB cutoff frequency. The filter uses a Kaiser-windowed
/// sinc kernel of `tap_count` taps (must be odd).
fn lowpass_filter(samples: &[f32], sample_rate: u32, cutoff_hz: f32, tap_count: usize) -> Vec<f32> {
    let tap_count = if tap_count.is_multiple_of(2) {
        tap_count + 1
    } else {
        tap_count
    };
    let half = tap_count / 2;
    let fc = cutoff_hz / sample_rate as f32;

    // Windowed-sinc kernel
    let mut kernel = vec![0.0f32; tap_count];
    for (i, k) in kernel.iter_mut().enumerate() {
        let n = i as f32 - half as f32;
        let sinc = if n.abs() < 1e-10 {
            2.0 * std::f32::consts::PI * fc
        } else {
            (2.0 * std::f32::consts::PI * fc * n).sin() / n
        };
        // Blackman window for good stop-band attenuation
        let w = 0.42 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (tap_count - 1) as f32).cos()
            + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / (tap_count - 1) as f32).cos();
        *k = sinc * w;
    }

    // Normalize kernel
    let sum: f32 = kernel.iter().sum();
    if sum.abs() > 1e-10 {
        for k in kernel.iter_mut() {
            *k /= sum;
        }
    }

    // Convolve (same length, zero-padded edges)
    let n = samples.len();
    let mut output = vec![0.0f32; n];
    for (i, out) in output.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (j, &k) in kernel.iter().enumerate() {
            let idx = i as isize + j as isize - half as isize;
            if idx >= 0 && (idx as usize) < n {
                acc += samples[idx as usize] * k;
            }
        }
        *out = acc;
    }
    output
}

/// Add white noise at a given SNR (in dB) relative to signal power.
fn add_white_noise(samples: &mut [f32], snr_db: f32) {
    // Compute signal power
    let signal_power: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;

    // Noise power for desired SNR: SNR = 10*log10(P_signal/P_noise)
    let noise_power = signal_power / 10.0f32.powf(snr_db / 10.0);
    let noise_std = noise_power.sqrt();

    // Simple PRNG (xorshift32) for reproducible noise
    let mut state: u32 = 0xDEAD_BEEF;
    for s in samples.iter_mut() {
        // Generate two uniform randoms for Box-Muller transform
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let u1 = (state as f32) / u32::MAX as f32;

        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let u2 = (state as f32) / u32::MAX as f32;

        // Box-Muller: generate normal(0, noise_std)
        let u1_clamped = u1.max(1e-10);
        let noise =
            noise_std * (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        *s += noise;
    }
}

/// Linear interpolation resample: change sample rate by a ratio then back.
///
/// Simulates the clock drift / jitter in a speaker→mic path.
fn resample_jitter(samples: &[f32], ratio: f32) -> Vec<f32> {
    // Phase 1: resample to ratio * original rate
    let intermediate_len = (samples.len() as f32 * ratio) as usize;
    let mut intermediate = vec![0.0f32; intermediate_len];
    for (i, val) in intermediate.iter_mut().enumerate() {
        let src_pos = i as f32 / ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f32;
        if idx + 1 < samples.len() {
            *val = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac;
        } else if idx < samples.len() {
            *val = samples[idx];
        }
    }

    // Phase 2: resample back to original rate
    let output_len = samples.len();
    let inv_ratio = intermediate_len as f32 / output_len as f32;
    let mut output = vec![0.0f32; output_len];
    for (i, out) in output.iter_mut().enumerate() {
        let src_pos = i as f32 * inv_ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f32;
        if idx + 1 < intermediate.len() {
            *out = intermediate[idx] * (1.0 - frac) + intermediate[idx + 1] * frac;
        } else if idx < intermediate.len() {
            *out = intermediate[idx];
        }
    }
    output
}

#[test]
#[ignore]
fn acoustic_lowpass_8khz() {
    let config = WatermarkConfig::robust();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    let num_samples = 48000 * 25;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    // Simulate speaker frequency response: 8kHz low-pass
    let filtered = lowpass_filter(&audio, config.sample_rate, 8000.0, 127);

    let results = agua_core::detect(&filtered, &key, &config);
    match results {
        Ok(r) => {
            assert_eq!(r[0].payload, payload, "payload mismatch after low-pass");
            println!("low-pass 8kHz: PASS (confidence: {:.4})", r[0].confidence);
        }
        Err(e) => panic!("low-pass 8kHz: watermark not detected: {e}"),
    }
}

#[test]
#[ignore]
fn acoustic_white_noise_20db() {
    let config = WatermarkConfig::robust();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33,
        0x44,
    ]);

    let num_samples = 48000 * 25;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    // Add white noise at 20dB SNR
    add_white_noise(&mut audio, 20.0);

    let results = agua_core::detect(&audio, &key, &config);
    match results {
        Ok(r) => {
            assert_eq!(r[0].payload, payload, "payload mismatch after noise");
            println!(
                "white noise 20dB SNR: PASS (confidence: {:.4})",
                r[0].confidence
            );
        }
        Err(e) => panic!("white noise 20dB SNR: watermark not detected: {e}"),
    }
}

#[test]
#[ignore]
fn acoustic_resample_jitter() {
    let config = WatermarkConfig::robust();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([0x55; 16]);

    let num_samples = 48000 * 25;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    // Simulate slight clock drift: resample to 48048 Hz and back (0.1% jitter)
    let jittered = resample_jitter(&audio, 1.001);

    let results = agua_core::detect(&jittered, &key, &config);
    match results {
        Ok(r) => {
            assert_eq!(r[0].payload, payload, "payload mismatch after resample");
            println!(
                "resample jitter 0.1%: PASS (confidence: {:.4})",
                r[0].confidence
            );
        }
        Err(e) => panic!("resample jitter 0.1%: watermark not detected: {e}"),
    }
}

#[test]
#[ignore]
fn acoustic_combined_channel() {
    let config = WatermarkConfig::robust();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    let num_samples = 48000 * 25;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    // Full acoustic channel simulation:
    // 1. Low-pass filter at 8kHz (speaker/mic frequency response)
    let mut degraded = lowpass_filter(&audio, config.sample_rate, 8000.0, 127);
    // 2. Add white noise at 20dB SNR (ambient noise)
    add_white_noise(&mut degraded, 20.0);
    // 3. Resample jitter (clock drift)
    let degraded = resample_jitter(&degraded, 1.001);

    let results = agua_core::detect(&degraded, &key, &config);
    match results {
        Ok(r) => {
            assert_eq!(
                r[0].payload, payload,
                "payload mismatch after full acoustic sim"
            );
            println!(
                "combined acoustic channel: PASS (confidence: {:.4})",
                r[0].confidence
            );
        }
        Err(e) => panic!("combined acoustic channel: watermark not detected: {e}"),
    }
}
