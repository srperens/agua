//! Acoustic channel simulation tests.
//!
//! Simulates degradations that occur in a speaker → air → microphone path:
//! low-pass filtering, additive noise, and sample-rate jitter.
//! These tests are marked `#[ignore]` since they exercise robustness limits
//! and may be sensitive to parameter tuning.

use agua_core::{Payload, PreProcessor, StreamDetector, WatermarkConfig, WatermarkKey};

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

// ── New acoustic channel simulation helpers ──────────────────────────────────

/// Deterministic xorshift32 PRNG.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Generate a synthetic room impulse response.
///
/// Models direct path + early reflections (5-50ms) + exponential decay tail.
/// `rt60_ms` is the approximate RT60 in milliseconds.
fn generate_rir(sample_rate: u32, rt60_ms: u32, seed: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let rt60_samples = (rt60_ms as f32 / 1000.0 * sr) as usize;
    let len = rt60_samples.max(1);
    let mut rir = vec![0.0f32; len];

    // Direct path
    rir[0] = 1.0;

    // Early reflections: 6 reflections between 5-50ms at -10 to -20 dB
    // (typical for a small room at ~1m speaker-mic distance)
    let mut rng = seed;
    for _ in 0..6 {
        let delay_ms = 5.0 + (xorshift32(&mut rng) as f32 / u32::MAX as f32) * 45.0;
        let delay_samples = (delay_ms / 1000.0 * sr) as usize;
        if delay_samples < len {
            let amp = 0.05 + (xorshift32(&mut rng) as f32 / u32::MAX as f32) * 0.15;
            let sign = if xorshift32(&mut rng) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            rir[delay_samples] += sign * amp;
        }
    }

    // Exponential decay tail (diffuse reverb)
    let decay_rate = -6.91 / rt60_samples as f32; // -60 dB at RT60
    for (i, sample) in rir.iter_mut().enumerate().skip(1) {
        let envelope = (decay_rate * i as f32).exp();
        let noise = (xorshift32(&mut rng) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *sample += noise * envelope * 0.01;
    }

    // Normalize so peak is 1.0
    let peak = rir.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        for s in rir.iter_mut() {
            *s /= peak;
        }
    }
    rir
}

/// Linear convolution, output truncated to signal length.
fn convolve(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let mut output = vec![0.0f32; n];
    for (i, out) in output.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (j, &k) in kernel.iter().enumerate() {
            if i >= j {
                acc += signal[i - j] * k;
            }
        }
        *out = acc;
    }
    output
}

/// Biquad filter (Direct Form I) for simulation use.
struct SimBiquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl SimBiquad {
    fn highpass(sample_rate: f32, cutoff: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * cutoff / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0_f32.sqrt());
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 + cos_w0) / 2.0) / a0,
            b1: (-(1.0 + cos_w0)) / a0,
            b2: ((1.0 + cos_w0) / 2.0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn lowpass(sample_rate: f32, cutoff: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * cutoff / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0_f32.sqrt());
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 - cos_w0) / 2.0) / a0,
            b1: (1.0 - cos_w0) / a0,
            b2: ((1.0 - cos_w0) / 2.0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Peaking EQ biquad.
    fn peaking_eq(sample_rate: f32, center_hz: f32, q: f32, gain_db: f32) -> Self {
        let a = 10.0f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * center_hz / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        Self {
            b0: (1.0 + alpha * a) / a0,
            b1: (-2.0 * cos_w0) / a0,
            b2: (1.0 - alpha * a) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha / a) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}

/// Simulate speaker + mic frequency response.
///
/// Cascaded biquads: HP 150Hz (small speaker rolloff) + peaking EQ at
/// 2500Hz/Q=2/+6dB (resonance) + LP 12kHz (mic rolloff).
fn apply_speaker_mic_response(samples: &mut [f32], sample_rate: u32) {
    let sr = sample_rate as f32;
    let mut hp = SimBiquad::highpass(sr, 150.0);
    let mut peak = SimBiquad::peaking_eq(sr, 2500.0, 2.0, 6.0);
    let mut lp = SimBiquad::lowpass(sr, 12000.0);

    hp.process(samples);
    peak.process(samples);
    lp.process(samples);
}

/// Simulate automatic gain control (AGC).
///
/// Envelope follower compressor with attack 10ms, release 200ms,
/// target RMS 0.1, gain range 0.1x-20x.
fn apply_agc(samples: &mut [f32], sample_rate: u32) {
    let sr = sample_rate as f32;
    let attack_coeff = 1.0 - (-1.0 / (0.01 * sr)).exp();
    let release_coeff = 1.0 - (-1.0 / (0.2 * sr)).exp();
    let target = 0.1f32;
    let min_gain = 0.1f32;
    let max_gain = 20.0f32;

    let mut envelope = target; // Start at target to avoid initial transient

    for s in samples.iter_mut() {
        let abs_s = s.abs();
        let coeff = if abs_s > envelope {
            attack_coeff
        } else {
            release_coeff
        };
        envelope += coeff * (abs_s - envelope);

        let gain = (target / envelope.max(1e-6)).clamp(min_gain, max_gain);
        *s *= gain;
    }
}

#[test]
#[ignore]
fn acoustic_reverb_150ms() {
    let config = WatermarkConfig::acoustic();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    // 30s for reliable detection through reverb
    let num_samples = 48000 * 30;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    // Apply room impulse response (RT60 = 150ms, typical small/treated room)
    let rir = generate_rir(config.sample_rate, 150, 0x1234);
    let degraded = convolve(&audio, &rir);

    let results = agua_core::detect(&degraded, &key, &config);
    match results {
        Ok(r) => {
            assert_eq!(r[0].payload, payload, "payload mismatch after reverb");
            println!(
                "reverb RT60=150ms: PASS (confidence: {:.4})",
                r[0].confidence
            );
        }
        Err(e) => panic!("reverb RT60=150ms: watermark not detected: {e}"),
    }
}

#[test]
#[ignore]
fn acoustic_speaker_mic_eq() {
    let config = WatermarkConfig::acoustic();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33,
        0x44,
    ]);

    let num_samples = 48000 * 25;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    // Apply speaker/mic frequency response
    apply_speaker_mic_response(&mut audio, config.sample_rate);

    let results = agua_core::detect(&audio, &key, &config);
    match results {
        Ok(r) => {
            assert_eq!(r[0].payload, payload, "payload mismatch after EQ");
            println!("speaker/mic EQ: PASS (confidence: {:.4})", r[0].confidence);
        }
        Err(e) => panic!("speaker/mic EQ: watermark not detected: {e}"),
    }
}

#[test]
#[ignore]
fn acoustic_agc() {
    let config = WatermarkConfig::acoustic();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([0xAA; 16]);

    let num_samples = 48000 * 25;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    // Apply AGC
    apply_agc(&mut audio, config.sample_rate);

    let results = agua_core::detect(&audio, &key, &config);
    match results {
        Ok(r) => {
            assert_eq!(r[0].payload, payload, "payload mismatch after AGC");
            println!("AGC: PASS (confidence: {:.4})", r[0].confidence);
        }
        Err(e) => panic!("AGC: watermark not detected: {e}"),
    }
}

#[test]
#[ignore]
fn acoustic_realistic_channel() {
    let config = WatermarkConfig::acoustic();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    // 30s for reliable detection through combined degradation
    let num_samples = 48000 * 30;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    // Full realistic channel: RIR → speaker/mic EQ → AGC → noise → jitter → preprocess
    let mut degraded = convolve(&audio, &generate_rir(config.sample_rate, 150, 0xABCD));
    apply_speaker_mic_response(&mut degraded, config.sample_rate);
    apply_agc(&mut degraded, config.sample_rate);
    add_white_noise(&mut degraded, 20.0);
    let mut degraded = resample_jitter(&degraded, 1.001);

    // Apply preprocessing (as the WASM detector would)
    let mut preprocessor = PreProcessor::new(config.sample_rate);
    preprocessor.process(&mut degraded);

    let results = agua_core::detect(&degraded, &key, &config);
    match results {
        Ok(r) => {
            assert_eq!(
                r[0].payload, payload,
                "payload mismatch after realistic channel"
            );
            println!(
                "realistic acoustic channel: PASS (confidence: {:.4})",
                r[0].confidence
            );
        }
        Err(e) => panic!("realistic acoustic channel: watermark not detected: {e}"),
    }
}

#[test]
#[ignore]
fn acoustic_streaming_arbitrary_offset() {
    let config = WatermarkConfig::acoustic();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    // Embed into 30s of audio
    let num_samples = 48000 * 30;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    // Apply degradation (moderate: small room reverb + EQ + noise)
    let mut degraded = convolve(&audio, &generate_rir(config.sample_rate, 150, 0x5678));
    apply_speaker_mic_response(&mut degraded, config.sample_rate);
    add_white_noise(&mut degraded, 25.0);

    // Start from arbitrary offset (~3 seconds in, NOT hop-aligned)
    let offset = 48000 * 3 + 137;
    let offset_audio = &degraded[offset..];

    // Feed through PreProcessor + StreamDetector in 128-sample chunks
    let mut preprocessor = PreProcessor::new(config.sample_rate);
    let mut detector = StreamDetector::new(&key, &config).unwrap();
    let mut results = Vec::new();

    for chunk in offset_audio.chunks(128) {
        let mut buf = chunk.to_vec();
        preprocessor.process(&mut buf);
        results.extend(detector.process(&buf));
    }
    if let Some(r) = detector.finalize() {
        results.push(r);
    }

    assert!(
        !results.is_empty(),
        "streaming detector failed to detect watermark with offset + acoustic degradation"
    );
    assert_eq!(results[0].payload, payload, "payload mismatch");
    println!(
        "streaming arbitrary offset + acoustic: PASS (confidence: {:.4})",
        results[0].confidence
    );
}
