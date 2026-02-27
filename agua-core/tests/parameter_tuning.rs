//! Parameter tuning sweep: strength x codec combinations.
//!
//! Requires `ffmpeg` on the system PATH. Marked `#[ignore]`.
//! Run with: `cargo test parameter_tuning -- --ignored --nocapture`

use std::path::Path;
use std::process::Command;

use agua_core::{Payload, WatermarkConfig, WatermarkKey};

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

fn write_wav_i16(path: &Path, samples: &[f32], sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("failed to create WAV writer");
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let val = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(val).expect("failed to write sample");
    }
    writer.finalize().expect("failed to finalize WAV");
}

fn read_wav_f32(path: &Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("failed to open WAV");
    let spec = reader.spec();
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.expect("failed to read sample"))
            .collect(),
        hound::SampleFormat::Int => {
            let max = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.expect("failed to read sample") as f32 / max)
                .collect()
        }
    }
}

fn ffmpeg_round_trip(input_wav: &Path, output_wav: &Path, codec: &str, bitrate: &str) -> bool {
    let ext = match codec {
        "libmp3lame" => "mp3",
        "aac" => "m4a",
        "libopus" => "ogg",
        _ => return false,
    };

    let lossy_path = input_wav.with_extension(ext);

    let encode = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input_wav.to_str().unwrap(),
            "-c:a",
            codec,
            "-b:a",
            bitrate,
            lossy_path.to_str().unwrap(),
        ])
        .output();

    let Ok(encode_out) = encode else {
        return false;
    };
    if !encode_out.status.success() {
        return false;
    }

    let decode = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            lossy_path.to_str().unwrap(),
            "-c:a",
            "pcm_s16le",
            output_wav.to_str().unwrap(),
        ])
        .output();

    let Ok(decode_out) = decode else {
        return false;
    };
    decode_out.status.success()
}

struct CodecSpec {
    name: &'static str,
    ffmpeg_codec: &'static str,
    bitrate: &'static str,
}

const CODECS: &[CodecSpec] = &[
    CodecSpec {
        name: "MP3 128k",
        ffmpeg_codec: "libmp3lame",
        bitrate: "128k",
    },
    CodecSpec {
        name: "MP3 192k",
        ffmpeg_codec: "libmp3lame",
        bitrate: "192k",
    },
    CodecSpec {
        name: "MP3 320k",
        ffmpeg_codec: "libmp3lame",
        bitrate: "320k",
    },
    CodecSpec {
        name: "AAC 128k",
        ffmpeg_codec: "aac",
        bitrate: "128k",
    },
    CodecSpec {
        name: "AAC 192k",
        ffmpeg_codec: "aac",
        bitrate: "192k",
    },
    CodecSpec {
        name: "Opus 64k",
        ffmpeg_codec: "libopus",
        bitrate: "64k",
    },
    CodecSpec {
        name: "Opus 128k",
        ffmpeg_codec: "libopus",
        bitrate: "128k",
    },
];

const STRENGTHS: &[f32] = &[0.01, 0.02, 0.03, 0.05, 0.08];

/// Strength values for the digital-only sweep (no codec).
const DIGITAL_STRENGTHS: &[f32] = &[
    0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25,
];

#[test]
#[ignore]
fn parameter_sweep() {
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    // Print table header
    print!("{:<12}", "Strength");
    for codec in CODECS {
        print!("{:<12}", codec.name);
    }
    println!();
    println!("{}", "-".repeat(12 + CODECS.len() * 12));

    for &strength in STRENGTHS {
        print!("{:<12.3}", strength);

        let config = WatermarkConfig {
            strength,
            ..WatermarkConfig::default()
        };

        let num_samples = 48000 * 25;
        let mut audio = make_test_audio(num_samples, config.sample_rate);
        agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let input_wav = dir.path().join("watermarked.wav");
        write_wav_i16(&input_wav, &audio, config.sample_rate);

        for codec in CODECS {
            let output_wav = dir.path().join(format!("decoded_{}.wav", codec.name));
            let ok = ffmpeg_round_trip(&input_wav, &output_wav, codec.ffmpeg_codec, codec.bitrate);

            if !ok {
                print!("{:<12}", "ERR");
                continue;
            }

            let decoded_samples = read_wav_f32(&output_wav);
            let result = agua_core::detect(&decoded_samples, &key, &config);

            match result {
                Ok(results) if results[0].payload == payload => {
                    print!("{:<12}", format!("{:.3}", results[0].confidence));
                }
                Ok(_) => {
                    print!("{:<12}", "WRONG");
                }
                Err(_) => {
                    print!("{:<12}", "FAIL");
                }
            }
        }
        println!();
    }
}

/// Pure digital strength sweep — no codec, no degradation.
///
/// Tests embed → detect at many strength levels to find the minimum viable
/// strength and verify that confidence scales as expected.
///
/// Run with: `cargo test digital_strength_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn digital_strength_sweep() {
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    let sample_rate = 48000u32;
    let num_samples = sample_rate as usize * 25; // 25 seconds

    println!();
    println!(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<8}",
        "Strength", "Detected", "Correct", "Confid.", "SyncCorr", "SNR(dB)"
    );
    println!("{}", "-".repeat(58));

    for &strength in DIGITAL_STRENGTHS {
        let config = WatermarkConfig {
            strength,
            ..WatermarkConfig::default()
        };

        // Generate fresh audio for each strength to avoid cumulative artifacts
        let original = make_test_audio(num_samples, sample_rate);
        let mut audio = original.clone();
        agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

        // Compute embedding SNR: how much did we change the signal?
        let mut signal_power = 0.0f64;
        let mut noise_power = 0.0f64;
        for (o, w) in original.iter().zip(audio.iter()) {
            signal_power += (*o as f64) * (*o as f64);
            noise_power += ((*w - *o) as f64) * ((*w - *o) as f64);
        }
        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        };

        // Detect
        let (result, diag) = agua_core::detect_with_diagnostics(&audio, &key, &config);
        let (detected, correct, confidence, sync_corr) = match result {
            Ok(ref results) if !results.is_empty() => {
                let r = &results[0];
                (
                    true,
                    r.payload == payload,
                    r.confidence,
                    diag.best_sync_corr,
                )
            }
            Ok(_) => (false, false, 0.0, diag.best_sync_corr),
            Err(_) => (false, false, 0.0, diag.best_sync_corr),
        };

        println!(
            "{:<10.4} {:<10} {:<10} {:<10.4} {:<10.4} {:<8.1}",
            strength,
            if detected { "YES" } else { "no" },
            if correct {
                "YES"
            } else if detected {
                "WRONG"
            } else {
                "-"
            },
            confidence,
            sync_corr,
            snr_db,
        );
    }
}

/// Same sweep but through the StreamDetector (streaming path, as WASM uses).
///
/// Feeds audio in small chunks (128 samples) to verify that streaming detection
/// matches batch detection.
///
/// Run with: `cargo test streaming_strength_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn streaming_strength_sweep() {
    use agua_core::StreamDetector;

    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    let sample_rate = 48000u32;
    let num_samples = sample_rate as usize * 25;

    println!();
    println!(
        "{:<10} {:<10} {:<10} {:<10} {:<10}",
        "Strength", "Detected", "Correct", "Confid.", "Chunks"
    );
    println!("{}", "-".repeat(50));

    for &strength in DIGITAL_STRENGTHS {
        let config = WatermarkConfig {
            strength,
            ..WatermarkConfig::default()
        };

        let mut audio = make_test_audio(num_samples, sample_rate);
        agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

        let mut detector = StreamDetector::new(&key, &config).unwrap();
        let mut results = Vec::new();
        let mut chunk_count = 0u32;

        for chunk in audio.chunks(128) {
            let r = detector.process(chunk);
            chunk_count += 1;
            results.extend(r);
        }
        if let Some(r) = detector.finalize() {
            results.push(r);
        }

        let (detected, correct, confidence) = if let Some(r) = results.first() {
            (true, r.payload == payload, r.confidence)
        } else {
            (false, false, 0.0)
        };

        println!(
            "{:<10.4} {:<10} {:<10} {:<10.4} {:<10}",
            strength,
            if detected { "YES" } else { "no" },
            if correct {
                "YES"
            } else if detected {
                "WRONG"
            } else {
                "-"
            },
            confidence,
            chunk_count,
        );
    }
}

// ── Acoustic simulation helpers (duplicated from acoustic_simulation.rs) ──

fn lowpass_filter(samples: &[f32], sample_rate: u32, cutoff_hz: f32, tap_count: usize) -> Vec<f32> {
    let tap_count = if tap_count % 2 == 0 {
        tap_count + 1
    } else {
        tap_count
    };
    let half = tap_count / 2;
    let fc = cutoff_hz / sample_rate as f32;
    let mut kernel = vec![0.0f32; tap_count];
    for (i, k) in kernel.iter_mut().enumerate() {
        let n = i as f32 - half as f32;
        let sinc = if n.abs() < 1e-10 {
            2.0 * std::f32::consts::PI * fc
        } else {
            (2.0 * std::f32::consts::PI * fc * n).sin() / n
        };
        let w = 0.42 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (tap_count - 1) as f32).cos()
            + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / (tap_count - 1) as f32).cos();
        *k = sinc * w;
    }
    let sum: f32 = kernel.iter().sum();
    if sum.abs() > 1e-10 {
        for k in kernel.iter_mut() {
            *k /= sum;
        }
    }
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

fn add_white_noise(samples: &mut [f32], snr_db: f32) {
    let signal_power: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    let noise_power = signal_power / 10.0f32.powf(snr_db / 10.0);
    let noise_std = noise_power.sqrt();
    let mut state: u32 = 0xDEAD_BEEF;
    for s in samples.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let u1 = (state as f32) / u32::MAX as f32;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let u2 = (state as f32) / u32::MAX as f32;
        let noise = noise_std
            * (-2.0 * u1.max(1e-10).ln()).sqrt()
            * (2.0 * std::f32::consts::PI * u2).cos();
        *s += noise;
    }
}

fn resample_jitter(samples: &[f32], ratio: f32) -> Vec<f32> {
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

/// Compare "audiowmark-style" (spacing=1, pairs=30) vs "agua-style" (spacing=8, pairs=60)
/// across multiple acoustic degradation scenarios.
///
/// Run with: `cargo test spacing_comparison -- --ignored --nocapture`
#[test]
#[ignore]
fn spacing_comparison() {
    use agua_core::PreProcessor;
    use rayon::prelude::*;

    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    let sample_rate = 48000u32;
    let num_samples = sample_rate as usize * 25;
    let strength = 0.10f32;

    struct Profile {
        name: &'static str,
        bin_spacing: usize,
        num_bin_pairs: usize,
    }

    let profiles = [
        Profile {
            name: "audiowmark-style (s=1, p=30)",
            bin_spacing: 1,
            num_bin_pairs: 30,
        },
        Profile {
            name: "agua default     (s=8, p=40)",
            bin_spacing: 8,
            num_bin_pairs: 40,
        },
        Profile {
            name: "hybrid           (s=4, p=40)",
            bin_spacing: 4,
            num_bin_pairs: 40,
        },
        Profile {
            name: "wide sparse      (s=8, p=30)",
            bin_spacing: 8,
            num_bin_pairs: 30,
        },
    ];

    struct Scenario {
        name: &'static str,
    }

    let scenarios = [
        Scenario { name: "Digital" },
        Scenario { name: "LP 8kHz" },
        Scenario { name: "Noise 20dB" },
        Scenario {
            name: "Reverb 150ms",
        },
        Scenario {
            name: "Spkr/Mic EQ",
        },
        Scenario { name: "AGC" },
        Scenario {
            name: "LP+N+jitter",
        },
        Scenario { name: "Realistic" },
    ];

    println!();
    println!("=== Spacing comparison (strength={:.2}) ===", strength);
    println!();
    print!("{:<32}", "Profile");
    for s in &scenarios {
        print!("{:<16}", s.name);
    }
    println!();
    println!("{}", "-".repeat(32 + scenarios.len() * 16));

    for profile in &profiles {
        print!("{:<32}", profile.name);

        let config = WatermarkConfig {
            strength,
            bin_spacing: profile.bin_spacing,
            num_bin_pairs: profile.num_bin_pairs,
            sample_rate,
            ..WatermarkConfig::default()
        };

        let original = make_test_audio(num_samples, sample_rate);

        let mut results: Vec<(usize, String)> = scenarios
            .par_iter()
            .enumerate()
            .map(|(i, _scenario)| {
                let mut audio = original.clone();
                agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

                let degraded = match i {
                    0 => audio,
                    1 => lowpass_filter(&audio, sample_rate, 8000.0, 127),
                    2 => {
                        let mut a = audio;
                        add_white_noise(&mut a, 20.0);
                        a
                    }
                    3 => convolve(&audio, &generate_rir(sample_rate, 150, 0xABCD)),
                    4 => {
                        let mut a = audio;
                        apply_speaker_mic_response(&mut a, sample_rate);
                        a
                    }
                    5 => {
                        let mut a = audio;
                        apply_agc(&mut a, sample_rate);
                        a
                    }
                    6 => {
                        let mut d = lowpass_filter(&audio, sample_rate, 8000.0, 127);
                        add_white_noise(&mut d, 20.0);
                        resample_jitter(&d, 1.001)
                    }
                    7 => {
                        // Realistic: reverb → EQ → AGC → noise → jitter → preprocess
                        let mut d = convolve(&audio, &generate_rir(sample_rate, 150, 0x5678));
                        apply_speaker_mic_response(&mut d, sample_rate);
                        apply_agc(&mut d, sample_rate);
                        add_white_noise(&mut d, 20.0);
                        let mut d = resample_jitter(&d, 1.001);
                        let mut preprocessor = PreProcessor::new(sample_rate);
                        preprocessor.process(&mut d);
                        d
                    }
                    _ => unreachable!(),
                };

                let cell = match agua_core::detect(&degraded, &key, &config) {
                    Ok(r) if !r.is_empty() && r[0].payload == payload => {
                        format!("{:.4}", r[0].confidence)
                    }
                    Ok(_) => "WRONG".to_string(),
                    Err(_) => "FAIL".to_string(),
                };
                (i, cell)
            })
            .collect();

        results.sort_by_key(|(i, _)| *i);
        for (_, cell) in results {
            print!("{:<16}", cell);
        }
        println!();
    }
}

// ── Room / speaker / mic / AGC simulation helpers ──

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn generate_rir(sample_rate: u32, rt60_ms: u32, seed: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let rt60_samples = (rt60_ms as f32 / 1000.0 * sr) as usize;
    let len = rt60_samples.max(1);
    let mut rir = vec![0.0f32; len];
    rir[0] = 1.0;
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
    let decay_rate = -6.91 / rt60_samples as f32;
    for (i, sample) in rir.iter_mut().enumerate().skip(1) {
        let envelope = (decay_rate * i as f32).exp();
        let noise = (xorshift32(&mut rng) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *sample += noise * envelope * 0.01;
    }
    let peak = rir.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        for s in rir.iter_mut() {
            *s /= peak;
        }
    }
    rir
}

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
        let (cos_w0, sin_w0) = (w0.cos(), w0.sin());
        let alpha = sin_w0 / (2.0_f32.sqrt());
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
        let (cos_w0, sin_w0) = (w0.cos(), w0.sin());
        let alpha = sin_w0 / (2.0_f32.sqrt());
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

    fn peaking_eq(sample_rate: f32, center_hz: f32, q: f32, gain_db: f32) -> Self {
        let a = 10.0f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * center_hz / sample_rate;
        let (cos_w0, sin_w0) = (w0.cos(), w0.sin());
        let alpha = sin_w0 / (2.0 * q);
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

    fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let y = self.b0 * *s + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2;
            self.x2 = self.x1;
            self.x1 = *s;
            self.y2 = self.y1;
            self.y1 = y;
            *s = y;
        }
    }
}

fn apply_speaker_mic_response(samples: &mut [f32], sample_rate: u32) {
    let sr = sample_rate as f32;
    SimBiquad::highpass(sr, 150.0).process(samples);
    SimBiquad::peaking_eq(sr, 2500.0, 2.0, 6.0).process(samples);
    SimBiquad::lowpass(sr, 12000.0).process(samples);
}

fn apply_agc(samples: &mut [f32], sample_rate: u32) {
    let sr = sample_rate as f32;
    let attack_coeff = 1.0 - (-1.0 / (0.01 * sr)).exp();
    let release_coeff = 1.0 - (-1.0 / (0.2 * sr)).exp();
    let target = 0.1f32;
    let mut envelope = target;
    for s in samples.iter_mut() {
        let abs_s = s.abs();
        let coeff = if abs_s > envelope {
            attack_coeff
        } else {
            release_coeff
        };
        envelope += coeff * (abs_s - envelope);
        *s *= (target / envelope.max(1e-6)).clamp(0.1, 20.0);
    }
}

/// Sweep bin_spacing × strength, both digital and through simulated acoustic channel.
///
/// Run with: `cargo test bin_spacing_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn bin_spacing_sweep() {
    use agua_core::PreProcessor;
    use rayon::prelude::*;

    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    let sample_rate = 48000u32;
    let num_samples = sample_rate as usize * 25;
    let spacings: &[usize] = &[1, 2, 3, 4, 6, 8, 12, 16];
    let strengths: &[f32] = &[0.02, 0.05, 0.08, 0.10];

    // ── Digital (clean) ──
    println!();
    println!("=== Digital (no degradation) ===");
    print!("{:<10}", "Spacing");
    for &s in strengths {
        print!("{:<12}", format!("s={:.2}", s));
    }
    println!();
    println!("{}", "-".repeat(10 + strengths.len() * 12));

    let strengths_len = strengths.len();
    let spacings_len = spacings.len();
    let total_cells = spacings_len * strengths_len;
    let results: Vec<(usize, String)> = (0..total_cells)
        .into_par_iter()
        .map(|idx| {
            let spacing = spacings[idx / strengths_len];
            let strength = strengths[idx % strengths_len];
            let config = WatermarkConfig {
                strength,
                bin_spacing: spacing,
                sample_rate,
                ..WatermarkConfig::default()
            };

            let mut audio = make_test_audio(num_samples, sample_rate);
            agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

            let cell = match agua_core::detect(&audio, &key, &config) {
                Ok(r) if !r.is_empty() && r[0].payload == payload => {
                    format!("{:.4}", r[0].confidence)
                }
                Ok(_) => "WRONG".to_string(),
                Err(_) => "FAIL".to_string(),
            };

            (idx, cell)
        })
        .collect();

    let mut cells = vec![String::new(); total_cells];
    for (idx, cell) in results {
        cells[idx] = cell;
    }

    for (row, &spacing) in spacings.iter().enumerate() {
        print!("{:<10}", spacing);
        let start = row * strengths_len;
        for col in 0..strengths_len {
            print!("{:<12}", cells[start + col]);
        }
        println!();
    }

    // ── Acoustic simulation (LP 8kHz + noise 20dB + jitter + preprocess) ──
    println!();
    println!("=== Acoustic simulation (LP 8kHz + noise 20dB + jitter 0.1%) ===");
    print!("{:<10}", "Spacing");
    for &s in strengths {
        print!("{:<12}", format!("s={:.2}", s));
    }
    println!();
    println!("{}", "-".repeat(10 + strengths.len() * 12));

    let results: Vec<(usize, String)> = (0..total_cells)
        .into_par_iter()
        .map(|idx| {
            let spacing = spacings[idx / strengths_len];
            let strength = strengths[idx % strengths_len];
            let config = WatermarkConfig {
                strength,
                bin_spacing: spacing,
                sample_rate,
                ..WatermarkConfig::default()
            };

            let mut audio = make_test_audio(num_samples, sample_rate);
            agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

            // Simulate acoustic channel
            let mut degraded = lowpass_filter(&audio, sample_rate, 8000.0, 127);
            add_white_noise(&mut degraded, 20.0);
            let mut degraded = resample_jitter(&degraded, 1.001);
            let mut preprocessor = PreProcessor::new(sample_rate);
            preprocessor.process(&mut degraded);

            let cell = match agua_core::detect(&degraded, &key, &config) {
                Ok(r) if !r.is_empty() && r[0].payload == payload => {
                    format!("{:.4}", r[0].confidence)
                }
                Ok(_) => "WRONG".to_string(),
                Err(_) => "FAIL".to_string(),
            };

            (idx, cell)
        })
        .collect();

    let mut cells = vec![String::new(); total_cells];
    for (idx, cell) in results {
        cells[idx] = cell;
    }

    for (row, &spacing) in spacings.iter().enumerate() {
        print!("{:<10}", spacing);
        let start = row * strengths_len;
        for col in 0..strengths_len {
            print!("{:<12}", cells[start + col]);
        }
        println!();
    }
}

/// Compare num_bin_pairs=30 vs 60 across strength, digital clean, lossy codecs,
/// and simulated acoustic channel.
///
/// Tests: (1) digital clean, (2) MP3 128k, (3) AAC 128k, (4) acoustic simulation
#[test]
#[ignore]
fn bin_pairs_30_vs_60() {
    use agua_core::PreProcessor;

    let sample_rate = 48000u32;
    let num_samples = 48000 * 25; // 25s
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC,
        0xBA, 0x98,
    ]);

    let bin_pairs_options: &[usize] = &[20, 30, 40, 50, 60];
    let strengths: &[f32] = &[0.03, 0.05, 0.08, 0.10];

    let dir = tempfile::tempdir().expect("tempdir");

    println!();
    println!("=== num_bin_pairs comparison ===");
    println!();

    // --- Digital clean ---
    println!("--- Digital clean (one-shot detect) ---");
    print!("{:<12}", "bins\\str");
    for &s in strengths {
        print!("{:<12}", format!("{:.2}", s));
    }
    println!();

    for &nbp in bin_pairs_options {
        print!("{:<12}", nbp);
        for &strength in strengths {
            let config = WatermarkConfig {
                strength,
                num_bin_pairs: nbp,
                sample_rate,
                ..WatermarkConfig::default()
            };

            let mut audio = make_test_audio(num_samples, sample_rate);
            agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

            let cell = match agua_core::detect(&audio, &key, &config) {
                Ok(r) if !r.is_empty() && r[0].payload == payload => {
                    format!("{:.4}", r[0].confidence)
                }
                Ok(_) => "WRONG".to_string(),
                Err(_) => "FAIL".to_string(),
            };
            print!("{:<12}", cell);
        }
        println!();
    }

    // --- MP3 128k ---
    println!();
    println!("--- MP3 128k ---");
    print!("{:<12}", "bins\\str");
    for &s in strengths {
        print!("{:<12}", format!("{:.2}", s));
    }
    println!();

    for &nbp in bin_pairs_options {
        print!("{:<12}", nbp);
        for &strength in strengths {
            let config = WatermarkConfig {
                strength,
                num_bin_pairs: nbp,
                sample_rate,
                ..WatermarkConfig::default()
            };

            let mut audio = make_test_audio(num_samples, sample_rate);
            agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

            let input_wav = dir.path().join(format!("bp{nbp}_s{strength}_mp3.wav"));
            let output_wav = dir.path().join(format!("bp{nbp}_s{strength}_mp3_out.wav"));
            write_wav_i16(&input_wav, &audio, sample_rate);

            let cell = if ffmpeg_round_trip(&input_wav, &output_wav, "libmp3lame", "128k") {
                let decoded = read_wav_f32(&output_wav);
                match agua_core::detect(&decoded, &key, &config) {
                    Ok(r) if !r.is_empty() && r[0].payload == payload => {
                        format!("{:.4}", r[0].confidence)
                    }
                    Ok(_) => "WRONG".to_string(),
                    Err(_) => "FAIL".to_string(),
                }
            } else {
                "NO_FF".to_string()
            };
            print!("{:<12}", cell);
        }
        println!();
    }

    // --- AAC 128k ---
    println!();
    println!("--- AAC 128k ---");
    print!("{:<12}", "bins\\str");
    for &s in strengths {
        print!("{:<12}", format!("{:.2}", s));
    }
    println!();

    for &nbp in bin_pairs_options {
        print!("{:<12}", nbp);
        for &strength in strengths {
            let config = WatermarkConfig {
                strength,
                num_bin_pairs: nbp,
                sample_rate,
                ..WatermarkConfig::default()
            };

            let mut audio = make_test_audio(num_samples, sample_rate);
            agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

            let input_wav = dir.path().join(format!("bp{nbp}_s{strength}_aac.wav"));
            let output_wav = dir.path().join(format!("bp{nbp}_s{strength}_aac_out.wav"));
            write_wav_i16(&input_wav, &audio, sample_rate);

            let cell = if ffmpeg_round_trip(&input_wav, &output_wav, "aac", "128k") {
                let decoded = read_wav_f32(&output_wav);
                match agua_core::detect(&decoded, &key, &config) {
                    Ok(r) if !r.is_empty() && r[0].payload == payload => {
                        format!("{:.4}", r[0].confidence)
                    }
                    Ok(_) => "WRONG".to_string(),
                    Err(_) => "FAIL".to_string(),
                }
            } else {
                "NO_FF".to_string()
            };
            print!("{:<12}", cell);
        }
        println!();
    }

    // --- Acoustic simulation ---
    println!();
    println!("--- Acoustic simulation (lowpass + noise + jitter) ---");
    print!("{:<12}", "bins\\str");
    for &s in strengths {
        print!("{:<12}", format!("{:.2}", s));
    }
    println!();

    for &nbp in bin_pairs_options {
        print!("{:<12}", nbp);
        for &strength in strengths {
            let config = WatermarkConfig {
                strength,
                num_bin_pairs: nbp,
                sample_rate,
                ..WatermarkConfig::default()
            };

            let mut audio = make_test_audio(num_samples, sample_rate);
            agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

            // Simulate acoustic channel
            let mut degraded = lowpass_filter(&audio, sample_rate, 8000.0, 127);
            add_white_noise(&mut degraded, 20.0);
            let mut degraded = resample_jitter(&degraded, 1.001);
            let mut preprocessor = PreProcessor::new(sample_rate);
            preprocessor.process(&mut degraded);

            let cell = match agua_core::detect(&degraded, &key, &config) {
                Ok(r) if !r.is_empty() && r[0].payload == payload => {
                    format!("{:.4}", r[0].confidence)
                }
                Ok(_) => "WRONG".to_string(),
                Err(_) => "FAIL".to_string(),
            };
            print!("{:<12}", cell);
        }
        println!();
    }
}
