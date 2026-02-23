//! Lossy codec round-trip robustness tests.
//!
//! These tests require `ffmpeg` on the system PATH and are marked `#[ignore]`.
//! Run with: `cargo test -- --ignored --nocapture`

use std::path::Path;
use std::process::Command;

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

/// Write samples to a WAV file as 16-bit integer (common format for codec input).
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

/// Read a WAV file back as f32 samples.
fn read_wav_f32(path: &Path) -> (Vec<f32>, u32) {
    let reader = hound::WavReader::open(path).expect("failed to open WAV");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
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
    };
    (samples, spec.sample_rate)
}

/// Encode a WAV file to a lossy format and decode back to WAV using ffmpeg.
fn ffmpeg_round_trip(input_wav: &Path, output_wav: &Path, codec: &str, bitrate: &str) {
    let ext = match codec {
        "libmp3lame" => "mp3",
        "aac" => "m4a",
        "libopus" => "ogg",
        _ => panic!("unsupported codec: {codec}"),
    };

    let lossy_path = input_wav.with_extension(ext);

    // Encode WAV → lossy
    let encode_status = Command::new("ffmpeg")
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
        .output()
        .expect("failed to run ffmpeg encode");

    assert!(
        encode_status.status.success(),
        "ffmpeg encode failed: {}",
        String::from_utf8_lossy(&encode_status.stderr)
    );

    // Decode lossy → WAV
    let decode_status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            lossy_path.to_str().unwrap(),
            "-c:a",
            "pcm_s16le",
            output_wav.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run ffmpeg decode");

    assert!(
        decode_status.status.success(),
        "ffmpeg decode failed: {}",
        String::from_utf8_lossy(&decode_status.stderr)
    );
}

/// Run a full embed → encode → decode → detect test for a given codec/bitrate.
fn lossy_round_trip_test(codec: &str, bitrate: &str, codec_name: &str) {
    let config = WatermarkConfig::robust();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    let num_samples = 48000 * 25;
    let mut audio = make_test_audio(num_samples, config.sample_rate);

    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let input_wav = dir.path().join("watermarked.wav");
    let output_wav = dir.path().join("decoded.wav");

    write_wav_i16(&input_wav, &audio, config.sample_rate);
    ffmpeg_round_trip(&input_wav, &output_wav, codec, bitrate);

    let (decoded_samples, sr) = read_wav_f32(&output_wav);
    assert_eq!(sr, config.sample_rate);

    let result = agua_core::detect(&decoded_samples, &key, &config);
    match result {
        Ok(results) => {
            assert_eq!(
                results[0].payload, payload,
                "{codec_name} @ {bitrate}: detected but payload mismatch"
            );
            println!(
                "{codec_name} @ {bitrate}: PASS (confidence: {:.4})",
                results[0].confidence
            );
        }
        Err(_) => {
            println!("{codec_name} @ {bitrate}: FAIL (not detected)");
            panic!("{codec_name} @ {bitrate}: watermark not detected after lossy round-trip");
        }
    }
}

// --- MP3 tests ---

#[test]
#[ignore]
fn mp3_128k_round_trip() {
    lossy_round_trip_test("libmp3lame", "128k", "MP3");
}

#[test]
#[ignore]
fn mp3_192k_round_trip() {
    lossy_round_trip_test("libmp3lame", "192k", "MP3");
}

#[test]
#[ignore]
fn mp3_320k_round_trip() {
    lossy_round_trip_test("libmp3lame", "320k", "MP3");
}

// --- AAC tests ---

#[test]
#[ignore]
fn aac_128k_round_trip() {
    lossy_round_trip_test("aac", "128k", "AAC");
}

#[test]
#[ignore]
fn aac_192k_round_trip() {
    lossy_round_trip_test("aac", "192k", "AAC");
}

#[test]
#[ignore]
fn aac_320k_round_trip() {
    lossy_round_trip_test("aac", "320k", "AAC");
}

// --- Opus tests ---

#[test]
#[ignore]
fn opus_64k_round_trip() {
    lossy_round_trip_test("libopus", "64k", "Opus");
}

#[test]
#[ignore]
fn opus_128k_round_trip() {
    lossy_round_trip_test("libopus", "128k", "Opus");
}
