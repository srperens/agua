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
