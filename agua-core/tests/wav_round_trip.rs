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

/// Write samples to a WAV file as 32-bit float.
fn write_wav_f32(path: &std::path::Path, samples: &[f32], sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("failed to create WAV writer");
    for &s in samples {
        writer.write_sample(s).expect("failed to write sample");
    }
    writer.finalize().expect("failed to finalize WAV");
}

/// Write samples to a WAV file as 16-bit integer.
fn write_wav_i16(path: &std::path::Path, samples: &[f32], sample_rate: u32) {
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
fn read_wav_f32(path: &std::path::Path) -> (Vec<f32>, u32) {
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

#[test]
fn wav_f32_embed_detect_round_trip() {
    let config = WatermarkConfig {
        strength: 0.05,
        ..WatermarkConfig::default()
    };
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA,
        0x98,
    ]);

    let num_samples = 48000 * 22;
    let mut audio = make_test_audio(num_samples, config.sample_rate);

    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let wav_path = dir.path().join("watermarked_f32.wav");

    write_wav_f32(&wav_path, &audio, config.sample_rate);
    let (read_back, sr) = read_wav_f32(&wav_path);
    assert_eq!(sr, config.sample_rate);

    let results = agua_core::detect(&read_back, &key, &config).unwrap();
    assert!(
        !results.is_empty(),
        "no watermark detected after WAV f32 round-trip"
    );
    assert_eq!(results[0].payload, payload);
}

#[test]
fn wav_i16_embed_detect_round_trip() {
    let config = WatermarkConfig {
        strength: 0.05,
        ..WatermarkConfig::default()
    };
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([
        0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33,
        0x44,
    ]);

    let num_samples = 48000 * 22;
    let mut audio = make_test_audio(num_samples, config.sample_rate);

    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let wav_path = dir.path().join("watermarked_i16.wav");

    // Write as 16-bit — this quantizes the signal, testing robustness
    write_wav_i16(&wav_path, &audio, config.sample_rate);
    let (read_back, sr) = read_wav_f32(&wav_path);
    assert_eq!(sr, config.sample_rate);

    let results = agua_core::detect(&read_back, &key, &config).unwrap();
    assert!(
        !results.is_empty(),
        "no watermark detected after WAV i16 round-trip"
    );
    assert_eq!(results[0].payload, payload);
}

#[test]
fn wav_44100_sample_rate() {
    let config = WatermarkConfig {
        sample_rate: 44100,
        strength: 0.05,
        ..WatermarkConfig::default()
    };
    let key = WatermarkKey::from_passphrase("test-key-44100");
    let payload = Payload::new([0xAB; 16]);

    // 1024 frames * 1024 samples/frame = 1,048,576 minimum; at 44100 Hz that's ~24s
    let num_samples = 44100 * 25;
    let mut audio = make_test_audio(num_samples, config.sample_rate);

    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let wav_path = dir.path().join("watermarked_44100.wav");

    write_wav_f32(&wav_path, &audio, config.sample_rate);
    let (read_back, sr) = read_wav_f32(&wav_path);
    assert_eq!(sr, 44100);

    let results = agua_core::detect(&read_back, &key, &config).unwrap();
    assert!(!results.is_empty(), "no watermark detected at 44100 Hz");
    assert_eq!(results[0].payload, payload);
}

#[test]
fn wav_streaming_embed_file_detect() {
    use agua_core::StreamEmbedder;

    let config = WatermarkConfig {
        strength: 0.05,
        ..WatermarkConfig::default()
    };
    let key = WatermarkKey::new(&[7u8; 16]).unwrap();
    let payload = Payload::new([0x55; 16]);

    let audio = make_test_audio(48000 * 22, config.sample_rate);

    // Embed using streaming API in 4096-sample chunks
    let mut embedder = StreamEmbedder::new(&payload, &key, &config).unwrap();
    let mut watermarked = Vec::new();
    for chunk in audio.chunks(4096) {
        watermarked.extend(embedder.process(chunk));
    }
    watermarked.extend(embedder.flush());

    // Write to WAV, read back, detect with one-shot API
    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let wav_path = dir.path().join("stream_embedded.wav");

    write_wav_f32(&wav_path, &watermarked, config.sample_rate);
    let (read_back, _) = read_wav_f32(&wav_path);

    let results = agua_core::detect(&read_back, &key, &config).unwrap();
    assert!(
        !results.is_empty(),
        "no watermark detected after streaming embed + WAV round-trip"
    );
    assert_eq!(results[0].payload, payload);
}
