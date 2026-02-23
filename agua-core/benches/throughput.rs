use criterion::{Criterion, black_box, criterion_group, criterion_main};

use agua_core::{Payload, StreamDetector, StreamEmbedder, WatermarkConfig, WatermarkKey};

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

fn bench_embed(c: &mut Criterion) {
    let config = WatermarkConfig::default();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    // 1 second of audio at 48kHz
    let audio = make_test_audio(48000, config.sample_rate);

    c.bench_function("embed_1s_48khz", |b| {
        b.iter(|| {
            let mut samples = audio.clone();
            agua_core::embed(black_box(&mut samples), &payload, &key, &config).unwrap();
        });
    });
}

fn bench_detect(c: &mut Criterion) {
    let config = WatermarkConfig {
        strength: 0.03,
        ..WatermarkConfig::default()
    };
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    // Need enough audio for at least one block (1024 frames * 1024 samples = ~22s)
    let num_samples = 48000 * 22;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    c.bench_function("detect_22s_48khz", |b| {
        b.iter(|| {
            agua_core::detect(black_box(&audio), &key, &config).unwrap();
        });
    });
}

fn bench_stream_embed(c: &mut Criterion) {
    let config = WatermarkConfig::default();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    // 1 second of audio at 48kHz
    let audio = make_test_audio(48000, config.sample_rate);

    c.bench_function("stream_embed_1s_48khz_4096_chunks", |b| {
        b.iter(|| {
            let mut embedder = StreamEmbedder::new(&payload, &key, &config).unwrap();
            let mut output = Vec::new();
            for chunk in audio.chunks(4096) {
                output.extend(embedder.process(black_box(chunk)));
            }
            output.extend(embedder.flush());
            black_box(output);
        });
    });
}

fn bench_stream_detect(c: &mut Criterion) {
    let config = WatermarkConfig {
        strength: 0.03,
        ..WatermarkConfig::default()
    };
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    let num_samples = 48000 * 22;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    c.bench_function("stream_detect_22s_48khz_4096_chunks", |b| {
        b.iter(|| {
            let mut detector = StreamDetector::new(&key, &config).unwrap();
            let mut results = Vec::new();
            for chunk in audio.chunks(4096) {
                results.extend(detector.process(black_box(chunk)));
            }
            if let Some(r) = detector.finalize() {
                results.push(r);
            }
            black_box(results);
        });
    });
}

#[cfg(feature = "parallel")]
fn bench_parallel_embed(c: &mut Criterion) {
    let config = WatermarkConfig::default();
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    let audio = make_test_audio(48000, config.sample_rate);

    c.bench_function("parallel_embed_1s_48khz", |b| {
        b.iter(|| {
            let mut samples = audio.clone();
            agua_core::embed_parallel(black_box(&mut samples), &payload, &key, &config).unwrap();
        });
    });
}

#[cfg(feature = "parallel")]
fn bench_parallel_detect(c: &mut Criterion) {
    let config = WatermarkConfig {
        strength: 0.03,
        ..WatermarkConfig::default()
    };
    let key = WatermarkKey::new(&[42u8; 16]).unwrap();
    let payload = Payload::new([0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    let num_samples = 48000 * 22;
    let mut audio = make_test_audio(num_samples, config.sample_rate);
    agua_core::embed(&mut audio, &payload, &key, &config).unwrap();

    c.bench_function("parallel_detect_22s_48khz", |b| {
        b.iter(|| {
            agua_core::detect_parallel(black_box(&audio), &key, &config).unwrap();
        });
    });
}

fn bench_fft_frame(c: &mut Criterion) {
    let config = WatermarkConfig::default();
    let audio = make_test_audio(config.frame_size, config.sample_rate);

    c.bench_function("fft_forward_inverse_1024", |b| {
        b.iter(|| {
            let mut fft = agua_core::fft::FftProcessor::new(config.frame_size).unwrap();
            let mut buf = audio.clone();
            fft.forward(black_box(&mut buf)).unwrap();
            fft.inverse(black_box(&mut buf)).unwrap();
            fft.normalize(&mut buf);
            black_box(buf);
        });
    });
}

#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    bench_embed,
    bench_detect,
    bench_stream_embed,
    bench_stream_detect,
    bench_fft_frame,
);

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_embed,
    bench_detect,
    bench_stream_embed,
    bench_stream_detect,
    bench_fft_frame,
    bench_parallel_embed,
    bench_parallel_detect,
);

criterion_main!(benches);
