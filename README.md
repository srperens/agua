# Agua

High-performance audio watermarking library in Rust. Embeds and detects invisible 128-bit payloads in audio using the patchwork algorithm in the FFT domain.

Survives lossy compression (MP3, AAC, Opus), 16-bit quantization, and sample rate conversion. Runs at 4500x real-time on modern hardware.

## Features

- **128-bit payload** with CRC-32 integrity check
- **Convolutional coding** (rate 1/6) with soft-decision Viterbi decoding for error correction
- **Streaming API** for real-time processing with arbitrary chunk sizes
- **GStreamer plugin** (`aguaembed` audio filter element)
- **CLI tool** for embedding and detecting watermarks in WAV files
- **Optional parallelism** via rayon (`parallel` feature flag)
- **Patent-safe** algorithm (Bender et al. 1996, all related patents expired)

## Project Structure

```
agua/
├── agua-core/    Core library crate
├── agua-cli/     CLI binary
└── agua-gst/     GStreamer plugin
```

## Quick Start

### Build

```bash
cargo build --release
```

### CLI

```bash
# Embed a watermark
agua embed -i input.wav -o watermarked.wav \
  -p deadbeef0123456789abcdef01234567 \
  -k my-secret-key

# Detect a watermark
agua detect -i watermarked.wav -k my-secret-key
```

The payload is a 32-character hex string (128 bits). The key is a passphrase used to derive the embedding pattern.

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-p, --payload` | | 32-char hex payload (embed only) |
| `-k, --key` | `agua-default-key` | Key passphrase |
| `-s, --strength` | `0.01` | Embedding strength (0.001-0.1) |
| `--profile` | | Preset profile (`music`) |
| `--offset-seconds` | `0` | Delay before embedding starts |
| `--frame-size` | `1024` | FFT frame size (power of 2) |
| `--num-bin-pairs` | `200` | Frequency bin pairs per frame |
| `--min-bin` | `5` | Minimum FFT bin index |
| `--max-bin` | `500` | Maximum FFT bin index |

The `music` profile uses strength 0.05, 50 bin pairs, and max bin 300 for better robustness with lossy codecs.

## Rust API

```rust
use agua_core::{embed, detect, WatermarkConfig, WatermarkKey, Payload};

// Configure
let config = WatermarkConfig::default();  // 48kHz, strength 0.01
let key = WatermarkKey::from_passphrase("my-secret-key");
let payload = Payload::from_hex("deadbeef0123456789abcdef01234567")?;

// Embed (in-place)
let mut samples: Vec<f32> = load_audio();
embed(&mut samples, &payload, &key, &config)?;

// Detect
let results = detect(&samples, &key, &config)?;
for r in &results {
    println!("Payload: {} (confidence: {:.2})", r.payload.to_hex(), r.confidence);
}
```

### Streaming

For real-time or chunked processing:

```rust
use agua_core::{StreamEmbedder, StreamDetector};

// Embedding
let mut embedder = StreamEmbedder::new(&payload, &key, &config)?;
for chunk in input_chunks {
    let output = embedder.process(&chunk);
    write_audio(&output);
}
let final_output = embedder.flush();

// Detection
let mut detector = StreamDetector::new(&key, &config)?;
for chunk in input_chunks {
    let detections = detector.process(&chunk);
    handle_detections(&detections);
}
```

### Configuration

```rust
// Default: strength 0.01, suitable for lossless or high-bitrate audio
let config = WatermarkConfig::default();

// Robust: strength 0.05, tuned for MP3/AAC/Opus survival
let config = WatermarkConfig::robust();
```

### Parallel Processing

Enable the `parallel` feature for rayon-based multi-threaded embedding and detection:

```toml
agua-core = { version = "0.1", features = ["parallel"] }
```

```rust
agua_core::embed_parallel(&mut samples, &payload, &key, &config)?;
let results = agua_core::detect_parallel(&samples, &key, &config)?;
```

## GStreamer Plugin

Build and load the plugin:

```bash
cargo build -p agua-gst --release
export GST_PLUGIN_PATH=/path/to/agua/target/release
```

Verify:

```bash
gst-inspect-1.0 aguaembed
```

### Pipeline Example

```bash
gst-launch-1.0 \
  filesrc location=input.wav ! wavparse ! \
  audioconvert ! audio/x-raw,format=F32LE ! \
  aguaembed payload=deadbeef0123456789abcdef01234567 key=my-key strength=0.01 ! \
  audioconvert ! wavenc ! filesink location=output.wav
```

### Element Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `payload` | String | | 32-char hex payload |
| `key` | String | `agua-default-key` | Key passphrase |
| `strength` | Float | 0.01 | Embedding strength |
| `frame-size` | UInt | 1024 | FFT frame size (power of 2) |
| `num-bin-pairs` | UInt | 200 | Bin pairs per frame |
| `min-bin` | UInt | 5 | Minimum FFT bin index |
| `max-bin` | UInt | 500 | Maximum FFT bin index |
| `offset-seconds` | Float | 0.0 | Delay before embedding starts |

Audio format: F32LE, interleaved, any sample rate and channel count.

## Performance

Benchmarks on release builds (criterion):

| Operation | Duration | Real-time Factor |
|-----------|----------|------------------|
| Embed 1s @ 48kHz | ~223 us | 4500x |
| Detect 22s @ 48kHz | ~4.9 ms | 4500x |
| Stream embed 1s (4096-sample chunks) | ~266 us | 3760x |
| Stream detect 22s (4096-sample chunks) | ~4.7 ms | 4680x |

Run benchmarks:

```bash
cargo bench --release
```

## How It Works

Agua uses the **patchwork algorithm** operating in the frequency domain:

1. **Payload encoding**: 128-bit payload + CRC-32 (160 bits) is convolutionally encoded at rate 1/6 (960 coded bits), then interleaved with a 64-bit sync pattern
2. **Embedding**: Audio is split into non-overlapping frames. Each frame is FFT-transformed, and pseudo-random frequency bin pairs (derived from the key via AES-128 PRNG) are scaled up or down according to the watermark bit
3. **Detection**: Frames are FFT-transformed and the patchwork statistic (magnitude difference between bin pairs) yields soft bit values. A Viterbi decoder recovers the payload, verified by CRC-32

Minimum audio duration for reliable detection: ~22 seconds at 48kHz.

## Development

```bash
cargo fmt              # Format code
cargo clippy --all     # Lint (no warnings allowed)
cargo test --all       # Run tests (47 passing)
cargo bench --release  # Run benchmarks
```

Lossy codec round-trip tests (requires ffmpeg):

```bash
cargo test -p agua-core --test lossy_codec_round_trip -- --ignored
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

Copyright 2026 [Eyevinn Technology AB](https://www.eyevinn.se).
