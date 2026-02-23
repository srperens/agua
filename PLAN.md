# Agua: Audio Watermarking Library in Rust

## Context

New Rust library for audio watermarking at Eyevinn. Needs a base crate for embedding/detecting watermarks, with GStreamer plugin support later. Must be fast (real-time capable), robust (survive MP3/AAC), and patent-safe for open source use.

## Research Summary

### Algorithm: Patchwork in FFT Domain
audiowmark uses the **patchwork algorithm** (Bender et al., 1996) in the frequency domain — NOT classical spread-spectrum. It splits audio into 1024-sample frames, applies FFT, uses an AES-128 key to pseudo-randomly select frequency bin pairs, and modifies their magnitudes to encode bits. Convolutional coding + interleaved sync frames provide error correction and blind detection. 128-bit payload capacity.

### Patent Status: LOW RISK
- Patchwork algorithm published academically (1996, IBM Systems Journal) — **not patented**
- Cox/NEC spread-spectrum patents: **expired** (2015-2017)
- Microsoft audio watermarking patents: **expired** (2021)
- Digimarc time-frequency domain patent: **expired** (2020)
- audiowmark has been open source since 2018 without patent challenges
- Caveat: Digimarc/Verance have large portfolios with some newer patents — avoid their specific techniques

### Port vs. Clean-Room Implementation: CLEAN-ROOM
- audiowmark is **GPLv3** — cannot port code if we want MIT/Apache-2.0 license
- We implement the same general approach based on **published academic papers** (Bender 1996, Steinebach 2004)
- The algorithm is well-documented; no need to reverse-engineer anything

## Architecture

### Workspace Structure
```
agua/
  Cargo.toml              # workspace: [agua-core, agua-cli]
  LICENSE-MIT
  LICENSE-APACHE
  agua-core/               # library crate
    Cargo.toml
    src/
      lib.rs               # Public API re-exports
      error.rs             # Error types (thiserror)
      key.rs               # WatermarkKey, AES-PRNG for bin selection
      fft.rs               # realfft wrapper, pre-allocated buffers
      frame.rs             # Windowing, overlap-add reconstruction
      patchwork.rs         # Core: per-frame embed/detect in frequency domain
      sync.rs              # Sync frame generation + correlation detection
      codec.rs             # Convolutional encoder + soft-decision Viterbi decoder
      payload.rs           # 128-bit payload, CRC-32
      config.rs            # WatermarkConfig (strength, sample_rate)
      embed.rs             # High-level embedding pipeline
      detect.rs            # High-level detection pipeline
  agua-cli/                # binary crate
    Cargo.toml
    src/main.rs
```

### Public API (agua-core)
```rust
// One-shot (for files)
pub fn embed(samples: &mut [f32], payload: &Payload, key: &WatermarkKey, config: &WatermarkConfig) -> Result<(), Error>;
pub fn detect(samples: &[f32], key: &WatermarkKey, config: &WatermarkConfig) -> Result<Vec<DetectionResult>, Error>;

// Streaming (for GStreamer / real-time)
pub struct StreamEmbedder { ... }  // process(&mut self, input: &[f32]) -> Vec<f32>
pub struct StreamDetector { ... }  // process(&mut self, input: &[f32]) -> Vec<DetectionResult>
```

All APIs operate on mono f32 sample buffers. No file I/O in the core library.

### Algorithm Pipeline

**Embedding:**
```
Payload (128 bits) -> CRC-32 append (160 bits) -> Conv. encode rate 1/6 (960 bits)
-> Interleave with sync frames -> Per-frame patchwork embed (FFT -> modify bins -> IFFT)
-> Overlap-add -> Watermarked audio
```

**Detection:**
```
Audio -> Frame FFT -> Sync correlation search -> Soft-bit extraction per frame
-> Viterbi decode (960 soft values -> 160 bits) -> CRC check -> Payload + confidence
```

### Dependencies
| Crate | Purpose | Why |
|-------|---------|-----|
| `realfft` 3.5 | FFT/IFFT | 2x faster for real signals, AVX auto-detection |
| `aes` 0.8 + `cipher` 0.4 | AES-128 PRNG | AES-NI accelerated, deterministic bin selection |
| `thiserror` 2 | Error types | Standard |
| `hound` 3.5 (dev) | WAV I/O for tests | |
| `clap` 4 (cli only) | CLI args | |
| `criterion` 0.5 (dev) | Benchmarks | |

**No external FEC crate** — Viterbi decoder implemented from scratch (~300-500 lines) because existing Rust crates lack soft-decision support.

### Performance
- FFT: rustfft has AVX auto-detection, realfft halves the work for real signals
- AES: AES-NI accelerated via RustCrypto
- Zero allocations per frame in steady state (pre-allocated buffers)
- Streaming latency: 1 frame = 1024 samples = ~21ms at 48kHz
- Optional `rayon` feature flag for parallel detection

## Implementation Order

### Phase 1: Foundations
1. Set up workspace structure (Cargo.toml files, directories)
2. `error.rs` — Error enum
3. `key.rs` — WatermarkKey + AES-PRNG with tests
4. `fft.rs` — FFT wrapper with round-trip tests

### Phase 2: Core Algorithm
5. `config.rs` + `payload.rs` — Config and payload types
6. `patchwork.rs` — Single-frame embed/detect with tests
7. `codec.rs` — Convolutional encoder + Viterbi decoder with tests

### Phase 3: Integration
8. `sync.rs` — Sync generation + correlation detection
9. `frame.rs` — Windowing + overlap-add
10. `embed.rs` — Full embedding pipeline
11. `detect.rs` — Full detection pipeline + round-trip integration tests

### Phase 4: Streaming + CLI
12. `StreamEmbedder` / `StreamDetector` — Streaming wrappers
13. `agua-cli` — CLI binary (embed/detect commands)
14. Integration tests with WAV files, benchmarks

### Phase 5: Robustness (later)
15. MP3/AAC round-trip robustness tests
16. Parameter tuning
17. Optional rayon parallelism

## Verification
- Unit tests per module (determinism, round-trip, correctness)
- Integration test: generate sine wave -> embed -> detect -> verify payload
- WAV round-trip test with `hound`
- Benchmark: embed/detect throughput with criterion
- Later: MP3 robustness tests (require ffmpeg/lame)

## Key Parameters
| Parameter | Value |
|-----------|-------|
| Frame size | 1024 samples |
| FFT bins | 513 (real FFT of 1024) |
| Bin pairs per frame | ~30 |
| Useful bin range | 5-500 |
| Code rate | 1/6 |
| Constraint length | 7 (64 states) |
| Payload | 128 bits |
| CRC | 32 bits |
| Sync pattern | 64 bits |

## License
Dual MIT/Apache-2.0 (Rust ecosystem convention, fully permissive).
