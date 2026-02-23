# Progress Log

## 2026-02-23 — Initial Evaluation

### Phase 1: Foundations — COMPLETE
- [x] Workspace structure (Cargo.toml, agua-core, agua-cli)
- [x] `error.rs` — Error enum with thiserror
- [x] `key.rs` — WatermarkKey + AES-PRNG (6 tests)
- [x] `fft.rs` — FFT wrapper with round-trip tests (3 tests)

### Phase 2: Core Algorithm — COMPLETE
- [x] `config.rs` — WatermarkConfig with defaults
- [x] `payload.rs` — 128-bit payload + CRC-32 (5 tests)
- [x] `patchwork.rs` — Single-frame embed/detect (5 tests)
- [x] `codec.rs` — Convolutional encoder + Viterbi decoder (7 tests)

### Phase 3: Integration — COMPLETE
- [x] `sync.rs` — Sync generation + correlation detection (5 tests)
- [x] `frame.rs` — Windowing + overlap-add (3 tests)
- [x] `embed.rs` — Full embedding pipeline (3 tests)
- [x] `detect.rs` — Full detection pipeline + round-trip test (3 tests)

### Phase 4: Streaming + CLI — COMPLETE
- [x] `stream.rs` — StreamEmbedder / StreamDetector (1 test)
- [x] `agua-cli` — CLI binary (embed/detect commands)
- [x] WAV integration tests (4 tests: f32 round-trip, i16 quantization, 44100 Hz, streaming embed)
- [x] Criterion benchmarks (embed, detect, streaming, FFT)

### Phase 5: Robustness, Parallelism & Licensing — COMPLETE
- [x] LICENSE-MIT and LICENSE-APACHE files added
- [x] `WatermarkConfig::robust()` constructor (strength 0.05 for lossy codec survival)
- [x] MP3/AAC/Opus round-trip robustness tests (8 tests, `#[ignore]`, require ffmpeg)
- [x] Parameter tuning sweep test (`#[ignore]`, strength x codec matrix)
- [x] Optional rayon parallelism (`parallel` feature: `embed_parallel`, `detect_parallel`)
- [x] Parallel benchmarks (embed + detect)

### Phase 6: GStreamer Plugin — COMPLETE
- [x] `agua-gst` crate added to workspace
- [x] `aguawatermarkembed` GStreamer element (BaseTransform, in-place processing)
- [x] Properties: key, payload, strength, frame-size, num-bin-pairs, min-bin, max-bin, offset-frames, profile
- [x] Multi-channel support (per-channel embedding with independent frame tracking)
- [x] No tests yet for agua-gst

### Outstanding Issues
- [ ] 2 clippy warnings in `agua-cli` (needless_update in main.rs:188, main.rs:317)
- [ ] 3 clippy warnings in `agua-gst` (needless_range_loop embed.rs:347, collapsible_if embed.rs:358, needless_range_loop embed.rs:415)
- [ ] No tests for `agua-gst`
- [ ] Lossy codec robustness tests not yet validated (require ffmpeg)

### Test Summary
- **44/44 tests passing** (40 unit + 4 WAV integration), default features
- **47/47 tests passing** with `--features parallel` (adds 3 parallel unit tests)
- 9 additional `#[ignore]` tests (lossy codec round-trips + parameter tuning, require ffmpeg)
- `cargo fmt --all --check` — clean
- `cargo clippy` — 0 warnings in `agua-core`, 2 in `agua-cli`, 3 in `agua-gst`

### Benchmark Results (release build)
| Benchmark | Time | Real-time ratio |
|-----------|------|-----------------|
| `embed_1s_48khz` | 223 µs | **~4500x real-time** |
| `detect_22s_48khz` | 4.9 ms | **~4500x real-time** |
| `stream_embed_1s (4096 chunks)` | 266 µs | **~3760x real-time** |
| `stream_detect_22s (4096 chunks)` | 4.7 ms | **~4680x real-time** |
| `fft_forward_inverse_1024` | 18.6 µs | — |

Conclusion: **easily real-time capable** — thousands of times faster than real-time for both embedding and detection.

### Git History
| Commit | Description |
|--------|-------------|
| `f9366e8` | feat: implement Agua audio watermarking library |
| `5e6f2dc` | test: add WAV integration tests and criterion benchmarks |
| `a8b011b` | some additions |
| `8e4f0ae` | fix: correct GStreamer plugin lib name for symbol resolution |
| `89db9a4` | feat: add robustness tests, rayon parallelism, and license files (Phase 5) |
| `d1e0948` | feat: add GStreamer plugin for agua audio watermarking |
| `dc9cf6b` | chore: add agua-gst to workspace and fmt agua-cli |

### Workspace Crates
| Crate | Type | Description |
|-------|------|-------------|
| `agua-core` | lib | Core watermarking library (14 modules, 47 tests w/ parallel) |
| `agua-cli` | bin | CLI for embed/detect on WAV files |
| `agua-gst` | lib (cdylib) | GStreamer plugin element `aguawatermarkembed` |

### Files Implemented
| Module | Tests | Status |
|--------|-------|--------|
| **agua-core** | | |
| lib.rs | — | Public API re-exports |
| error.rs | — | Error types |
| key.rs | 6 | AES-PRNG bin selection |
| fft.rs | 3 | realfft wrapper |
| config.rs | — | Configuration + `robust()` |
| payload.rs | 5 | 128-bit payload + CRC |
| patchwork.rs | 5 | Core embed/detect per frame |
| codec.rs | 7 | Conv. encoder + Viterbi |
| sync.rs | 5 | Sync pattern + correlation |
| frame.rs | 3 | Windowing + overlap-add |
| embed.rs | 3 | Embedding pipeline |
| detect.rs | 3 | Detection pipeline |
| stream.rs | 1 | Streaming wrappers |
| parallel.rs | 3 | Rayon parallel embed/detect (feature-gated) |
| tests/wav_round_trip.rs | 4 | WAV file integration tests |
| tests/lossy_codec_round_trip.rs | 8 | MP3/AAC/Opus robustness (ignored) |
| tests/parameter_tuning.rs | 1 | Strength x codec sweep (ignored) |
| benches/throughput.rs | 5+2 | Criterion benchmarks (+ parallel) |
| **agua-cli** | | |
| main.rs | — | CLI binary (embed/detect commands) |
| **agua-gst** | | |
| lib.rs | — | Plugin registration |
| embed.rs | — | `AguaWatermarkEmbed` BaseTransform element |
