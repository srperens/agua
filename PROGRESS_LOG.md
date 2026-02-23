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

### Phase 5: Robustness — NOT STARTED
- [ ] MP3/AAC round-trip robustness tests
- [ ] Parameter tuning
- [ ] Optional rayon parallelism

### Outstanding Issues
- [ ] 2 clippy warnings (codec.rs needless range loop, stream.rs collapsible if)
- [ ] No LICENSE-MIT / LICENSE-APACHE files
- [ ] No git commits yet — entire project is untracked

### Test Summary
- **44/44 tests passing** (40 unit + 4 WAV integration)
- All modules have unit tests
- CLI verified working end-to-end (embed + detect with broadband audio)

### Benchmark Results (release build)
| Benchmark | Time | Real-time ratio |
|-----------|------|-----------------|
| `embed_1s_48khz` | 223 µs | **~4500x real-time** |
| `detect_22s_48khz` | 4.9 ms | **~4500x real-time** |
| `stream_embed_1s (4096 chunks)` | 266 µs | **~3760x real-time** |
| `stream_detect_22s (4096 chunks)` | 4.7 ms | **~4680x real-time** |
| `fft_forward_inverse_1024` | 18.6 µs | — |

Conclusion: **easily real-time capable** — thousands of times faster than real-time for both embedding and detection.

### Files Implemented (13 .rs source + 2 test/bench)
| Module | Tests | Status |
|--------|-------|--------|
| lib.rs | — | Public API re-exports |
| error.rs | — | Error types |
| key.rs | 6 | AES-PRNG bin selection |
| fft.rs | 3 | realfft wrapper |
| config.rs | — | Configuration |
| payload.rs | 5 | 128-bit payload + CRC |
| patchwork.rs | 5 | Core embed/detect per frame |
| codec.rs | 7 | Conv. encoder + Viterbi |
| sync.rs | 5 | Sync pattern + correlation |
| frame.rs | 3 | Windowing + overlap-add |
| embed.rs | 3 | Embedding pipeline |
| detect.rs | 3 | Detection pipeline |
| stream.rs | 1 | Streaming wrappers |
| tests/wav_round_trip.rs | 4 | WAV file integration tests |
| benches/throughput.rs | 5 | Criterion benchmarks |
