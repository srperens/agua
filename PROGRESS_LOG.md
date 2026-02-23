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
- [ ] Integration tests with WAV files (no standalone WAV integration tests yet)
- [ ] Criterion benchmarks (not added)

### Phase 5: Robustness — NOT STARTED
- [ ] MP3/AAC round-trip robustness tests
- [ ] Parameter tuning
- [ ] Optional rayon parallelism

### Outstanding Issues
- [ ] 2 clippy warnings (codec.rs needless range loop, stream.rs collapsible if)
- [ ] No LICENSE-MIT / LICENSE-APACHE files
- [ ] No criterion benchmarks (dev-dependency not added)
- [ ] No git commits yet — entire project is untracked
- [ ] `cargo fmt` must be run before committing (currently clean)

### Test Summary
- **40/40 tests passing** (including end-to-end embed/detect round-trip)
- All modules have unit tests
- `stream.rs` has minimal coverage (1 test)

### Files Implemented (13 .rs files)
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
