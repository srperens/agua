# Agua v0.2: Acoustic-Robust Watermarking

## Context

agua-core v0.1 is integrated in strom via agua-gst, but its current algorithm is too weak to survive lossy codecs (MP3/AAC) and especially the acoustic channel (speaker -> air -> microphone). The goal is to upgrade the algorithm with techniques proven by audiowmark: narrow frequency band, power-law encoding, Hann windowed overlap-add, and stronger FEC. This enables a future use case: WASM-compiled detection running in a mobile browser using the phone's microphone.

audiowmark compatibility is NOT required now but the architecture should allow adding it as an option later. There is no need to preserve backward compatibility with agua v0.1 - it is not used in production.

## Key Design Decisions

1. **No legacy/profile system.** agua v0.1 is not deployed anywhere. We replace the algorithm wholesale - simpler code, no branching.

2. **Sample rate stays configurable** (default 48000). Bin range computed dynamically from target frequencies (860-4300 Hz). At 48kHz/1024 FFT: bins 19-92.

3. **FEC: K=15, rate 1/6.** 16384 Viterbi states, ~5 MB memory. Acceptable for WASM on mobile. Architecture supports changing K later (e.g. K=16 for audiowmark compat).

4. **Hann window + overlap-add always on.** 50% hop. Frame advance = 512 samples, block duration ~11.6s at 48kHz (shorter than current 21.8s).

5. **Public API stays the same**: `embed()`, `detect()`, `StreamEmbedder`, `StreamDetector`, `WatermarkConfig`. Internals change, signatures don't.

## New Parameters (replaces v0.1 defaults)

| Parameter | Value |
|---|---|
| Frequency range | ~860-4300 Hz (computed from sample_rate) |
| Bin pairs/frame | 30 |
| Encoding | Power-law: `mag^(1+/-delta)` |
| FFT window | Hann + overlap-add (50% hop) |
| Frame advance | 512 samples |
| FEC constraint length K | 15 (16384 states) |
| FEC rate | 1/6 |
| Sync bits | 128 |
| Coded data bits | 960 (160 data * 6) |
| Frames/block | 1088 (128 sync + 960 data) |
| Block duration (48kHz) | ~11.6s |
| Default strength | 0.02 |
| Payload | 128 bits + CRC-32 (kept) |

## Phases

### Phase 1: Config + Power-Law Encoding

**Files:** `agua-core/src/config.rs`, `agua-core/src/patchwork.rs`

Config changes:
- Change defaults: `num_bin_pairs=30`, `strength=0.02`
- Replace `min_bin`/`max_bin` with `min_freq_hz: f32` (860.0) and `max_freq_hz: f32` (4300.0)
- Add `effective_bin_range(&self) -> (usize, usize)` computing bins from freq + sample_rate + frame_size
- Keep `frame_size=1024`, `sample_rate=48000`

Patchwork changes (replace, not branch):
- `embed_frame()`: power-law encoding `mag.powf(1.0 +/- delta)`, scale complex bin by magnitude ratio
- `detect_frame()`: log-ratio detection `(mag_a / mag_b).ln()` as soft value

**Test:** Embed+detect round-trip with new encoding. Verify soft values have correct sign.

### Phase 2: Hann Window + Overlap-Add

**Files:** `agua-core/src/frame.rs`, `agua-core/src/embed.rs`, `agua-core/src/detect.rs`, `agua-core/src/stream.rs`

- `frame.rs`: Pre-computed normalized Hann window. Overlap-add buffer utilities.
- `embed.rs`: Process with `hop_size=512` advance. Analysis window (Hann) before FFT, synthesis window (Hann) after IFFT, overlap-add output.
- `detect.rs`: Analysis frames with `hop_size` advance + Hann window before FFT.
- `stream.rs`: Update `StreamEmbedder`/`StreamDetector` buffering - retain `frame_size - hop_size` samples overlap between chunks.

**Test:** COLA unity gain (embed strength=0, output == input). Full embed+detect round-trip.

### Phase 3: Stronger FEC (K=15 Viterbi)

**Files:** `agua-core/src/codec.rs`, `agua-core/src/sync.rs`

Codec changes:
- Change constants: `CONSTRAINT_LENGTH=15`, generators to 6 good K=15 rate-1/6 polynomials (u16 wide)
- `NUM_STATES = 16384`
- Widen shift register and generator masks from u8 to u16
- Encoder and Viterbi decoder structurally unchanged, just wider state space
- WASM: use `u16` for state indices in survivor matrix to save memory

Sync changes:
- Increase sync pattern from 64 to 128 bits
- `generate_sync_pattern()`: encrypt two AES blocks to get 128 bits

**Test:** Encode/decode round-trip. Noise tolerance test (inject bit errors, verify K=15 corrects more than old K=7 could).

### Phase 4: Integration + Full Wiring

**Files:** `agua-core/src/embed.rs`, `agua-core/src/detect.rs`, `agua-core/src/stream.rs`, `agua-core/src/key.rs`, `agua-core/src/parallel.rs`

- Wire `config.effective_bin_range()` into bin pair generation in `key.rs`
- Update block size calculations throughout (1088 frames/block)
- Update `parallel.rs` - parallelization at block level (frames overlap, can't parallelize within block trivially)
- End-to-end integration of all Phase 1-3 changes

**Test:** Full round-trip embed+detect. Streaming embed+detect matches batch. Parallel matches sequential.

### Phase 5: GStreamer Element + CLI

**Files:** `agua-gst/src/embed.rs`, `agua-cli/src/main.rs`

- GStreamer: update property defaults (`strength=0.02`), replace `min-bin`/`max-bin` props with `min-freq`/`max-freq`, update `num-bin-pairs` default to 30
- CLI: update defaults

**No strom changes needed** beyond bumping the agua-gst dependency tag.

### Phase 6: Robustness Validation

**Files:** `agua-core/tests/`

- Update `lossy_codec_round_trip.rs` for new parameters
- New `acoustic_simulation.rs`: low-pass filter (8kHz) + white noise (20dB SNR) + resample jitter -> detect succeeds
- Update `benches/throughput.rs` with new algorithm benchmarks
- WASM compile test: `cargo build --target wasm32-unknown-unknown -p agua-core`

## Phase Dependencies

```
Phase 1 (Config + Power-law) ──┐
Phase 2 (Overlap-add) ─────────┼──> Phase 4 (Integration) ──> Phase 5 (GST+CLI)
Phase 3 (FEC K=15) ────────────┘                          ──> Phase 6 (Validation)
```

Phases 1, 2, 3 can be developed largely in parallel (they touch different files), with Phase 4 integrating them.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| K=15 polynomial selection | Bad polynomials = weak FEC | Use published tables (Proakis, Lin & Costello). Fallback: K=13 (4096 states) |
| Overlap-add streaming complexity | Bugs in StreamEmbedder buffering | Compare streaming vs batch output in tests |
| WASM Viterbi memory (~5 MB) | Might be tight on low-end phones | K=13 fallback (1.3 MB) if needed; test on real devices |

## Verification

1. `cargo test` - all tests pass
2. `cargo test --features parallel` - parallel mode tests pass
3. Lossy codec round-trip: embed, encode MP3 128k, detect -> success
4. Acoustic simulation: low-pass + noise -> detects successfully
5. GStreamer: `gst-launch-1.0 audiotestsrc ! audioconvert ! aguaembed payload=... ! fakesink` runs
6. WASM: `cargo build --target wasm32-unknown-unknown -p agua-core` compiles

## License

Dual MIT/Apache-2.0 (unchanged).
