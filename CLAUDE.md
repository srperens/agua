## Language & Tools
- Rust (edition 2024)
- cargo, clippy, rustfmt
- Tests: cargo test / cargo nextest
- All code, comments, PRs, and commit messages in English

## Project Structure
- `agua-core` — core library crate (watermarking algorithm)
- `agua-gst` — GStreamer plugin crate (`aguaembed` element)
- `agua-cli` — CLI binary crate

## Algorithm Overview
Patchwork-based audio watermarking in the FFT domain. Embeds a 128-bit payload into audio by modifying frequency bin magnitudes. See `PLAN.md` for the full v0.2 design.

Key parameters (v0.2):
- **Frequency range:** 860-4300 Hz (bin range computed dynamically from sample_rate)
- **Bin pairs/frame:** 60
- **Encoding:** Power-law: `mag^(1+/-delta)` (NOT linear)
- **FFT window:** Hann + overlap-add with 50% hop (frame_size=1024, hop_size=512)
- **FEC:** Convolutional code K=15, rate 1/6 (16384 Viterbi states)
- **Sync pattern:** 128 bits (two AES blocks)
- **Block structure:** 128 sync frames + 960 data frames = 1088 frames/block
- **Payload:** 128 bits + CRC-32, convolutionally encoded to 960 coded bits
- **Default strength:** 0.1

There is no legacy/profile system. v0.1 is not deployed anywhere.

## Public API (agua-core)
```rust
// One-shot
pub fn embed(samples: &mut [f32], payload: &Payload, key: &WatermarkKey, config: &WatermarkConfig) -> Result<()>;
pub fn detect(samples: &[f32], key: &WatermarkKey, config: &WatermarkConfig) -> Result<Vec<DetectionResult>>;

// Streaming (for GStreamer / real-time)
pub struct StreamEmbedder;  // process(&mut self, input: &[f32]) -> Vec<f32>
pub struct StreamDetector;  // process(&mut self, input: &[f32]) -> Vec<DetectionResult>
```

All APIs operate on mono f32 sample buffers. No file I/O in the core library.

## WASM Compatibility
agua-core must compile to `wasm32-unknown-unknown`. Avoid dependencies that require std I/O or system calls. The WASM target enables browser-based watermark detection via microphone.

## Code Standards
- Run `cargo fmt` before commit
- No clippy warnings allowed
- All public APIs must have doc comments
- Error handling with thiserror, no unwrap() in production code
- No `unsafe` code without explicit justification and review
- Follow Rust API guidelines (RFC 430): snake_case modules, CamelCase types
- Write unit tests for all new logic

## Dependencies
- Minimize dependencies; prefer well-maintained crates
- Audit and justify new dependency additions
- Be mindful of compile time and binary size impact
- All dependencies must support `wasm32-unknown-unknown`

## Logging
- Use `tracing` for structured logging
- Log levels: error (failures), warn (recoverable issues), info (key events), debug/trace (development)

## Security
- Never commit secrets, credentials, `.env` files, or infrastructure config
- Keep `.gitignore` up to date (target/, .env, *.pem, editor files)

## Git Conventions
- Conventional commits: feat:, fix:, refactor:, test:, docs:, chore:
- One commit per logical change

## CI
- All of fmt, clippy, and test must pass before merge

## Teamwork
- Architecture changes require plan approval
- Code review required before merge
- All tests must pass before a task is marked complete
