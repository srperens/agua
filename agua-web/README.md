# agua-web

Browser-based real-time audio watermark detection using agua-core compiled to WASM.

Open the page on any device, grant microphone access, and it listens for agua watermarks in ambient audio.

## Prerequisites

- [Rust](https://rustup.rs/) with `wasm32-unknown-unknown` target
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## Build

```bash
./build.sh
```

This runs `wasm-pack build --target web` and outputs to `static/pkg/`.

## Run

```bash
cd static
python3 -m http.server 8080
```

Open http://localhost:8080 in a browser.

## Mobile testing

`getUserMedia` requires a secure context (HTTPS) on mobile browsers.

Options:

- **localhost** works without HTTPS on desktop browsers
- **npx serve**: `cd static && npx serve --ssl` (generates self-signed cert)
- **ngrok**: `ngrok http 8080` for a public HTTPS URL
- **Self-signed cert + nginx**: reverse proxy with TLS termination

## How it works

1. The browser captures microphone audio via `getUserMedia`
2. An `AudioWorkletProcessor` forwards 128-sample chunks to the main thread
3. The main thread accumulates samples and periodically feeds them to the WASM detector
4. The detector (agua-core's `StreamDetector`) performs FFT-based patchwork analysis
5. When a full watermark block is received (~11 seconds at 48kHz), detection is attempted
6. Detected payloads are displayed as 32-character hex strings

## Configuration

The "key passphrase" must match the key used when embedding the watermark. The default is `agua-demo`.
