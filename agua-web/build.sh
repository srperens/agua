#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Building agua-web WASM module..."
wasm-pack build --target web --out-dir static/pkg --release

echo ""
echo "Build complete. Serve with:"
echo "  cd static && python3 -m http.server 8080"
