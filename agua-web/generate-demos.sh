#!/bin/bash
#
# Generate demo WAV files for the agua web detector.
#
# Downloads the Sintel trailer audio from Xiph.org (Jan Morgenstern,
# CC BY 3.0), loops it to ~3 min, converts to 48 kHz mono WAV,
# then embeds watermarks at multiple strengths.
#
# Each strength level gets a unique payload for easy identification.
#
# Requirements: curl, ffmpeg, cargo (builds agua-cli if needed)
#
# Source music:
#   Sintel trailer OST — https://media.xiph.org/sintel/
#   License: CC BY 3.0 — https://creativecommons.org/licenses/by/3.0/
#   Credit: Jan Morgenstern
#
# Other free music sources:
#   Big Buck Bunny OST (CC BY 3.0): https://archive.org/details/JanMorgenstern-BigBuckBunny
#   Kevin MacLeod / Incompetech (CC BY 4.0): https://incompetech.com/music/royalty-free/music.html
#   Xiph.org test media: https://media.xiph.org/
#   Free Music Archive: https://freemusicarchive.org/
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$SCRIPT_DIR/static"
SOURCE_DIR="$SCRIPT_DIR/source"
SOURCE_FLAC="$SOURCE_DIR/sintel-trailer-audio.flac"
SOURCE_WAV="$SOURCE_DIR/sintel-trailer-48k-mono.wav"
SOURCE_URL="https://media.xiph.org/sintel/sintel_trailer-audio.flac"
LOOP_DURATION=200  # ~3:20, enough for multiple detection blocks

# Strengths and their payloads (last 8 hex chars encode strength)
declare -A DEMOS=(
  ["demo-s007.wav"]="0.07 deadbeef0123456789abcdef00070707"
  ["demo-s008.wav"]="0.08 deadbeef0123456789abcdef00080808"
  ["demo-s010.wav"]="0.10 deadbeef0123456789abcdef00101010"
  ["demo-s015.wav"]="0.15 deadbeef0123456789abcdef00151515"
)

# --- Download and convert source ---
mkdir -p "$SOURCE_DIR"
if [ ! -f "$SOURCE_WAV" ]; then
  if [ ! -f "$SOURCE_FLAC" ]; then
    echo "=== Downloading Sintel trailer audio (4.5 MB) ==="
    curl -L "$SOURCE_URL" -o "$SOURCE_FLAC"
  fi
  echo "=== Looping to ${LOOP_DURATION}s and converting to 48 kHz mono WAV ==="
  ffmpeg -y -stream_loop 3 -i "$SOURCE_FLAC" -t "$LOOP_DURATION" -ar 48000 -ac 1 "$SOURCE_WAV" 2>&1 | tail -1
fi
echo "Source: $SOURCE_WAV ($(du -h "$SOURCE_WAV" | cut -f1))"

# --- Build CLI ---
echo "=== Building agua CLI ==="
cargo build -p agua-cli --release --manifest-path "$REPO_DIR/Cargo.toml" 2>&1 | tail -3
AGUA="$REPO_DIR/target/release/agua"

# --- Embed demos ---
echo "=== Embedding demos ==="
for file in "${!DEMOS[@]}"; do
  read -r strength payload <<< "${DEMOS[$file]}"
  echo "  $file  strength=$strength  payload=$payload"
  "$AGUA" embed \
    -i "$SOURCE_WAV" \
    -o "$OUT_DIR/$file" \
    -s "$strength" \
    -p "$payload"
done

# --- Verify ---
echo ""
echo "=== Verifying detection ==="
all_ok=true
for file in "${!DEMOS[@]}"; do
  read -r strength payload <<< "${DEMOS[$file]}"
  if "$AGUA" detect -i "$OUT_DIR/$file" 2>&1 | grep -q "$payload"; then
    echo "  $file  OK"
  else
    echo "  $file  FAILED (may still work via streaming/acoustic)"
    all_ok=false
  fi
done

echo ""
if $all_ok; then
  echo "All demos generated and verified."
else
  echo "Some demos did not detect in single-pass (weak strengths may still work in streaming mode)."
fi
echo ""
echo "Credit: Music by Jan Morgenstern, from Sintel (CC BY 3.0)"
