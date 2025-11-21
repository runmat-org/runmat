#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script (website/scripts)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEBSITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$WEBSITE_DIR/.." && pwd)"

# Ensure cargo is available (install minimal toolchain if needed)
if ! command -v cargo >/dev/null 2>&1; then
  echo "[export-builtins] Installing Rust toolchain via rustup" >&2
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi

# Run the exporter with doc_export feature, targeting the workspace manifest
OUT_PATH="$WEBSITE_DIR/content/builtins.json"
CMD=(cargo run --manifest-path "$ROOT_DIR/Cargo.toml" -p runmat-runtime --bin export_builtins --features doc_export -- --out "$OUT_PATH")

echo "[export-builtins] Running: ${CMD[*]}" >&2
"${CMD[@]}"

echo "[export-builtins] Wrote $OUT_PATH" >&2
