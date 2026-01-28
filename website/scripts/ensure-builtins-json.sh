#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEBSITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$WEBSITE_DIR/.." && pwd)"

TARGET="$WEBSITE_DIR/content/builtins-json"
SOURCE="$ROOT_DIR/crates/runmat-runtime/src/builtins/builtins-json"

if [ -L "$TARGET" ]; then
  echo "[ensure-builtins-json] Using existing symlink at $TARGET" >&2
  exit 0
fi

if [ -d "$TARGET" ]; then
  echo "[ensure-builtins-json] Directory already exists at $TARGET" >&2
  exit 0
fi

if [ ! -d "$SOURCE" ]; then
  echo "[ensure-builtins-json] Source directory not found at $SOURCE; skipping copy." >&2
  exit 0
fi

cp -r "$SOURCE" "$TARGET"
echo "[ensure-builtins-json] Copied builtins-json to $TARGET" >&2
