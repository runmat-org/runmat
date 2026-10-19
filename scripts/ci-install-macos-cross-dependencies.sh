#!/usr/bin/env bash
set -euo pipefail

target="${1:?usage: ci-install-macos-cross-dependencies.sh <rust-target>}"

if brew tap | grep -qx 'aws/tap'; then
    brew untap aws/tap
fi
brew update
brew list cmake >/dev/null 2>&1 || brew install cmake
brew list zeromq >/dev/null 2>&1 || brew install zeromq

if [[ "$target" != "aarch64-apple-darwin" && "$target" != "x86_64-apple-darwin" ]]; then
    echo "unsupported macOS target: $target" >&2
    exit 1
fi
