#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WASM_SUITE="${REPO_ROOT}/scripts/runtime/test-wasm-regression-suite.sh"

export CHROME_BIN="${CHROME_BIN:-${REPO_ROOT}/scripts/runtime/chrome-headless.sh}"
export CHROMEDRIVER_ARGS="${CHROMEDRIVER_ARGS:---log-level=SEVERE}"
export WASM_BINDGEN_TEST_TIMEOUT="${WASM_BINDGEN_TEST_TIMEOUT:-300}"
# Ensure at least opt-level=1 so the test binary stays within the wasm
# spec limit on locals per function (opt-level=0 exceeds it).
# In CI this is already set at the job level; this is a local-run fallback.
export RUSTFLAGS="${RUSTFLAGS:--Copt-level=1}"

echo "==> wasm-bindgen timeout: ${WASM_BINDGEN_TEST_TIMEOUT}s"
"${WASM_SUITE}" all

echo "All wasm headless tests completed."
