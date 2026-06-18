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

echo "==> regenerating wasm registry"
"${REPO_ROOT}/scripts/regenerate-wasm-registry.sh"
echo "==> wasm-bindgen timeout: ${WASM_BINDGEN_TEST_TIMEOUT}s"
echo "==> cargo check runmat-core (wasm32 compatibility)"
cargo check -p runmat-core --target wasm32-unknown-unknown --no-default-features

"${WASM_SUITE}" symptom-closure
"${WASM_SUITE}" replay-smoke

if [[ "${RUNMAT_WASM_INCLUDE_RUNTIME:-0}" == "1" ]]; then
  "${WASM_SUITE}" runtime
else
  echo "==> skipping runmat-runtime (set RUNMAT_WASM_INCLUDE_RUNTIME=1 to attempt; requires gating I/O + ctor tests)"
fi

echo "All wasm headless tests completed."
