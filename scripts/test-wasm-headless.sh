#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROME_WRAPPER="${REPO_ROOT}/scripts/chrome-headless.sh"

export CHROME_BIN="${CHROME_BIN:-${CHROME_WRAPPER}}"
export CHROMEDRIVER_ARGS="${CHROMEDRIVER_ARGS:---log-level=SEVERE}"
export WASM_BINDGEN_TEST_TIMEOUT="${WASM_BINDGEN_TEST_TIMEOUT:-300}"
export RUNMAT_GENERATE_WASM_REGISTRY=1
# Ensure at least opt-level=1 so the test binary stays within the wasm
# spec limit on locals per function (opt-level=0 exceeds it).
# In CI this is already set at the job level; this is a local-run fallback.
export RUSTFLAGS="${RUSTFLAGS:--Copt-level=1}"

echo "==> regenerating wasm registry"
cargo check -p runmat-runtime --target wasm32-unknown-unknown >/dev/null
# Unset so wasm-pack test does not re-trigger proc-macro writes to the registry
# file (which would cause an infinite rebuild loop via cargo's include! tracking).
unset RUNMAT_GENERATE_WASM_REGISTRY
echo "==> wasm-bindgen timeout: ${WASM_BINDGEN_TEST_TIMEOUT}s"

run_crate_tests () {
  local crate="$1"
  shift
  echo "==> wasm-pack test ${crate} $*"
  pushd "${REPO_ROOT}/crates/${crate}" >/dev/null
  wasm-pack test --chrome --headless "$@"
  popd >/dev/null
}

# runmat-core has no #[wasm_bindgen_test] tests yet; wasm-pack test would
# compile a 7+ GB wasm binary per test file for zero runnable tests in the
# browser. Use cargo check instead to verify wasm32 compatibility cheaply.
# runmat-wasm (built in the preceding CI step) already depends on runmat-core,
# so full compilation is also verified there.
echo "==> cargo check runmat-core (wasm32 compatibility)"
cargo check -p runmat-core --target wasm32-unknown-unknown --no-default-features

if [[ "${RUNMAT_WASM_INCLUDE_RUNTIME:-0}" == "1" ]]; then
  run_crate_tests runmat-runtime -- --no-default-features --features plot-web
else
  echo "==> skipping runmat-runtime (set RUNMAT_WASM_INCLUDE_RUNTIME=1 to attempt; requires gating I/O + ctor tests)"
fi

echo "All wasm headless tests completed."
