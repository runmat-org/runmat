#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROME_WRAPPER="${REPO_ROOT}/scripts/chrome-headless.sh"
CHROMEDRIVER_RESOLVER="${REPO_ROOT}/scripts/resolve-chromedriver.sh"

export CHROME_BIN="${CHROME_BIN:-${CHROME_WRAPPER}}"
export CHROMEDRIVER_ARGS="${CHROMEDRIVER_ARGS:---log-level=SEVERE}"
export WASM_BINDGEN_TEST_TIMEOUT="${WASM_BINDGEN_TEST_TIMEOUT:-300}"
# Ensure at least opt-level=1 so the test binary stays within the wasm
# spec limit on locals per function (opt-level=0 exceeds it).
# In CI this is already set at the job level; this is a local-run fallback.
export RUSTFLAGS="${RUSTFLAGS:--Copt-level=1}"

WASM_PACK_CHROMEDRIVER_ARGS=()
if [[ -z "${RUNMAT_CHROMEDRIVER_BIN:-}" ]] && [[ -x "${CHROMEDRIVER_RESOLVER}" ]]; then
  RUNMAT_CHROMEDRIVER_BIN="$("${CHROMEDRIVER_RESOLVER}" 2>/dev/null || true)"
fi
if [[ -n "${RUNMAT_CHROMEDRIVER_BIN:-}" ]]; then
  WASM_PACK_CHROMEDRIVER_ARGS=(--chromedriver "${RUNMAT_CHROMEDRIVER_BIN}")
  echo "==> using chromedriver: ${RUNMAT_CHROMEDRIVER_BIN}"
else
  echo "==> using wasm-pack default chromedriver resolution"
fi

echo "==> regenerating wasm registry"
"${REPO_ROOT}/scripts/regenerate-wasm-registry.sh"
echo "==> wasm-bindgen timeout: ${WASM_BINDGEN_TEST_TIMEOUT}s"

run_crate_tests () {
  local crate="$1"
  shift
  echo "==> wasm-pack test ${crate} $*"
  pushd "${REPO_ROOT}/crates/${crate}" >/dev/null
  wasm-pack test --chrome --headless "${WASM_PACK_CHROMEDRIVER_ARGS[@]}" "$@"
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
