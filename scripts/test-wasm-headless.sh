#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROME_WRAPPER="${REPO_ROOT}/scripts/chrome-headless.sh"

export CHROME_BIN="${CHROME_BIN:-${CHROME_WRAPPER}}"
export WASM_BINDGEN_TEST_TIMEOUT="${WASM_BINDGEN_TEST_TIMEOUT:-120}"
export RUNMAT_GENERATE_WASM_REGISTRY=1

echo "==> regenerating wasm registry"
cargo check -p runmat-runtime --target wasm32-unknown-unknown >/dev/null

run_crate_tests () {
  local crate="$1"
  shift
  echo "==> wasm-pack test ${crate} $*"
  pushd "${REPO_ROOT}/crates/${crate}" >/dev/null
  wasm-pack test --chrome --headless "$@" >/dev/null
  popd >/dev/null
}

# Core crates that ship in the wasm bundle today.
run_crate_tests runmat-core -- --no-default-features

if [[ "${RUNMAT_WASM_INCLUDE_RUNTIME:-0}" == "1" ]]; then
  run_crate_tests runmat-runtime -- --no-default-features --features plot-web
else
  echo "==> skipping runmat-runtime (set RUNMAT_WASM_INCLUDE_RUNTIME=1 to attempt; requires gating I/O + ctor tests)"
fi

echo "All wasm headless tests completed."

