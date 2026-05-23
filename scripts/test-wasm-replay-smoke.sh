#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROME_WRAPPER="${REPO_ROOT}/scripts/chrome-headless.sh"
CHROMEDRIVER_RESOLVER="${REPO_ROOT}/scripts/resolve-chromedriver.sh"

export CHROME_BIN="${CHROME_BIN:-${CHROME_WRAPPER}}"
export CHROMEDRIVER_ARGS="${CHROMEDRIVER_ARGS:---log-level=SEVERE}"
export WASM_BINDGEN_TEST_TIMEOUT="${WASM_BINDGEN_TEST_TIMEOUT:-300}"

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

echo "==> wasm-pack test runmat-wasm replay_smoke"
pushd "${REPO_ROOT}/crates/runmat-wasm" >/dev/null
wasm-pack test --chrome --headless "${WASM_PACK_CHROMEDRIVER_ARGS[@]}" --test replay_smoke
popd >/dev/null
