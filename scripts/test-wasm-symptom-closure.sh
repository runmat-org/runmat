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

pushd "${REPO_ROOT}/crates/runmat-wasm" >/dev/null

echo "==> wasm-pack test --node --test symptom_node_regressions"
wasm-pack test --node --test symptom_node_regressions

echo "==> wasm-pack test --chrome --headless --test symptom_browser_regressions"
wasm-pack test --chrome --headless "${WASM_PACK_CHROMEDRIVER_ARGS[@]}" --test symptom_browser_regressions

popd >/dev/null
