#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHROME_WRAPPER="${REPO_ROOT}/scripts/runtime/chrome-headless.sh"
CHROMEDRIVER_RESOLVER="${REPO_ROOT}/scripts/runtime/resolve-chromedriver.sh"

usage() {
  cat <<'USAGE'
Usage:
  scripts/runtime/test-wasm-regression-suite.sh <suite>

Suites:
  symptom-closure   Run focused wasm symptom closure proofs (node + browser)
  replay-smoke      Run replay smoke browser tests
  runtime           Run runtime browser tests behind RUNMAT_WASM_INCLUDE_RUNTIME
USAGE
}

resolve_chromedriver_args() {
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
}

regenerate_wasm_registry() {
  echo "==> regenerating wasm builtin registry"
  "${REPO_ROOT}/scripts/regenerate-wasm-registry.sh"
}

run_symptom_closure_suite() {
  echo "==> wasm-pack test --node --test symptom_node_regressions"
  wasm-pack test --node --test symptom_node_regressions

  echo "==> wasm-pack test --chrome --headless --test symptom_browser_regressions"
  wasm-pack test --chrome --headless "${WASM_PACK_CHROMEDRIVER_ARGS[@]}" --test symptom_browser_regressions
}

run_replay_smoke_suite() {
  echo "==> wasm-pack test --chrome --headless --test replay_smoke"
  wasm-pack test --chrome --headless "${WASM_PACK_CHROMEDRIVER_ARGS[@]}" --test replay_smoke
}

run_runtime_suite() {
  echo "==> wasm-pack test runmat-runtime --no-default-features --features plot-web"
  pushd "${REPO_ROOT}/crates/runmat-runtime" >/dev/null
  wasm-pack test --chrome --headless "${WASM_PACK_CHROMEDRIVER_ARGS[@]}" -- --no-default-features --features plot-web
  popd >/dev/null
}

main() {
  local suite="${1:-}"
  if [[ -z "${suite}" ]]; then
    usage
    exit 1
  fi

  export CHROME_BIN="${CHROME_BIN:-${CHROME_WRAPPER}}"
  export CHROMEDRIVER_ARGS="${CHROMEDRIVER_ARGS:---log-level=SEVERE}"
  export WASM_BINDGEN_TEST_TIMEOUT="${WASM_BINDGEN_TEST_TIMEOUT:-300}"
  export RUSTFLAGS="${RUSTFLAGS:--Copt-level=1}"

  resolve_chromedriver_args

  regenerate_wasm_registry

  pushd "${REPO_ROOT}/crates/runmat-wasm" >/dev/null
  case "${suite}" in
    symptom-closure)
      run_symptom_closure_suite
      ;;
    replay-smoke)
      run_replay_smoke_suite
      ;;
    runtime)
      run_runtime_suite
      ;;
    *)
      usage
      popd >/dev/null
      exit 1
      ;;
  esac
  popd >/dev/null
}

main "$@"
