#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY_PATH="${REPO_ROOT}/crates/runmat-runtime/src/builtins/generated_wasm_registry.rs"
TMP_DIR="${REPO_ROOT}/target/wasm-registry"

mkdir -p "${TMP_DIR}"
TMP_REGISTRY="$(mktemp "${TMP_DIR}/generated_wasm_registry.XXXXXX")"
rm -f "${TMP_REGISTRY}"

cleanup() {
  rm -f "${TMP_REGISTRY}"
}
trap cleanup EXIT

pushd "${REPO_ROOT}" >/dev/null
echo "==> generating wasm builtin registry for runmat-runtime/plot-web,occt-wasm-host"
cargo clean -p runmat-runtime --target wasm32-unknown-unknown >/dev/null
RUNMAT_GENERATE_WASM_REGISTRY=1 \
RUNMAT_WASM_REGISTRY_OUT="${TMP_REGISTRY}" \
cargo check -p runmat-runtime --target wasm32-unknown-unknown --no-default-features --features plot-web,occt-wasm-host >/dev/null

entry_count="$(grep -c "__runmat_wasm_register_" "${TMP_REGISTRY}" || true)"
builtin_count="$(grep -c "__runmat_wasm_register_builtin_" "${TMP_REGISTRY}" || true)"
if [[ "${entry_count}" -le 0 || "${builtin_count}" -le 0 ]]; then
  echo "generated wasm registry is empty or incomplete (${entry_count} entries, ${builtin_count} builtins)" >&2
  exit 1
fi

perl -0pi -e "s/pub const REGISTRY_COMPLETE: bool = false;/pub const REGISTRY_COMPLETE: bool = true;/; s/pub const REGISTRY_ENTRY_COUNT: usize = 0;/pub const REGISTRY_ENTRY_COUNT: usize = ${entry_count};/" "${TMP_REGISTRY}"

if ! grep -q "pub const REGISTRY_COMPLETE: bool = true;" "${TMP_REGISTRY}"; then
  echo "failed to mark generated wasm registry complete" >&2
  exit 1
fi
if ! grep -q "pub const REGISTRY_ENTRY_COUNT: usize = ${entry_count};" "${TMP_REGISTRY}"; then
  echo "failed to stamp generated wasm registry entry count" >&2
  exit 1
fi

mv "${TMP_REGISTRY}" "${REGISTRY_PATH}"
trap - EXIT
popd >/dev/null

echo "==> wrote ${REGISTRY_PATH} (${entry_count} registry entries, ${builtin_count} builtins)"
