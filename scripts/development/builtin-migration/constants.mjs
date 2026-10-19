export const CATALOG_ROOT = "crates/runmat-builtins/src/catalog/entries";
export const RUNTIME_ROOT = "crates/runmat-runtime/src/builtins";
export const SIDECAR_ROOT = "docs/builtins/reference";
export const SHADOW_ROOT = `${RUNTIME_ROOT}/builtins-json`;
export const WASM_REGISTRY = `${RUNTIME_ROOT}/generated_wasm_registry.rs`;

export function compareCodePoint(left, right) {
  const a = Array.from(String(left), (character) => character.codePointAt(0));
  const b = Array.from(String(right), (character) => character.codePointAt(0));
  const sharedLength = Math.min(a.length, b.length);
  for (let index = 0; index < sharedLength; index += 1) {
    if (a[index] !== b[index]) return a[index] < b[index] ? -1 : 1;
  }
  return a.length < b.length ? -1 : a.length > b.length ? 1 : 0;
}

export function sorted(values) { return [...values].sort(compareCodePoint); }
export function unique(values) { return sorted(new Set(values)); }
export function safeRead(repository, sourcePath, read) { try { return read(repository, sourcePath); } catch { return ""; } }
export function rustLeaf(identity) {
  return identity.replace(/([a-z0-9])([A-Z])/g, "$1_$2").replace(/[^A-Za-z0-9]+/g, "_").toLowerCase();
}
