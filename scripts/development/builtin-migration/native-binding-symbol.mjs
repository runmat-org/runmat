// Mirrors the versioned Rust native_binding_symbol ABI encoder. All migration
// products and validators use this one encoder for the current binding ABI.
export function nativeBindingSymbol(name, variant) {
  return `runmat_builtin_binding_v3_${Buffer.from(name).toString("hex")}_${Buffer.from(variant).toString("hex")}`;
}
