use serde::Serialize;

/// Physical source provenance emitted by `#[runtime_builtin]`.
///
/// This is development evidence derived from the same declaration that creates
/// the executable registration. It is not a builtin contract or dispatch
/// authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinImplementationProvenance {
    pub name: &'static str,
    pub binding_variant: Option<&'static str>,
    pub source_file: &'static str,
    pub module_path: &'static str,
    pub function: &'static str,
    pub builtin_path: &'static str,
    pub authority: BuiltinImplementationAuthority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinImplementationAuthority {
    CanonicalBinding,
    LegacyFunction,
}

#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(BuiltinImplementationProvenance);

#[cfg(not(target_arch = "wasm32"))]
pub fn builtin_implementation_provenance() -> Vec<&'static BuiltinImplementationProvenance> {
    inventory::iter::<BuiltinImplementationProvenance>
        .into_iter()
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub fn builtin_implementation_provenance() -> Vec<&'static BuiltinImplementationProvenance> {
    crate::wasm_registry::builtin_implementation_provenance()
}
