//! Operation-owned fixed-phase epilogues, independent of scalar/reduction templates.
use super::{builtin_fusion_spec_registrations, canonical_fusion_declaration};
use runmat_builtins::{
    builtin_catalog_entry_by_name, builtin_implementation_provenance,
    BuiltinImplementationAuthority,
};

/// These declarations authorize a form, not arbitrary arguments: the execution
/// planner must validate represented classes, exact arity and phase ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionEpilogueOperation {
    Maximum,
    Minimum,
    RealPower,
    Diagonal,
}

impl FusionEpilogueOperation {
    pub const fn input_arity(self) -> usize {
        match self {
            Self::Diagonal => 1,
            Self::Maximum | Self::Minimum | Self::RealPower => 2,
        }
    }
}

/// A scoped service is dispatch authority. Catalog provenance alone cannot
/// authorize bypassing an override, an absent binding, or multiple overloads.
/// Legacy inventory is never consulted when a scoped service owns the name.
pub fn fusion_binding_is_selected(
    name: &str,
    context: Option<&crate::context::RuntimeContext>,
) -> bool {
    let active = crate::context::legacy::active();
    let Some(service) = context
        .or(active.as_ref())
        .and_then(|c| c.service_ports().builtin())
    else {
        return true;
    };
    let selected = service.bindings_by_name(name);
    let compiled = crate::builtin::runtime_builtin_bindings_by_name(name);
    let ([selected], [compiled]) = (selected.as_slice(), compiled.as_slice()) else {
        return false;
    };
    selected.identity == compiled.identity
        && std::ptr::fn_addr_eq(selected.implementation, compiled.implementation)
}

/// Canonical declarations take precedence. The compatibility join for an
/// unmigrated identity requires exactly one compiled implementation and one
/// registration at its physical owner; a matching spec name alone is not proof.
pub fn fusion_epilogue_operation(
    name: &str,
    context: Option<&crate::context::RuntimeContext>,
) -> Option<FusionEpilogueOperation> {
    if !fusion_binding_is_selected(name, context) {
        return None;
    }
    if let Some(entry) = builtin_catalog_entry_by_name(name) {
        let [binding] = entry.bindings else {
            return None;
        };
        return canonical_fusion_declaration(entry.binding_identity(binding))?.epilogue;
    }
    let provenance = builtin_implementation_provenance();
    let mut implementations = provenance.iter().filter(|p| {
        p.name == name && p.authority == BuiltinImplementationAuthority::LegacyFunction
    });
    let implementation = implementations.next()?;
    if implementations.next().is_some() {
        return None;
    }
    let mut declarations = builtin_fusion_spec_registrations()
        .filter(|s| s.builtin_path == implementation.builtin_path && s.spec.name == name);
    let declaration = declarations.next()?;
    if declarations.next().is_some() {
        return None;
    }
    declaration.epilogue
}
