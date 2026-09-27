use super::{
    builtin_fusion_spec_registrations, BuiltinFusionSpec, FusionSemanticDeclaration,
    FusionStructuralOperation,
};
use runmat_builtins::{
    builtin_catalog_entry_by_name, builtin_implementation_provenance, BuiltinBindingIdentity,
    BuiltinImplementationAuthority,
};

/// Resolve a fusion declaration for an already selected canonical binding.
///
/// Typed semantic declarations identify the exact binding variant. Legacy
/// declarations without that field authorize only an unambiguous,
/// single-binding catalog entry. Duplicate declarations fail closed.
/// The selected binding's compiled implementation provenance is the join key.
/// A spec's textual name is checked for consistency, never used to select an
/// implementation. Whole-registry validation remains the inventory's owner.
#[derive(Clone, Copy)]
pub struct CanonicalFusionDeclaration {
    pub binding: BuiltinBindingIdentity,
    pub spec: &'static BuiltinFusionSpec,
    pub semantics: Option<&'static FusionSemanticDeclaration>,
    pub structural: Option<FusionStructuralOperation>,
}

impl std::fmt::Debug for CanonicalFusionDeclaration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CanonicalFusionDeclaration")
            .field("binding", &self.binding)
            .field(
                "revision",
                &self.semantics.map(|semantics| semantics.revision),
            )
            .finish_non_exhaustive()
    }
}

pub fn canonical_fusion_spec(
    identity: BuiltinBindingIdentity,
) -> Option<&'static BuiltinFusionSpec> {
    canonical_fusion_declaration(identity).map(|declaration| declaration.spec)
}

pub fn canonical_fusion_declaration(
    identity: BuiltinBindingIdentity,
) -> Option<CanonicalFusionDeclaration> {
    let entry = builtin_catalog_entry_by_name(identity.builtin.name)?;
    if !entry
        .bindings
        .iter()
        .any(|binding| entry.binding_identity(binding) == identity)
    {
        return None;
    }
    let provenance = builtin_implementation_provenance();
    let mut implementations = provenance.iter().filter(|implementation| {
        implementation.authority == BuiltinImplementationAuthority::CanonicalBinding
            && implementation.name == identity.builtin.name
            && implementation.binding_variant == Some(identity.variant)
    });
    let implementation = implementations.next()?;
    if implementations.next().is_some() {
        return None;
    }
    let mut candidates = builtin_fusion_spec_registrations().filter(|spec| {
        spec.builtin_path == implementation.builtin_path
            && match spec.semantics {
                Some(semantics) => semantics.binding == identity,
                None => entry.bindings.len() == 1,
            }
    });
    let registration = candidates.next()?;
    if candidates.next().is_some() || registration.spec.name != identity.builtin.name {
        return None;
    }
    // A structural operation needs explicit binding authority and must not
    // compete with an unrelated scalar/reduction execution declaration.
    if registration.structural.is_some()
        && (registration.semantics.is_none()
            || registration.spec.elementwise.is_some()
            || registration.spec.reduction.is_some())
    {
        return None;
    }
    Some(CanonicalFusionDeclaration {
        binding: identity,
        spec: registration.spec,
        semantics: registration.semantics,
        structural: registration.structural,
    })
}
