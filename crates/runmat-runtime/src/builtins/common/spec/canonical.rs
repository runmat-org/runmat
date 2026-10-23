use super::{
    registration_index::with_registrations, BuiltinFusionSpec, FusionEpilogueOperation,
    FusionSemanticDeclaration, FusionStructuralOperation,
};
use runmat_builtins::{builtin_catalog_entry_by_name, BuiltinBindingIdentity};

/// Resolve a fusion declaration for an already selected canonical binding.
///
/// Typed semantic declarations identify the exact binding variant. Legacy
/// declarations without that field authorize only an unambiguous,
/// single-binding catalog entry. Duplicate declarations fail closed.
/// The selected local Runtime binding owns its physical declaration join.
/// Process-wide provenance remains evidence, never execution authority.
/// A spec's textual name is checked for consistency, never used to select an
/// implementation. Whole-registry validation remains the inventory's owner.
#[derive(Clone, Copy)]
pub struct CanonicalFusionDeclaration {
    pub binding: BuiltinBindingIdentity,
    pub spec: &'static BuiltinFusionSpec,
    pub semantics: Option<&'static FusionSemanticDeclaration>,
    pub structural: Option<FusionStructuralOperation>,
    pub epilogue: Option<FusionEpilogueOperation>,
    pub reduction: Option<&'static super::FusionReductionDeclaration>,
    pub host_resources: Option<super::FusionHostResourceCallback>,
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
    let implementation =
        crate::builtin::with_runtime_builtin_bindings_by_name(identity.builtin.name, |bindings| {
            implementation_owner(identity, bindings)
        })?;
    let registration = with_registrations(implementation.builtin_path, |registrations| {
        let mut candidates = registrations
            .iter()
            .copied()
            .filter(|spec| match spec.semantics {
                Some(semantics) => semantics.binding == identity,
                None => entry.bindings.len() == 1,
            });
        let registration = candidates.next()?;
        candidates.next().is_none().then_some(registration)
    })?;
    if registration.spec.name != identity.builtin.name {
        return None;
    }
    if registration.semantics.is_some_and(|semantics| {
        semantics.required_real_unary.is_some()
            && (semantics.admit.is_none() || registration.spec.elementwise.is_none())
    }) {
        return None;
    }
    // A structural operation needs explicit binding authority and must not
    // compete with an unrelated scalar/reduction execution declaration.
    if registration.structural.is_some()
        && (registration.semantics.is_none()
            || registration.spec.elementwise.is_some()
            || registration.spec.reduction.is_some()
            || registration.reduction.is_some())
    {
        return None;
    }
    Some(CanonicalFusionDeclaration {
        binding: identity,
        spec: registration.spec,
        semantics: registration.semantics,
        structural: registration.structural,
        epilogue: registration.epilogue,
        reduction: registration.reduction,
        host_resources: registration.host_resources,
    })
}

fn implementation_owner(
    identity: BuiltinBindingIdentity,
    bindings: &[crate::builtin::RuntimeBuiltinBinding],
) -> Option<&'static runmat_builtins::BuiltinImplementationProvenance> {
    let mut selected = bindings
        .iter()
        .filter(|binding| binding.identity == identity);
    let binding = selected.next()?;
    if selected.next().is_some() {
        return None;
    }
    binding.physical_provenance()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_owner_is_the_exact_local_binding_and_retains_duplicate_rejection() {
        crate::builtins::wasm_registry::register_all();
        let binding = crate::builtin::runtime_builtin_bindings_by_name("sin")[0];
        let identity = binding.identity;
        let source = implementation_owner(identity, &[binding]).expect("declaration-owned source");
        assert_eq!(source.binding_variant, Some(identity.variant));
        assert_eq!(source.name, identity.builtin.name);
        assert!(canonical_fusion_declaration(identity).is_some());
        assert!(implementation_owner(identity, &[binding, binding]).is_none());
        let mut detached = binding;
        detached.provenance = None;
        assert!(implementation_owner(identity, &[detached]).is_none());
        detached = binding;
        detached.identity.variant = "foreign";
        assert!(implementation_owner(detached.identity, &[detached]).is_none());
    }
}
