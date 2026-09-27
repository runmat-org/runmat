use super::{builtin_fusion_spec_registrations, BuiltinFusionSpec};
use runmat_builtins::{
    builtin_catalog_entry_by_name, builtin_implementation_provenance, BuiltinBindingIdentity,
    BuiltinImplementationAuthority,
};

/// Resolve a fusion declaration for an already selected canonical binding.
///
/// Current spec declarations identify a builtin, not a binding variant. They
/// therefore authorize only an unambiguous, single-binding catalog entry. A
/// future multi-variant owner must declare variant-specific fusion semantics
/// before it can use this boundary. Duplicate declarations fail closed.
/// The selected binding's compiled implementation provenance is the join key.
/// A spec's textual name is checked for consistency, never used to select an
/// implementation. Whole-registry validation remains the inventory's owner.
pub fn canonical_fusion_spec(
    identity: BuiltinBindingIdentity,
) -> Option<&'static BuiltinFusionSpec> {
    let entry = builtin_catalog_entry_by_name(identity.builtin.name)?;
    let [binding] = entry.bindings else {
        return None;
    };
    if entry.binding_identity(binding) != identity {
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
    let mut candidates = builtin_fusion_spec_registrations()
        .filter(|spec| spec.builtin_path == implementation.builtin_path);
    let registration = candidates.next()?;
    if candidates.next().is_some() || registration.spec.name != identity.builtin.name {
        return None;
    }
    Some(registration.spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_binding_cannot_borrow_another_variants_expression() {
        let entry = builtin_catalog_entry_by_name("single").expect("single catalog entry");
        let identity = entry.binding_identity(&entry.bindings[0]);
        assert!(canonical_fusion_spec(identity).is_some());
        assert!(canonical_fusion_spec(BuiltinBindingIdentity {
            variant: "not-the-selected-binding",
            ..identity
        })
        .is_none());
    }
}
