use super::{builtin_fusion_specs, BuiltinFusionSpec};
use runmat_builtins::{builtin_catalog_entry_by_name, BuiltinBindingIdentity};

/// Resolve a fusion declaration for an already selected canonical binding.
///
/// Current spec declarations identify a builtin, not a binding variant. They
/// therefore authorize only an unambiguous, single-binding catalog entry. A
/// future multi-variant owner must declare variant-specific fusion semantics
/// before it can use this boundary. Duplicate declarations fail closed.
/// Physical declaration provenance remains checked by the migration inventory;
/// this lookup is not a replacement for that whole-registry validation.
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
    let mut candidates = builtin_fusion_specs().filter(|spec| spec.name == identity.builtin.name);
    let spec = candidates.next()?;
    candidates.next().is_none().then_some(spec)
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
