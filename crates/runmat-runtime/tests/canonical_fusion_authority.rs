//! A single runtime library owns the compiled implementation inventory.
use runmat_builtins::{builtin_catalog_entry_by_name, BuiltinBindingIdentity};
use runmat_runtime::builtins::common::spec::canonical_fusion_declaration;

#[test]
fn nested_specification_module_joins_the_implementation_binding() {
    let entry = builtin_catalog_entry_by_name("heaviside").unwrap();
    let identity = entry.binding_identity(&entry.bindings[0]);
    let declaration =
        canonical_fusion_declaration(identity).expect("owning implementation declaration");
    assert_eq!(declaration.binding, identity);
    assert!(declaration.spec.elementwise.is_some());
}

#[test]
fn selected_binding_cannot_borrow_another_variants_expression() {
    for (name, revision) in [("single", 2), ("double", 2), ("power", 6)] {
        let entry = builtin_catalog_entry_by_name(name).expect("canonical entry");
        let identity = entry.binding_identity(&entry.bindings[0]);
        let declaration = canonical_fusion_declaration(identity).expect("selected declaration");
        assert_eq!(declaration.binding, identity);
        let semantics = declaration.semantics.expect("typed semantics");
        assert_eq!(semantics.binding, identity);
        assert_eq!(semantics.revision, revision);
        assert!(canonical_fusion_declaration(BuiltinBindingIdentity {
            variant: "not-the-selected-binding",
            ..identity
        })
        .is_none());
    }
}

#[test]
fn iir_fusion_boundaries_have_no_legacy_declaration() {
    for name in ["filter", "filtfilt"] {
        let entry = builtin_catalog_entry_by_name(name).expect("canonical IIR entry");
        assert_eq!(
            entry.placement.fusion,
            runmat_builtins::BuiltinFusionPolicy::Boundary
        );
        let identity = entry.binding_identity(&entry.bindings[0]);
        assert!(canonical_fusion_declaration(identity).is_none());
    }
}

#[test]
fn admission_work_identity_distinguishes_operand_and_provider_precision() {
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_runtime::builtins::common::spec::FusionAdmissionWork;
    let entry = builtin_catalog_entry_by_name("power").unwrap();
    let work = |provider_precision| FusionAdmissionWork::ProviderProbe {
        binding: entry.binding_identity(&entry.bindings[0]),
        precision: ProviderPrecision::F32,
        provider_precision,
        inputs: Vec::new(),
    };
    assert_ne!(work(ProviderPrecision::F32), work(ProviderPrecision::F64));
    assert_ne!(
        serde_json::to_vec(&work(ProviderPrecision::F32)).unwrap(),
        serde_json::to_vec(&work(ProviderPrecision::F64)).unwrap()
    );
}
