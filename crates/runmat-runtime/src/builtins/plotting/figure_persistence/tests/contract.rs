use super::*;
use runmat_builtins::{
    BuiltinIntegerAuditKind, HGLOAD_INTEGER_AUDIT, HGSAVE_INTEGER_AUDIT, OPENFIG_INTEGER_AUDIT,
    SAVEAS_DESCRIPTOR,
};

#[test]
fn saveas_descriptor_covers_core_forms() {
    let labels = SAVEAS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"ok = saveas(fig, filename)"));
    assert!(labels.contains(&"ok = saveas(fig, filename, formattype)"));
}

#[test]
fn figure_load_and_save_forms_are_explicitly_integer_inapplicable() {
    for audit in [
        OPENFIG_INTEGER_AUDIT,
        HGSAVE_INTEGER_AUDIT,
        HGLOAD_INTEGER_AUDIT,
    ] {
        assert_eq!(audit.kind, BuiltinIntegerAuditKind::NotApplicable);
        assert!(audit.canonical_builtin.is_none());
    }
    for name in [OPENFIG, HGSAVE, HGLOAD] {
        let builtin = runmat_builtins::builtin_function_by_name(name)
            .expect("registered graphics persistence builtin");
        assert_eq!(
            builtin.integer_audit.map(|audit| audit.kind),
            Some(BuiltinIntegerAuditKind::NotApplicable),
            "{name}"
        );
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
    }
}

#[test]
fn hgload_and_hgsave_reject_resident_numeric_arguments_without_provider_access() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    assert!(block_on(hgload_builtin(vec![resident.clone()])).is_err());
    assert!(block_on(hgsave_builtin(vec![resident, Value::from("unused.fig")])).is_err());
}

#[test]
fn openfig_rejects_resident_filename_without_provider_access() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let error = block_on(openfig_builtin(vec![resident])).expect_err("resident filename");

    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn hgsave_rejects_typed_integer_handle_spellings() {
    for value in [
        Value::Int(runmat_value::IntValue::U64(1)),
        Value::Tensor(
            Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![1]), vec![1, 1]).unwrap(),
        ),
    ] {
        let error = block_on(hgsave_builtin(vec![value, Value::from("unused.fig")]))
            .expect_err("typed integer handles are outside hgsave's contract");
        assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }
}
