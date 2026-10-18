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
    assert!(labels.contains(&"saveas(fig, filename)"));
    assert!(labels.contains(&"saveas(fig, filename, formattype)"));
    assert!(labels.contains(&"ok = saveas(___)"));
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
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(name)
            .expect("canonical graphics persistence builtin");
        assert_eq!(
            builtin.integer_audit.map(|audit| audit.kind),
            Some(BuiltinIntegerAuditKind::NotApplicable),
            "{name}"
        );
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert!(
            crate::builtin::runtime_builtin_bindings_by_name(name)
                .iter()
                .any(|binding| binding.identity.variant == "default"),
            "{name} must have its canonical runtime binding"
        );
        assert!(
            runmat_builtins::builtin_function_by_name(name).is_none(),
            "{name} must not retain a legacy registration"
        );
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

#[test]
fn sink_status_output_is_runmat_only_and_admitted_before_file_effects() {
    let _guard = setup();
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let _one = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    for call in [SAVEAS, SAVEFIG, HGSAVE] {
        let error = match call {
            SAVEAS => block_on(saveas_builtin(vec![
                Value::Num(current_figure_handle().as_u32() as f64),
                Value::from("/never-created.fig"),
            ])),
            SAVEFIG => block_on(savefig_builtin(vec![Value::from("/never-created.fig")])),
            HGSAVE => block_on(hgsave_builtin(vec![Value::from("/never-created.fig")])),
            _ => unreachable!(),
        }
        .expect_err("status output requires RunMat compatibility");
        assert_eq!(
            error.identifier(),
            runmat_builtins::FIGURE_PERSISTENCE_STATUS_OUTPUT_EXTENSION.error_identifier
        );
    }
}

#[test]
fn canonical_catalog_placement_has_no_legacy_gpu_or_fusion_registration() {
    let gpu_names = crate::builtins::common::spec::builtin_gpu_specs()
        .map(|spec| spec.name)
        .collect::<Vec<_>>();
    assert!(!gpu_names.contains(&SAVEAS));
    assert!(!gpu_names.contains(&SAVEFIG));

    let fusion_names = crate::builtins::common::spec::builtin_fusion_specs()
        .map(|spec| spec.name)
        .collect::<Vec<_>>();
    assert!(!fusion_names.contains(&"figure_persistence"));
}
