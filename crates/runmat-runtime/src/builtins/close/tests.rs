use super::*;
use runmat_builtins::CLOSE_VARIADIC_TARGETS_EXTENSION;
use runmat_builtins::{BuiltinIntegerBackendRule, BuiltinIntegerComputationDomain};
use runmat_value::{IntegerStorage, Tensor, Value};

#[test]
fn compatibility_mode_rejects_all_integer_figure_number_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];

    for storage in storages {
        let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
        let err = futures::executor::block_on(close_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![value]),
        ))
        .expect_err("typed integer extension must be gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:CloseIntegerFigureNumberExtension")
        );
    }
}

#[test]
fn close_integer_capability_is_structural_and_host_only() {
    let capability = &CLOSE_INTEGER_CAPABILITIES[0];
    assert_eq!(capability.inputs[0].classes.len(), 8);
    assert_eq!(
        capability.computation_domain,
        BuiltinIntegerComputationDomain::Structural
    );
    assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
}

#[test]
fn compatibility_mode_rejects_separate_variadic_targets_before_dispatch() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = futures::executor::block_on(close_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("clients".into()),
            Value::String("servers".into()),
        ]),
    ))
    .expect_err("RunMat-only variadic close targets");
    assert_eq!(
        err.identifier(),
        CLOSE_VARIADIC_TARGETS_EXTENSION.error_identifier
    );
}

#[test]
fn resident_numeric_figure_target_rejects_without_provider_dispatch() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident numeric target");
    let caller_alias = allocation.handle.clone();
    let resident = Value::GpuTensor(allocation.handle);
    let err = futures::executor::block_on(close_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![resident]),
    ))
    .expect_err("resident figure target");
    assert!(!err.message().to_ascii_lowercase().contains("provider"));
    assert_eq!(allocation.probe.retirements(), 0);
    caller_alias.allocation_lease().validate_live().unwrap();
    drop(err);
    drop(caller_alias);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn invalid_network_structure_uses_canonical_invalid_handle_error() {
    let invalid = Value::Struct(runmat_value::StructValue::new());
    let err = futures::executor::block_on(close_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![invalid]),
    ))
    .expect_err("invalid networking handle");
    assert_eq!(err.identifier(), Some("RunMat:close:InvalidHandle"));
    assert!(CLOSE_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == Some("RunMat:close:InvalidHandle")));
}
