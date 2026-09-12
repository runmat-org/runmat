use super::contract::CLOSE_VARIADIC_TARGETS_EXTENSION;
use super::*;
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
        let err = futures::executor::block_on(close_builtin(vec![value]))
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
    let err = futures::executor::block_on(close_builtin(vec![
        Value::String("clients".into()),
        Value::String("servers".into()),
    ]))
    .expect_err("RunMat-only variadic close targets");
    assert_eq!(
        err.identifier(),
        CLOSE_VARIADIC_TARGETS_EXTENSION.error_identifier
    );
}

#[test]
fn resident_numeric_figure_target_rejects_without_provider_dispatch() {
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    });
    let err = futures::executor::block_on(close_builtin(vec![resident]))
        .expect_err("resident figure target");
    assert!(!err.message().to_ascii_lowercase().contains("provider"));
}

#[test]
fn invalid_network_structure_uses_canonical_invalid_handle_error() {
    let invalid = Value::Struct(runmat_value::StructValue::new());
    let err = futures::executor::block_on(close_builtin(vec![invalid]))
        .expect_err("invalid networking handle");
    assert_eq!(err.identifier(), Some("RunMat:close:InvalidHandle"));
    assert!(CLOSE_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == Some("RunMat:close:InvalidHandle")));
}
