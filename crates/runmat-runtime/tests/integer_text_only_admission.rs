use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
};
use runmat_value::{IntValue, Value};

const PACKET: [&str; 8] = [
    "normalizeWords",
    "readWordEmbedding",
    "regexp",
    "regexpi",
    "replace",
    "rethrow",
    "splitlines",
    "strip",
];

fn args_for(name: &str, first: Value) -> Vec<Value> {
    match name {
        "regexp" | "regexpi" => vec![first, Value::String("x".into())],
        "replace" => vec![
            first,
            Value::String("old".into()),
            Value::String("new".into()),
        ],
        _ => vec![first],
    }
}

fn unregistered_resident_value() -> Value {
    let allocation = TestAllocationOwner::new()
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(!runmat_accelerate_api::registered_providers()
        .iter()
        .any(|provider| provider.device_id() == allocation.handle.device_id()));
    Value::GpuTensor(allocation.handle)
}

#[test]
fn text_only_packet_registers_explicit_integer_inapplicability() {
    for name in PACKET {
        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert_eq!(
            builtin.integer_audit.expect("integer audit").kind,
            runmat_builtins::BuiltinIntegerAuditKind::NotApplicable,
            "{name}"
        );
    }
}

#[test]
fn text_only_packet_rejects_all_integer_classes_without_conversion() {
    let values = [
        IntValue::I8(1),
        IntValue::I16(1),
        IntValue::I32(1),
        IntValue::I64(1),
        IntValue::U8(1),
        IntValue::U16(1),
        IntValue::U32(1),
        IntValue::U64(u64::MAX),
    ];
    for name in PACKET {
        for value in values.iter().cloned() {
            let error = block_on(runmat_runtime::call_builtin_async(
                name,
                &args_for(name, Value::Int(value)),
            ))
            .expect_err("integer input must reject");
            assert!(!error.message().is_empty(), "{name}");
        }
    }
}

#[test]
fn text_only_packet_rejects_unregistered_residency_before_provider_access() {
    for name in PACKET {
        let resident = unregistered_resident_value();
        let Value::GpuTensor(caller) = resident.clone() else {
            unreachable!()
        };
        let descriptor = caller.wire_descriptor();
        let physical = caller.allocation_lease().downgrade();
        let error = block_on(runmat_runtime::call_builtin_async(
            name,
            &args_for(name, resident),
        ))
        .expect_err("resident numeric input must reject");
        assert!(
            !error.message().to_ascii_lowercase().contains("provider"),
            "{name}: {}",
            error.message()
        );
        assert_eq!(caller.wire_descriptor(), descriptor);
        assert!(physical.acquire().is_some());
        drop(caller);
        assert!(physical.acquire().is_none());
    }
}
