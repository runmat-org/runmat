use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{IntValue, Value};

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

fn unowned_resident_value() -> Value {
    Value::GpuTensor(GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    })
}

#[test]
fn text_only_packet_registers_explicit_integer_inapplicability() {
    for name in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
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
fn text_only_packet_rejects_unowned_residency_before_provider_access() {
    for name in PACKET {
        let error = block_on(runmat_runtime::call_builtin_async(
            name,
            &args_for(name, unowned_resident_value()),
        ))
        .expect_err("resident numeric input must reject");
        assert!(
            !error.message().to_ascii_lowercase().contains("provider"),
            "{name}: {}",
            error.message()
        );
    }
}
