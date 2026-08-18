#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{IntValue, Value};
use runmat_vm::Instr;
use test_helpers::{compile_source, execute_source};

#[test]
fn compiled_radix_literals_load_exact_scalar_classes() {
    let bytecode = compile_source(
        "a=0xFF; b=0x100; c=0x10000; d=0x100000000; e=0xFFs8; f=0xFFFFFFFFFFFFFFFFs64; g=0xFFFFFFFFFFFFFFFFu64;",
    )
    .expect("compile exact radix literals");
    assert!(bytecode.instructions.iter().any(
        |instruction| matches!(instruction, Instr::LoadInt(IntValue::U64(value)) if *value == u64::MAX)
    ));

    let values = test_helpers::interpret(&bytecode).expect("execute exact radix literals");
    for expected in [
        IntValue::U8(255),
        IntValue::U16(256),
        IntValue::U32(65_536),
        IntValue::U64(4_294_967_296),
        IntValue::I8(-1),
        IntValue::I64(-1),
        IntValue::U64(u64::MAX),
    ] {
        assert!(
            values.contains(&Value::Int(expected.clone())),
            "{expected:?}"
        );
    }
}

#[test]
fn compiled_radix_literal_arrays_preserve_wide_exact_storage() {
    let values = execute_source(
        "a=[0xFF000000001F123As64 0x1234FFFFFFFFFFFs64]; b=[0x20000000000001u64 0xFFFFFFFFFFFFFFFFu64];",
    )
    .expect("execute radix literal arrays");

    assert!(values.iter().any(|value| matches!(
        value,
        Value::Tensor(tensor)
            if tensor.integer_storage().is_some_and(|storage|
                storage.exact_values() == vec![
                    IntValue::I64(-72_057_594_035_891_654),
                    IntValue::I64(81_997_179_153_022_975),
                ]
            )
    )));
    assert!(values.iter().any(|value| matches!(
        value,
        Value::Tensor(tensor)
            if tensor.integer_storage().is_some_and(|storage|
                storage.exact_values() == vec![
                    IntValue::U64(9_007_199_254_740_993),
                    IntValue::U64(u64::MAX),
                ]
            )
    )));
}

#[test]
fn signed_radix_literals_use_twos_complement_at_every_width() {
    let values = execute_source("a=0xFFs8; b=0xFFFFs16; c=0xFFFFFFFFs32; d=0xFFFFFFFFFFFFFFFFs64;")
        .expect("execute signed radix literals");
    for expected in [
        IntValue::I8(-1),
        IntValue::I16(-1),
        IntValue::I32(-1),
        IntValue::I64(-1),
    ] {
        assert!(
            values.contains(&Value::Int(expected.clone())),
            "{expected:?}"
        );
    }
}

#[test]
fn explicit_radix_suffixes_preserve_all_eight_integer_classes() {
    let values = execute_source(
        "a=0x80s8; b=0x8000s16; c=0x80000000s32; d=0x8000000000000000s64; e=0xFFu8; f=0xFFFFu16; g=0xFFFFFFFFu32; h=0xFFFFFFFFFFFFFFFFu64;",
    )
    .expect("execute every explicit integer suffix");
    for expected in [
        IntValue::I8(i8::MIN),
        IntValue::I16(i16::MIN),
        IntValue::I32(i32::MIN),
        IntValue::I64(i64::MIN),
        IntValue::U8(u8::MAX),
        IntValue::U16(u16::MAX),
        IntValue::U32(u32::MAX),
        IntValue::U64(u64::MAX),
    ] {
        assert!(
            values.contains(&Value::Int(expected.clone())),
            "{expected:?}"
        );
    }
}

#[test]
fn typed_radix_literals_round_trip_through_resident_storage_exactly() {
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let provider = runmat_accelerate_api::provider().expect("in-process provider");
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source(
        "source=[0x20000000000001u64 0xFFFFFFFFFFFFFFFFu64]; resident=gpuArray(source); restored=gather(resident);",
    )
    .expect("resident typed-literal round trip");
    let expected = vec![
        IntValue::U64(9_007_199_254_740_993),
        IntValue::U64(u64::MAX),
    ];
    assert!(values.iter().any(|value| matches!(
        value,
        Value::Tensor(tensor)
            if tensor.integer_storage().is_some_and(|storage| storage.exact_values() == expected)
    )));
}

#[test]
fn typed_radix_literals_remain_exact_in_argument_defaults_and_validators() {
    let values = execute_source(
        r#"
        defaulted = with_default();
        checked = with_member(0xFFFFFFFFFFFFFFFFu64);
        function out = with_default(value)
            arguments
                value = 0x20000000000001u64
            end
            out = value;
        end
        function out = with_member(value)
            arguments
                value {mustBeMember(0xFFFFFFFFFFFFFFFFu64)}
            end
            out = value;
        end
        "#,
    )
    .expect("typed literal argument metadata");
    for expected in [
        IntValue::U64(9_007_199_254_740_993),
        IntValue::U64(u64::MAX),
    ] {
        assert!(values.contains(&Value::Int(expected)));
    }
}
