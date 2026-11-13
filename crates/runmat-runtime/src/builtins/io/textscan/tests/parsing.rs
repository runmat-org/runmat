use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn one_long_field_yields_through_the_invocation_budget() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use runmat_execution::cancellation::CancellationSource;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::task::{Context, Poll};

    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4096).expect("quantum")),
        source.token(),
        || Box::pin(std::future::pending::<Result<(), ExecutionServiceError>>()),
    );
    let mut cooperation = TextscanCooperation::with_budget(budget);
    let field = "x".repeat(8192);
    let format = parse_format_spec("%s").expect("format");
    let options = TextscanOptions::default();
    let mut call = Box::pin(parse_textscan(
        &field,
        &format,
        None,
        &options,
        &mut cooperation,
    ));
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(
        Future::poll(call.as_mut(), &mut cx),
        Poll::Pending
    ));
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_descriptor_covers_core_forms() {
    let labels: Vec<&str> = TEXTSCAN_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"C = textscan(textOrFileID, formatSpec)"));
    assert!(labels.contains(&"C = textscan(textOrFileID, formatSpec, args...)"));
}

#[test]
fn textscan_scalar_parsers_read_typed_integer_storage_exactly() {
    let header =
        Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("typed header lines");
    assert_eq!(
        nonnegative_usize(&Value::Tensor(header), "HeaderLines").unwrap(),
        2
    );

    let invalid = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1])
        .expect("typed invalid count");
    assert!(nonnegative_usize(&Value::Tensor(invalid), "HeaderLines").is_err());
    assert_eq!(
        nonnegative_usize(&Value::Int(IntValue::U64(u64::MAX)), "HeaderLines").ok(),
        usize::try_from(u64::MAX).ok()
    );
    assert!(nonnegative_usize(&Value::Num(1.0e300), "HeaderLines").is_err());

    let fid = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("typed fid");
    assert_eq!(numeric_fid(&Value::Tensor(fid)), Some(7));
    assert_eq!(numeric_fid(&Value::Int(IntValue::U32(7))), Some(7));
    assert_eq!(numeric_fid(&Value::Int(IntValue::U64(u64::MAX))), None);

    let fid_too_large =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("typed fid");
    assert_eq!(numeric_fid(&Value::Tensor(fid_too_large)), None);
}

#[test]
fn textscan_huge_header_lines_stop_at_eof() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(textscan_builtin(
        None,
        Value::from("x"),
        Value::from("%s"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("HeaderLines"),
            Value::Int(IntValue::U64(usize::MAX as u64)),
        ]),
    ))
    .expect("textscan");
    assert!(text_column(&out, 0).is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_reads_mixed_columns_from_text() {
    let out = block_on(textscan_builtin(
        None,
        Value::from("1 alpha\n2 beta\n"),
        Value::from("%f %s"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("textscan");
    assert_eq!(numeric_column(&out, 0), vec![1.0, 2.0]);
    assert_eq!(
        text_column(&out, 1),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn textscan_integer_conversions_preserve_native_classes_and_wide_values() {
    let out = block_on(textscan_builtin(
        None,
        Value::from("-999 18446744073709551615 1.25\n127 9223372036854775808 -2.5\n"),
        Value::from("%d8 %u64 %f32"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("textscan");
    let Value::Tensor(signed) = output_value(&out, 0) else {
        panic!("expected signed tensor");
    };
    assert_eq!(
        signed.integer_storage(),
        Some(&IntegerStorage::I8(vec![i8::MIN, i8::MAX]))
    );
    let Value::Tensor(unsigned) = output_value(&out, 1) else {
        panic!("expected unsigned tensor");
    };
    assert_eq!(
        unsigned.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
    );
    let Value::Tensor(single) = output_value(&out, 2) else {
        panic!("expected single tensor");
    };
    assert_eq!(single.numeric_dtype(), NumericDType::F32);
    assert_eq!(single.materialize_f64(), vec![1.25, -2.5]);
}

#[test]
fn textscan_hex_binary_and_overflow_conversions_preserve_integer_contracts() {
    let out = block_on(textscan_builtin(
        None,
        Value::from("0xFFFFFFFFFFFFFFFF -0x81 0b1111111111111111 999999999999999999999999999999999999999 -7\n"),
        Value::from("%x %xs8 %bu16 %d64 %u8"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("textscan");

    let Value::Tensor(hex) = output_value(&out, 0) else {
        panic!("expected hexadecimal tensor");
    };
    assert_eq!(
        hex.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX]))
    );
    let Value::Tensor(signed_hex) = output_value(&out, 1) else {
        panic!("expected signed hexadecimal tensor");
    };
    assert_eq!(
        signed_hex.integer_storage(),
        Some(&IntegerStorage::I8(vec![i8::MIN]))
    );
    let Value::Tensor(binary) = output_value(&out, 2) else {
        panic!("expected binary tensor");
    };
    assert_eq!(
        binary.integer_storage(),
        Some(&IntegerStorage::U16(vec![u16::MAX]))
    );
    let Value::Tensor(decimal) = output_value(&out, 3) else {
        panic!("expected signed decimal tensor");
    };
    assert_eq!(
        decimal.integer_storage(),
        Some(&IntegerStorage::I64(vec![i64::MAX]))
    );
    let Value::Tensor(unsigned) = output_value(&out, 4) else {
        panic!("expected unsigned decimal tensor");
    };
    assert_eq!(
        unsigned.integer_storage(),
        Some(&IntegerStorage::U8(vec![0]))
    );
}

#[test]
fn textscan_documented_integer_formats_remain_enabled_in_matlab_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let out = block_on(textscan_builtin(
        None,
        Value::from("9007199254740993 0xFFFFFFFFFFFFFFFF\n"),
        Value::from("%u64 %x"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("documented integer textscan formats");
    let Value::Tensor(decimal) = output_value(&out, 0) else {
        panic!("expected decimal integer tensor");
    };
    assert_eq!(
        decimal.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993]))
    );
    let Value::Tensor(hex) = output_value(&out, 1) else {
        panic!("expected hexadecimal integer tensor");
    };
    assert_eq!(
        hex.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX]))
    );
}

#[test]
fn textscan_compatibility_gates_typed_controls_before_scanning() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(textscan_builtin(
        None,
        Value::from("1\n"),
        Value::from("%d"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U8(1))]),
    ))
    .expect_err("typed repeat count must be gated");
    assert_eq!(
        error.identifier(),
        TEXTSCAN_TYPED_INTEGER_CONTROL_EXTENSION.error_identifier
    );
}

#[test]
fn textscan_collect_output_groups_only_adjacent_same_class_columns() {
    let out = block_on(textscan_builtin(
        None,
        Value::from("1 2 3\n4 5 6\n"),
        Value::from("%d16 %d16 %u16"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("CollectOutput"),
            Value::Bool(true),
        ]),
    ))
    .expect("textscan");
    assert_eq!(output_cell(&out).cols, 2);
    let Value::Tensor(signed) = output_value(&out, 0) else {
        panic!("expected collected signed tensor");
    };
    assert_eq!(signed.shape(), &[2, 2]);
    assert_eq!(
        signed.integer_storage(),
        Some(&IntegerStorage::I16(vec![1, 4, 2, 5]))
    );
    let Value::Tensor(unsigned) = output_value(&out, 1) else {
        panic!("expected unsigned tensor");
    };
    assert_eq!(
        unsigned.integer_storage(),
        Some(&IntegerStorage::U16(vec![3, 6]))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_honors_delimiter_header_comments_and_treat_empty() {
    let text = "# header\nA,1.5\nB,NA\n% ignored\nC,3.5\n";
    let out = block_on(textscan_builtin(
        None,
        Value::from(text),
        Value::from("%s %f"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Delimiter"),
            Value::from(","),
            Value::from("HeaderLines"),
            Value::Num(1.0),
            Value::from("CommentStyle"),
            Value::from("%"),
            Value::from("TreatAsEmpty"),
            Value::from("NA"),
        ]),
    ))
    .expect("textscan");
    assert_eq!(
        text_column(&out, 0),
        vec!["A".to_string(), "B".to_string(), "C".to_string()]
    );
    let nums = numeric_column(&out, 1);
    assert_eq!(nums[0], 1.5);
    assert!(nums[1].is_nan());
    assert_eq!(nums[2], 3.5);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_supports_repeat_skip_collect_and_quotes() {
    let out = block_on(textscan_builtin(
        None,
        Value::from("1,drop,2,\"hello, world\"\n3,drop,4,\"tail\"\n"),
        Value::from("%f %*s %f %q"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::from("Delimiter"),
            Value::from(","),
            Value::from("CollectOutput"),
            Value::Bool(true),
        ]),
    ))
    .expect("textscan");
    let Value::Tensor(group) = output_value(&out, 0) else {
        panic!("expected collected numeric group");
    };
    assert_eq!(group.shape(), &[1, 2]);
    assert_eq!(group.materialize_f64(), vec![1.0, 2.0]);
    assert_eq!(text_column(&out, 1), vec!["hello, world".to_string()]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_return_on_error_false_reports_parse_failure() {
    let err = block_on(textscan_builtin(
        None,
        Value::from("1\nbad\n"),
        Value::from("%f"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("ReturnOnError"),
            Value::Bool(false),
        ]),
    ))
    .expect_err("parse failure");
    assert!(err.message().contains("cannot parse"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_honors_literals_in_format_spec() {
    let out = block_on(textscan_builtin(
        None,
        Value::from("1,2\n3,4\n"),
        Value::from("%f,%f"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("textscan");
    assert_eq!(numeric_column(&out, 0), vec![1.0, 3.0]);
    assert_eq!(numeric_column(&out, 1), vec![2.0, 4.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_width_and_char_conversions_leave_remainder() {
    let strings = block_on(textscan_builtin(
        None,
        Value::from("abcdef"),
        Value::from("%2s%s"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("textscan strings");
    assert_eq!(text_column(&strings, 0), vec!["ab".to_string()]);
    assert_eq!(text_column(&strings, 1), vec!["cdef".to_string()]);

    let chars = block_on(textscan_builtin(
        None,
        Value::from("abc"),
        Value::from("%c%c%c"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("textscan chars");
    assert_eq!(text_column(&chars, 0), vec!["a".to_string()]);
    assert_eq!(text_column(&chars, 1), vec!["b".to_string()]);
    assert_eq!(text_column(&chars, 2), vec!["c".to_string()]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_collect_output_preserves_column_major_numeric_group() {
    let out = block_on(textscan_builtin(
        None,
        Value::from("1 2\n3 4\n"),
        Value::from("%f %f"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("CollectOutput"),
            Value::Bool(true),
        ]),
    ))
    .expect("textscan");
    let Value::Tensor(group) = output_value(&out, 0) else {
        panic!("expected collected numeric group");
    };
    assert_eq!(group.shape(), &[2, 2]);
    assert_eq!(group.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
}
