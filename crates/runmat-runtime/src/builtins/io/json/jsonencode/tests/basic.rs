use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prepared_option_identity_is_not_reparsed_after_materialization() {
    let original = vec![Value::from("pReTtYpRiNt"), Value::Bool(true)];
    let layout = prepare_option_layout(crate::call::FunctionArgumentView::from_slice(&original))
        .expect("known public spelling");
    // A materializer preserves host text in production. Even if a later value
    // carrier changes, the admitted semantic option is the one parsed once.
    let materialized = vec![Value::from("NotAnOption"), Value::Bool(false)];
    let options = parse_options(
        &layout,
        crate::call::FunctionArgumentView::from_slice(&materialized),
    )
    .expect("typed option identity");
    assert!(!options.pretty_print);
    assert!(options.convert_inf_and_nan);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = JSONENCODE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"jsonText = jsonencode(value)"));
    assert!(labels.contains(&"jsonText = jsonencode(value, options)"));
    assert!(labels.contains(&"jsonText = jsonencode(value, name, optionValue)"));
    assert!(labels.contains(&"jsonText = jsonencode(value, nameValuePairs...)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_scalar_double() {
    let encoded =
        block_on(jsonencode_builtin(None, Value::Num(5.0), Vec::new())).expect("jsonencode");
    assert_eq!(as_string(encoded), "5");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_symbolic_value_as_text() {
    let expr = SymbolicExpr::div_expr(
        SymbolicExpr::function(
            runmat_value::SymbolicFunction::Sin,
            SymbolicExpr::variable("x"),
        ),
        SymbolicExpr::variable("x"),
    );
    let encoded =
        block_on(jsonencode_builtin(None, Value::Symbolic(expr), Vec::new())).expect("jsonencode");

    assert_eq!(as_string(encoded), "\"sin(x)/x\"");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_symbolic_array_preserves_matrix_shape() {
    let array = SymbolicArray::new(
        vec![
            SymbolicExpr::variable("a"),
            SymbolicExpr::variable("c"),
            SymbolicExpr::variable("b"),
            SymbolicExpr::variable("d"),
        ],
        vec![2, 2],
    )
    .expect("symbolic array");

    let encoded = block_on(jsonencode_builtin(
        None,
        Value::SymbolicArray(array),
        Vec::new(),
    ))
    .expect("jsonencode");

    assert_eq!(as_string(encoded), "[[\"a\",\"b\"],[\"c\",\"d\"]]");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_matrix_pretty_print() {
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).expect("tensor");
    let args = vec![Value::from("PrettyPrint"), Value::Bool(true)];
    let encoded =
        block_on(jsonencode_builtin(None, Value::Tensor(tensor), args)).expect("jsonencode");
    let expected = "[\n    [1,2,3],\n    [4,5,6]\n]";
    assert_eq!(as_string(encoded), expected);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_integer_tensor_preserves_exact_uint64_values() {
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
        .expect("integer tensor");

    let encoded =
        block_on(jsonencode_builtin(None, Value::Tensor(tensor), Vec::new())).expect("jsonencode");

    assert_eq!(
        as_string(encoded),
        format!("[{},{}]", u64::MAX, 1_u64 << 63)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_integer_tensor_serializes_all_eight_classes_exactly() {
    let cases = [
        (IntegerStorage::I8(vec![i8::MIN]), "-128"),
        (IntegerStorage::I16(vec![i16::MIN]), "-32768"),
        (IntegerStorage::I32(vec![i32::MIN]), "-2147483648"),
        (IntegerStorage::I64(vec![i64::MIN]), "-9223372036854775808"),
        (IntegerStorage::U8(vec![u8::MAX]), "255"),
        (IntegerStorage::U16(vec![u16::MAX]), "65535"),
        (IntegerStorage::U32(vec![u32::MAX]), "4294967295"),
        (IntegerStorage::U64(vec![u64::MAX]), "18446744073709551615"),
    ];
    for (storage, expected) in cases {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
        let encoded = block_on(jsonencode_builtin(None, Value::Tensor(tensor), Vec::new()))
            .expect("jsonencode integer tensor");
        assert_eq!(as_string(encoded), expected);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_real_integer_payload_remains_public_and_exact_in_strict_mode() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let cases = [
        (Value::Int(IntValue::I8(i8::MIN)), "-128"),
        (Value::Int(IntValue::I16(i16::MIN)), "-32768"),
        (Value::Int(IntValue::I32(i32::MIN)), "-2147483648"),
        (Value::Int(IntValue::I64(i64::MIN)), "-9223372036854775808"),
        (Value::Int(IntValue::U8(u8::MAX)), "255"),
        (Value::Int(IntValue::U16(u16::MAX)), "65535"),
        (Value::Int(IntValue::U32(u32::MAX)), "4294967295"),
        (Value::Int(IntValue::U64(u64::MAX)), "18446744073709551615"),
    ];
    for (value, expected) in cases {
        let encoded = block_on(jsonencode_builtin(None, value, Vec::new()))
            .expect("documented real integer payload");
        assert_eq!(as_string(encoded), expected);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_struct_round_trip() {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields
        .fields
        .insert("name".to_string(), Value::from("RunMat"));
    fields
        .fields
        .insert("year".to_string(), Value::Int(IntValue::I32(2025)));
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::Struct(fields.finish()),
        Vec::new(),
    ))
    .expect("jsonencode");
    assert_eq!(as_string(encoded), "{\"name\":\"RunMat\",\"year\":2025}");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_struct_array_canonicalizes_keys_and_erases_empty_schema() {
    let mut first = runmat_value::UnboundedStructValueBuilder::new();
    first.insert("ä", Value::Num(2.0));
    first.insert("zeta", Value::Num(1.0));
    first.insert("alpha", Value::Num(3.0));
    let mut second = runmat_value::UnboundedStructValueBuilder::new();
    second.insert("ä", Value::Num(5.0));
    second.insert("zeta", Value::Num(4.0));
    second.insert("alpha", Value::Num(6.0));
    let second = second.finish();
    let first = first.finish();
    let array = runmat_value::StructArray::new(vec![first, second], vec![2, 1, 1]).unwrap();
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::StructArray(array),
        Vec::new(),
    ))
    .unwrap();
    assert_eq!(
        as_string(encoded),
        "[[[{\"alpha\":3,\"zeta\":1,\"ä\":2}]],[[{\"alpha\":6,\"zeta\":4,\"ä\":5}]]]"
    );

    let empty =
        runmat_value::StructArray::empty(vec!["alpha".into(), "zeta".into()], vec![0, 2, 1])
            .unwrap();
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::StructArray(empty),
        Vec::new(),
    ))
    .unwrap();
    assert_eq!(as_string(encoded), "[]");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_struct_options_enable_pretty_print() {
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).expect("tensor");
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.fields
        .insert("PrettyPrint".to_string(), Value::Bool(true));
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::Tensor(tensor),
        vec![Value::Struct(opts.finish())],
    ))
    .expect("jsonencode");
    let expected = "[\n    [1,2],\n    [4,5]\n]";
    assert_eq!(as_string(encoded), expected);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_options_accept_scalar_tensor_bool() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor_value = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
    let args = vec![Value::from("PrettyPrint"), Value::Tensor(tensor_value)];
    let encoded = block_on(jsonencode_builtin(None, Value::Num(42.0), args)).expect("jsonencode");
    assert_eq!(as_string(encoded), "42");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_options_read_wide_uint64_storage_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let false_option =
        Tensor::new_integer(IntegerStorage::U8(vec![0]), vec![1, 1]).expect("integer tensor");
    let compact = block_on(jsonencode_builtin(
        None,
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor")),
        vec![Value::from("PrettyPrint"), Value::Tensor(false_option)],
    ))
    .expect("jsonencode");
    assert_eq!(as_string(compact), "[1,2]");

    let true_option = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
        .expect("integer tensor");
    let pretty = block_on(jsonencode_builtin(
        None,
        Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("tensor")),
        vec![Value::from("PrettyPrint"), Value::Tensor(true_option)],
    ))
    .expect("jsonencode");
    assert_eq!(as_string(pretty), "[\n    [1,2],\n    [3,4]\n]");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_option_coercions_are_independently_gated() {
    let cases = [
        (
            Value::Int(IntValue::U64(u64::MAX)),
            JSONENCODE_TYPED_INTEGER_OPTION_EXTENSION.error_identifier,
        ),
        (
            Value::Num(1.0),
            JSONENCODE_NUMERIC_OPTION_EXTENSION.error_identifier,
        ),
        (
            Value::from("true"),
            JSONENCODE_TEXT_OPTION_EXTENSION.error_identifier,
        ),
    ];
    for (option, expected_identifier) in cases {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(jsonencode_builtin(
            None,
            Value::Num(1.0),
            vec![Value::from("PrettyPrint"), option],
        ))
        .expect_err("option coercion must be gated");
        assert_eq!(error.identifier(), expected_identifier);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_documented_convert_inf_numeric_bool_remains_public() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::Num(1.0),
        vec![Value::from("ConvertInfAndNaN"), Value::Num(1.0)],
    ))
    .expect("documented numeric one");
    assert_eq!(as_string(encoded), "1");

    let error = block_on(jsonencode_builtin(
        None,
        Value::Num(f64::NAN),
        vec![Value::from("ConvertInfAndNaN"), Value::Num(0.0)],
    ))
    .expect_err("documented numeric zero disables conversion");
    assert_eq!(error.message(), JSONENCODE_ERROR_INF_NAN.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_option_errors_precede_irrelevant_extension_gates() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let unknown = block_on(jsonencode_builtin(
        None,
        Value::Num(1.0),
        vec![Value::from("Unknown"), Value::Num(1.0)],
    ))
    .expect_err("unknown name must reject before numeric coercion gate");
    assert!(unknown
        .message()
        .starts_with(JSONENCODE_ERROR_UNKNOWN_OPTION.message));

    let nonscalar = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
    let invalid = block_on(jsonencode_builtin(
        None,
        Value::Num(1.0),
        vec![Value::from("PrettyPrint"), Value::Tensor(nonscalar)],
    ))
    .expect_err("nonscalar value must reject before numeric coercion gate");
    assert_eq!(invalid.message(), JSONENCODE_ERROR_OPTION_VALUE.message);
}
