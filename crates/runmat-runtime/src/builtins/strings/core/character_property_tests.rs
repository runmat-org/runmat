use super::{isletter::isletter_builtin, isspace::isspace_builtin, isstrprop::isstrprop_builtin};
use runmat_builtins::{
    ISLETTER_ERROR_TOO_MANY_OUTPUTS, ISSPACE_ERROR_INVALID_STRING_ARRAY,
    ISSPACE_ERROR_TOO_MANY_OUTPUTS, ISSTRPROP_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_value::{CharArray, IntValue, IntegerStorage, LogicalArray, StringArray, Tensor, Value};

fn evaluate<T>(future: impl std::future::Future<Output = T>) -> T {
    futures::executor::block_on(future)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn string_array_carriers_have_distinct_public_class_shape_and_error_contracts() {
    let singleton = Value::StringArray(StringArray::new(vec!["a b".into()], vec![1, 1]).unwrap());
    assert_eq!(
        evaluate(isletter_builtin(singleton.clone())).unwrap(),
        Value::Bool(false),
        "isletter does not unwrap even a one-element StringArray"
    );
    assert_eq!(
        evaluate(isspace_builtin(singleton)).unwrap(),
        Value::LogicalArray(LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap())
    );

    let multiple =
        Value::StringArray(StringArray::new(vec![" ".into(), "x".into()], vec![1, 2]).unwrap());
    assert_eq!(
        evaluate(isletter_builtin(multiple.clone())).unwrap(),
        Value::Bool(false)
    );
    let error = evaluate(isspace_builtin(multiple)).expect_err("nonscalar string array");
    assert_eq!(
        error.identifier(),
        ISSPACE_ERROR_INVALID_STRING_ARRAY.identifier
    );
    assert_eq!(error.message(), ISSPACE_ERROR_INVALID_STRING_ARRAY.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_string_array_has_declared_id_in_both_compatibility_modes() {
    for enabled in [false, true] {
        let _mode = crate::compatibility::push_runmat_extensions_enabled(enabled);
        let multiple =
            Value::StringArray(StringArray::new(vec![" ".into(), "x".into()], vec![1, 2]).unwrap());
        let error = evaluate(isspace_builtin(multiple)).expect_err("nonscalar string array");
        assert_eq!(
            error.identifier(),
            ISSPACE_ERROR_INVALID_STRING_ARRAY.identifier
        );
        assert_eq!(error.message(), ISSPACE_ERROR_INVALID_STRING_ARRAY.message);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn text_unicode_numeric_and_empty_shapes_remain_distinct() {
    assert_eq!(
        evaluate(isletter_builtin(Value::String("é".into()))).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        evaluate(isspace_builtin(Value::String("\u{2003}".into()))).unwrap(),
        Value::Bool(true)
    );
    let character =
        Value::CharArray(CharArray::from_unicode_scalars(vec!['a', ' ', '1', 'B'], 2, 2).unwrap());
    assert_eq!(
        evaluate(isletter_builtin(character.clone())).unwrap(),
        Value::LogicalArray(LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap())
    );
    assert_eq!(
        evaluate(isspace_builtin(character)).unwrap(),
        Value::LogicalArray(LogicalArray::new(vec![0, 1, 0, 0], vec![2, 2]).unwrap())
    );
    assert_eq!(
        evaluate(isletter_builtin(Value::CharArray(CharArray::new_row("")))).unwrap(),
        Value::LogicalArray(LogicalArray::new(vec![], vec![1, 0]).unwrap())
    );
    assert_eq!(
        evaluate(isspace_builtin(Value::CharArray(CharArray::new_row("")))).unwrap(),
        Value::LogicalArray(LogicalArray::new(vec![], vec![1, 0]).unwrap())
    );
    assert_eq!(
        evaluate(isletter_builtin(Value::Int(IntValue::U16(u16::from(b'A'))))).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        evaluate(isspace_builtin(Value::Int(IntValue::U16(u16::from(b' '))))).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        evaluate(isstrprop_builtin(
            None,
            Value::Int(IntValue::U16(u16::from(b'A'))),
            Value::String("alpha".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
        Value::Bool(true)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn output_count_errors_precede_classifier_input_errors() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = evaluate(isletter_builtin(Value::Int(IntValue::U8(65))))
        .expect_err("excess outputs precede nontext scalar-false route");
    assert_eq!(
        error.identifier(),
        ISLETTER_ERROR_TOO_MANY_OUTPUTS.identifier
    );
    let invalid_array =
        Value::StringArray(StringArray::new(vec![" ".into(), "x".into()], vec![1, 2]).unwrap());
    let error = evaluate(isspace_builtin(invalid_array))
        .expect_err("excess outputs precede invalid string array");
    assert_eq!(
        error.identifier(),
        ISSPACE_ERROR_TOO_MANY_OUTPUTS.identifier
    );
    let error = evaluate(isstrprop_builtin(
        None,
        Value::String("A".into()),
        Value::String("unknown-category".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("bad-option".into()),
            Value::Num(2.0),
        ]),
    ))
    .expect_err("excess outputs precede invalid category/option");
    assert_eq!(
        error.identifier(),
        ISSTRPROP_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn all_eight_integer_text_classes_are_exact_only_for_isstrprop() {
    for storage in [
        IntegerStorage::I8(vec![65, 49]),
        IntegerStorage::I16(vec![65, 49]),
        IntegerStorage::I32(vec![65, 49]),
        IntegerStorage::I64(vec![65, 49]),
        IntegerStorage::U8(vec![65, 49]),
        IntegerStorage::U16(vec![65, 49]),
        IntegerStorage::U32(vec![65, 49]),
        IntegerStorage::U64(vec![65, 49]),
    ] {
        let tensor = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
        assert_eq!(
            evaluate(isletter_builtin(tensor.clone())).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            evaluate(isspace_builtin(tensor.clone())).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            evaluate(isstrprop_builtin(
                None,
                tensor,
                Value::String("alpha".into()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn resident_numeric_is_false_without_download_for_two_and_exact_gather_for_isstrprop() {
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U16(&[u16::from(b'A'), u16::from(b'1')]),
                shape: &[1, 2],
            })
            .expect("resident exact character-code input");
        let weak = handle.allocation_lease().downgrade();
        let descriptor = handle.wire_descriptor();
        let before = provider.telemetry_snapshot().download_bytes;
        for output in [
            evaluate(isletter_builtin(Value::GpuTensor(handle.clone()))).unwrap(),
            evaluate(isspace_builtin(Value::GpuTensor(handle.clone()))).unwrap(),
        ] {
            assert_eq!(output, Value::Bool(false));
            assert_eq!(provider.telemetry_snapshot().download_bytes, before);
            assert_eq!(handle.wire_descriptor(), descriptor);
            assert!(weak.acquire().is_some());
        }
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(
            evaluate(isstrprop_builtin(
                None,
                Value::GpuTensor(handle.clone()),
                Value::String("alpha".into()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, before + 4);
        assert_eq!(handle.wire_descriptor(), descriptor);
        assert!(weak.acquire().is_some());
        drop(handle);
        assert!(weak.acquire().is_none());
    });
}
