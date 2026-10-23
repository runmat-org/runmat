use super::*;
use runmat_builtins::ISHANDLE_DESCRIPTOR;

#[test]
fn ishandle_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = ISHANDLE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"tf = ishandle(h)"));
}

#[test]
fn ishandle_vectorizes_numeric_handle_arrays() {
    let handles = runmat_value::Tensor::new(vec![f64::NAN, -1.0], vec![2, 1]).unwrap();
    let result = invoke(vec![Value::Tensor(handles)]).unwrap();
    match result {
        Value::LogicalArray(logical) => {
            assert_eq!(logical.shape, vec![2, 1]);
            assert_eq!(logical.data, vec![0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ishandle_numeric_tensor_masks_keep_low_rank_shapes() {
    for (data, shape) in [(vec![-1.0], Vec::new()), (vec![-1.0, -2.0], vec![2])] {
        let input = runmat_value::Tensor::new(data.clone(), shape.clone()).unwrap();
        let Value::LogicalArray(output) = invoke(vec![Value::Tensor(input)]).unwrap() else {
            panic!("numeric tensor input must return a logical array")
        };
        assert_eq!(output.shape, shape);
        assert_eq!(output.data, vec![0; data.len()]);
    }
}

#[test]
fn ishandle_vectorizes_typed_handles_without_a_floating_mirror() {
    let handles = runmat_value::Tensor::new_integer(
        runmat_value::IntegerStorage::I16(vec![-1, -2]),
        vec![2, 1],
    )
    .unwrap();
    let result = invoke(vec![Value::Tensor(handles)]).unwrap();
    match result {
        Value::LogicalArray(logical) => assert_eq!(logical.data, vec![0, 0]),
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[test]
fn compatibility_mode_treats_integer_arrays_as_nonobjects() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let handles = runmat_value::Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![u64::MAX]),
        vec![1, 1],
    )
    .unwrap();
    let Value::LogicalArray(result) = invoke(vec![Value::Tensor(handles)]).unwrap() else {
        panic!("expected logical array");
    };
    assert_eq!(result.data, vec![0]);
}

#[test]
fn ishandle_requires_exactly_one_argument() {
    assert!(invoke(Vec::new()).is_err());
    assert!(invoke(vec![Value::Num(0.0), Value::Num(1.0)]).is_err());
    assert_eq!(invoke(vec![Value::Num(0.0)]).unwrap(), Value::Bool(true));
}
fn invoke(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ishandle_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}
