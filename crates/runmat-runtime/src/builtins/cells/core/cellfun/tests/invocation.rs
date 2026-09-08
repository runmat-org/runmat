use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn maps_builtin_over_one_cell_array() {
    let input = cell(
        vec![Value::String("Ada".into()), Value::String("Linus".into())],
        &[1, 2],
    );
    let result = call(Value::FunctionHandle("strlength".into()), vec![input]).unwrap();
    assert_eq!(tensor_values(result), vec![3.0, 5.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn maps_equal_cell_arrays_in_lockstep() {
    let left = cell(vec![Value::Num(1.0), Value::Num(2.0)], &[2, 1]);
    let right = cell(vec![Value::Num(10.0), Value::Num(20.0)], &[2, 1]);
    let result = call(Value::FunctionHandle("plus".into()), vec![left, right]).unwrap();
    assert_eq!(tensor_values(result), vec![11.0, 22.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn passes_constant_arguments_after_cell_inputs() {
    let matrices = cell(
        vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]),
            tensor(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]),
        ],
        &[1, 2],
    );
    let result = call(
        Value::FunctionHandle("size".into()),
        vec![matrices, Value::Num(2.0)],
    )
    .unwrap();
    assert_eq!(tensor_values(result), vec![2.0, 2.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nonuniform_output_preserves_each_callback_value() {
    let input = cell(
        vec![Value::String("Ada".into()), Value::String("Linus".into())],
        &[1, 2],
    );
    let result = call(
        Value::FunctionHandle("upper".into()),
        vec![
            input,
            Value::String("UniformOutput".into()),
            Value::Bool(false),
        ],
    )
    .unwrap();
    let Value::Cell(output) = result else {
        panic!("expected cell output")
    };
    assert_eq!(output.shape, vec![1, 2]);
    assert_eq!(
        output.data,
        vec![Value::String("ADA".into()), Value::String("LINUS".into())]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn empty_inputs_preserve_shape_in_both_output_modes() {
    let uniform = call(
        Value::FunctionHandle("abs".into()),
        vec![cell(Vec::new(), &[0, 3])],
    )
    .unwrap();
    let Value::Tensor(uniform) = uniform else {
        panic!("expected an empty numeric array")
    };
    assert_eq!(uniform.shape, vec![0, 3]);
    assert!(uniform.is_empty());

    let nonuniform = call(
        Value::FunctionHandle("abs".into()),
        vec![
            cell(Vec::new(), &[0, 3]),
            Value::String("UniformOutput".into()),
            Value::Bool(false),
        ],
    )
    .unwrap();
    let Value::Cell(nonuniform) = nonuniform else {
        panic!("expected an empty cell array")
    };
    assert_eq!(nonuniform.shape, vec![0, 3]);
    assert!(nonuniform.data.is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejects_mismatched_shapes_and_late_cell_inputs() {
    let mismatch = call(
        Value::FunctionHandle("plus".into()),
        vec![
            cell(vec![Value::Num(1.0), Value::Num(2.0)], &[1, 2]),
            cell(vec![Value::Num(3.0), Value::Num(4.0)], &[2, 1]),
        ],
    )
    .unwrap_err();
    assert!(mismatch.message().contains("does not match"));

    let late = call(
        Value::FunctionHandle("plus".into()),
        vec![
            cell(vec![Value::Num(1.0)], &[1, 1]),
            Value::Num(2.0),
            cell(vec![Value::Num(3.0)], &[1, 1]),
        ],
    )
    .unwrap_err();
    assert!(late.message().contains("must precede"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn supports_closed_cellfun_shorthands() {
    let input = cell(
        vec![Value::Int(runmat_value::IntValue::I32(5)), Value::Num(3.0)],
        &[1, 2],
    );
    let result = call(
        Value::String("isclass".into()),
        vec![input, Value::String("int32".into())],
    )
    .unwrap();
    let Value::LogicalArray(output) = result else {
        panic!("expected logical output")
    };
    assert_eq!(output.data, vec![1, 0]);

    let sizes = call(
        Value::String("prodofsize".into()),
        vec![cell(
            vec![tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])],
            &[1, 1],
        )],
    )
    .unwrap();
    assert_eq!(tensor_values(sizes), vec![4.0]);
}
