use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_additional_scalar_argument() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![0.5, 1.0, -1.0], vec![3, 1]).unwrap();
    let expected: Vec<f64> = values(&tensor).into_iter().map(|y| y.atan2(1.0)).collect();
    let result = call(
        Value::FunctionHandle("atan2".into()),
        vec![Value::Tensor(tensor), Value::Num(1.0)],
    )
    .expect("arrayfun");
    match result.value() {
        Value::Tensor(out) => {
            assert_eq!(values(out), expected);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_uniform_false_returns_cell() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let expected: Vec<Value> = values(&tensor)
        .into_iter()
        .map(|x| Value::Num(x.sin()))
        .collect();
    let result = call(
        Value::FunctionHandle("sin".into()),
        vec![
            Value::Tensor(tensor),
            Value::String("UniformOutput".into()),
            Value::Bool(false),
        ],
    )
    .expect("arrayfun");
    let Value::Cell(cell) = result.value() else {
        panic!("expected cell, got something else");
    };
    assert_eq!(cell.rows, 2);
    assert_eq!(cell.cols, 1);
    for (row, value) in expected.iter().enumerate() {
        assert_eq!(cell.get(row, 0).unwrap(), *value);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_uniform_output_option_identifier() {
    let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let err = call(
        Value::FunctionHandle("sin".into()),
        vec![
            Value::Tensor(tensor),
            Value::String("UniformOutput".into()),
            Value::String("maybe".into()),
        ],
    )
    .expect_err("expected invalid uniform output option");
    assert_eq!(
        err.identifier(),
        ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_unknown_name_value_identifier() {
    let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let err = call(
        Value::FunctionHandle("sin".into()),
        vec![
            Value::Tensor(tensor),
            Value::String("MysteryFlag".into()),
            Value::Bool(true),
        ],
    )
    .expect_err("expected unknown name-value error");
    assert_eq!(err.identifier(), ARRAYFUN_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_size_mismatch_errors() {
    let taller = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let shorter = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
    let err = call(
        Value::FunctionHandle("sin".into()),
        vec![Value::Tensor(taller), Value::Tensor(shorter)],
    )
    .expect_err("expected size mismatch error");
    let err = err.to_string();
    assert!(
        err.contains("does not match"),
        "expected size mismatch error, got {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_uniform_logical_result() {
    let tensor = Tensor::new(vec![1.0, f64::NAN, 0.0, f64::INFINITY], vec![4, 1]).unwrap();
    let result = call(
        Value::FunctionHandle("isfinite".into()),
        vec![Value::Tensor(tensor)],
    )
    .expect("arrayfun isfinite");
    match result.value() {
        Value::LogicalArray(la) => {
            assert_eq!(la.shape, vec![4, 1]);
            assert_eq!(la.data, vec![1, 0, 1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_uniform_character_result() {
    let tensor = Tensor::new(vec![65.0, 66.0, 67.0], vec![1, 3]).unwrap();
    let result = call(
        Value::FunctionHandle("char".into()),
        vec![Value::Tensor(tensor)],
    )
    .expect("arrayfun char");
    match result.value() {
        Value::CharArray(ca) => {
            assert_eq!(ca.rows, 1);
            assert_eq!(ca.cols, 3);
            assert_eq!(ca.data(), runmat_value::encode_utf16_code_units("ABC"));
        }
        other => panic!("expected char array, got {other:?}"),
    }
}
