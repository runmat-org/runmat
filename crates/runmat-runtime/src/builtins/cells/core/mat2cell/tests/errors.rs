use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mismatch_partition_sum_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let err = run(
        Value::Tensor(tensor),
        vec![row_vector(&[1.0]), row_vector(&[3.0])],
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("partition sizes"),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn negative_partition_entry_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let err = run(Value::Tensor(tensor), vec![row_vector(&[-1.0, 5.0])])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("non-negative"),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn non_integer_partition_entry_errors() {
    let tensor = Tensor::new((1..=4).map(|v| v as f64).collect(), vec![4, 1]).unwrap();
    let err = run(Value::Tensor(tensor), vec![row_vector(&[1.5, 0.5, 2.0])])
        .unwrap_err()
        .to_string();
    assert!(err.contains("integers"), "unexpected error message: {err}");
}
