use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_plus_expands_row_and_column_vectors() {
    let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let row = Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap();
    let result = call(
        Value::FunctionHandle("plus".into()),
        Value::Tensor(column),
        Value::Tensor(row),
    )
    .expect("bsxfun plus");

    let Value::Tensor(tensor) = result else {
        panic!("expected tensor result");
    };
    assert_eq!(tensor.shape(), &[3, 2]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn bsxfun_normalizes_rank_zero_and_rank_one_tensor_inputs_for_broadcasting() {
    let scalar = Tensor::new(vec![2.0], vec![]).expect("rank-zero tensor");
    let vector = Tensor::new(vec![10.0, 20.0], vec![2]).expect("rank-one tensor");
    let Value::Tensor(output) = call(
        Value::FunctionHandle("plus".into()),
        Value::Tensor(scalar),
        Value::Tensor(vector),
    )
    .expect("low-rank bsxfun") else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.materialize_f64(), vec![12.0, 22.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_nd_expansion_uses_matlab_trailing_dimensions() {
    let left = Tensor::new((1..=24).map(|value| value as f64).collect(), vec![2, 3, 4]).unwrap();
    let right = Tensor::new(vec![100.0, 200.0, 300.0], vec![1, 3]).unwrap();
    let result = call(
        Value::FunctionHandle("plus".into()),
        Value::Tensor(left),
        Value::Tensor(right),
    )
    .expect("bsxfun nd trailing expansion");

    let Value::Tensor(tensor) = result else {
        panic!("expected tensor result");
    };
    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.materialize_f64()[0], 101.0);
    assert_eq!(tensor.materialize_f64()[1], 102.0);
    assert_eq!(tensor.materialize_f64()[2], 203.0);
    assert_eq!(tensor.materialize_f64()[6], 107.0);
    assert_eq!(tensor.materialize_f64()[8], 209.0);
    assert_eq!(tensor.materialize_f64()[23], 324.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_rejects_front_padded_nd_shapes() {
    let left = Tensor::new((1..=24).map(|value| value as f64).collect(), vec![2, 3, 4]).unwrap();
    let right = Tensor::new((1..=12).map(|value| value as f64).collect(), vec![3, 4]).unwrap();
    let err = call(
        Value::FunctionHandle("plus".into()),
        Value::Tensor(left),
        Value::Tensor(right),
    )
    .expect_err("expected MATLAB trailing-dimension mismatch");
    assert_eq!(err.identifier(), BSXFUN_ERROR_SIZE_MISMATCH.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_singleton_dimension_can_expand_to_zero() {
    let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let result = call(
        Value::FunctionHandle("plus".into()),
        Value::Tensor(empty),
        Value::Tensor(row),
    )
    .expect("bsxfun empty");

    let Value::Tensor(tensor) = result else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), &[0, 3]);
    assert!(tensor.materialize_f64().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_rejects_incompatible_sizes() {
    let left = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = call(
        Value::FunctionHandle("plus".into()),
        Value::Tensor(left),
        Value::Tensor(right),
    )
    .expect_err("expected size mismatch");
    assert_eq!(err.identifier(), BSXFUN_ERROR_SIZE_MISMATCH.identifier);
}
