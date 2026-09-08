use super::*;
use crate::builtins::common::test_support;
use runmat_value::NumericStorage;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn partition_matrix_into_quadrants() {
    let tensor = Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).unwrap();
    let result = run(
        Value::Tensor(tensor),
        vec![row_vector(&[2.0, 2.0]), row_vector(&[1.0, 3.0])],
    )
    .expect("mat2cell");

    let cell = match result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    assert_eq!(cell.shape, vec![2, 2]);

    let bottom_right = cell.data[3].clone();
    let gathered = test_support::gather(bottom_right).expect("gather");
    assert_eq!(gathered.shape, vec![2, 3]);
    assert_eq!(
        gathered.materialize_f64(),
        vec![7.0, 8.0, 11.0, 12.0, 15.0, 16.0]
    );
}

#[test]
fn native_single_data_and_partition_vectors_remain_typed() {
    let input = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let rows = Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap();
    let columns = Tensor::from_f32(vec![1.0, 1.0], vec![1, 2]).unwrap();
    let Value::Cell(cells) = run(
        Value::Tensor(input),
        vec![Value::Tensor(rows), Value::Tensor(columns)],
    )
    .expect("mat2cell") else {
        panic!("expected cell array");
    };
    assert_eq!(cells.shape, vec![1, 2]);
    let expected = [vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
    for (value, expected) in cells.data.into_iter().zip(expected) {
        let Value::Tensor(tensor) = value else {
            panic!("expected single tensor block");
        };
        assert_eq!(
            tensor.into_numeric_storage().unwrap(),
            NumericStorage::F32(expected)
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn row_vector_with_single_partition_vector() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]).unwrap();
    let result = run(
        Value::Tensor(tensor),
        vec![row_vector(&[1.0]), row_vector(&[2.0, 1.0, 3.0])],
    )
    .expect("mat2cell");
    let cell = match result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    assert_eq!(cell.shape, vec![1, 3]);
    assert_eq!(cell.data.len(), 3);
    let third = cell.data[2].clone();
    let gathered = test_support::gather(third).expect("gather");
    assert_eq!(gathered.materialize_f64(), vec![4.0, 5.0, 6.0]);
    assert_eq!(gathered.shape, vec![1, 3]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn column_vector_with_implicit_column_partition() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let result = run(Value::Tensor(tensor), vec![column_vector(&[2.0, 2.0])]).expect("mat2cell");
    let cell = match result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    assert_eq!(cell.shape, vec![2, 1]);
    let second = cell.data[1].clone();
    let gathered = test_support::gather(second).expect("gather");
    assert_eq!(gathered.shape, vec![2, 1]);
    assert_eq!(gathered.materialize_f64(), vec![3.0, 4.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_scalar_partition_yields_complex_value() {
    let result = run(
        Value::Complex(1.25, -2.5),
        vec![row_vector(&[1.0]), row_vector(&[1.0])],
    )
    .expect("mat2cell");
    let cell = match result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    assert_eq!(cell.shape, vec![1, 1]);
    let value = cell.data[0].clone();
    match value {
        Value::Complex(re, im) => {
            assert!((re - 1.25).abs() < 1e-12);
            assert!((im + 2.5).abs() < 1e-12);
        }
        other => panic!("expected complex scalar, got {other:?}"),
    }
}
