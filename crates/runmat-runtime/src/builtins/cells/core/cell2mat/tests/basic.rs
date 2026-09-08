use super::*;
use runmat_value::{NumericStorage, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn simple_numeric_cell() {
    let cell = scalar_cell(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let result = run(cell).expect("cell2mat");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn native_single_cells_preserve_class_and_block_layout() {
    let first =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25, 2.5]), vec![2, 1]).unwrap();
    let second =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![3.75, 4.5]), vec![2, 1]).unwrap();
    let cell = crate::make_cell(vec![Value::Tensor(first), Value::Tensor(second)], 1, 2).unwrap();
    let Value::Tensor(output) = run(cell).expect("cell2mat") else {
        panic!("expected tensor");
    };

    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(
        output.into_numeric_storage(),
        Ok(NumericStorage::F32(vec![1.25, 2.5, 3.75, 4.5]))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn block_concatenation() {
    let row1_left = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor"));
    let row1_right = Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).expect("tensor"));
    let row2_left = Value::Tensor(Tensor::new(vec![6.0, 7.0], vec![1, 2]).expect("tensor"));
    let row2_right = Value::Tensor(Tensor::new(vec![8.0, 9.0, 10.0], vec![1, 3]).expect("tensor"));
    let cell =
        crate::make_cell(vec![row1_left, row1_right, row2_left, row2_right], 2, 2).expect("cell");
    let result = run(cell).expect("cell2mat");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 5]);
            assert_eq!(
                t.materialize_f64(),
                vec![1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0]
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_cell() {
    let a = Value::Bool(true);
    let b = Value::Bool(false);
    let c = Value::Bool(true);
    let d = Value::Bool(false);
    let cell = crate::make_cell(vec![a, b, c, d], 2, 2).expect("cell");
    let result = run(cell).expect("cell2mat");
    match result {
        Value::LogicalArray(la) => {
            assert_eq!(la.shape, vec![2, 2]);
            assert_eq!(la.data, vec![1, 1, 0, 0]);
        }
        other => panic!("expected logical array result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_cell() {
    let values = vec![Value::Complex(1.0, 2.0), Value::Complex(3.0, 4.0)];
    let cell = crate::make_cell(values, 1, 2).expect("cell");
    let result = run(cell).expect("cell2mat");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![1, 2]);
            assert_eq!(ct.materialize_f64(), vec![(1.0, 2.0), (3.0, 4.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}
