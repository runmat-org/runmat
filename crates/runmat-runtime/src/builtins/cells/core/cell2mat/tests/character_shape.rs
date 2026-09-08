use super::*;
use runmat_value::{CharArray, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn char_cell() {
    let a = Value::CharArray(CharArray::new("hi".chars().collect(), 1, 2).unwrap());
    let b = Value::CharArray(CharArray::new("BY".chars().collect(), 1, 2).unwrap());
    let cell = crate::make_cell(vec![a, b], 2, 1).expect("cell");
    let result = run(cell).expect("cell2mat");
    match result {
        Value::CharArray(arr) => {
            assert_eq!(arr.rows, 2);
            assert_eq!(arr.cols, 2);
            assert_eq!(arr.data, vec!['h', 'i', 'B', 'Y']);
        }
        other => panic!("expected char array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mismatched_block_sizes_error() {
    let a = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let b = Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap());
    let cell = crate::make_cell(vec![a, b], 1, 2).expect("cell");
    let err = run(cell).unwrap_err().to_string();
    assert!(err.contains("block sizes"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn higher_dimensional_tiling() {
    let a = Value::Tensor(Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap());
    let b = Value::Tensor(Tensor::new(vec![2.0; 4], vec![2, 1, 2]).unwrap());
    let cell = crate::make_cell(vec![a, b], 1, 2).expect("cell");
    let result = run(cell).expect("cell2mat");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 3, 2]);
            assert_eq!(t.materialize_f64().len(), 12);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn empty_cell_returns_empty_double() {
    let cell = crate::make_cell(Vec::new(), 0, 0).expect("cell");
    let result = run(cell).expect("cell2mat");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![0, 0]);
            assert!(t.materialize_f64().is_empty());
        }
        other => panic!("expected empty tensor, got {other:?}"),
    }
}
