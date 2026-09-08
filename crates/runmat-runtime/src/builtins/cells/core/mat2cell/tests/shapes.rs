use super::*;
use crate::builtins::common::test_support;
use crate::object::cell::index_cell_value;
use runmat_value::{CharArray, LogicalArray};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn three_dimensional_tensor_partition() {
    let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
    let result = run(
        Value::Tensor(tensor),
        vec![
            row_vector(&[1.0, 2.0]),
            row_vector(&[2.0, 2.0]),
            row_vector(&[1.0, 1.0]),
        ],
    )
    .expect("mat2cell");
    let cell = match result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    assert_eq!(cell.shape, vec![2, 2, 2]);
    let block = index_cell_value(&cell, &[2, 2, 1]).expect("N-D cell subscript");
    let gathered = test_support::gather(block).expect("gather");
    assert_eq!(gathered.shape, vec![2, 2, 1]);
    assert_eq!(gathered.materialize_f64(), vec![8.0, 9.0, 11.0, 12.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn zero_sized_blocks() {
    let tensor = Tensor::new(vec![0.0; 6], vec![3, 2]).unwrap();
    let result = run(
        Value::Tensor(tensor),
        vec![row_vector(&[0.0, 3.0]), row_vector(&[1.0, 1.0])],
    )
    .expect("mat2cell");
    let cell = match result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    assert_eq!(cell.shape, vec![2, 2]);
    let top_left = cell.data[0].clone();
    let gathered = test_support::gather(top_left).expect("gather");
    assert_eq!(gathered.materialize_f64().len(), 0);
    assert_eq!(gathered.shape, vec![0, 1]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_partition_vector_supported() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let logical =
        LogicalArray::new(vec![1, 1, 1, 1], vec![4, 1]).expect("logical partition vector");
    let result = run(Value::Tensor(tensor), vec![Value::LogicalArray(logical)]).expect("mat2cell");
    let cell = match result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    assert_eq!(cell.shape, vec![4, 1]);
    let third = cell.data[2].clone();
    let gathered = test_support::gather(third).expect("gather");
    assert_eq!(gathered.shape, vec![1, 1]);
    assert_eq!(gathered.materialize_f64(), vec![3.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn char_array_partition() {
    let chars = CharArray::new(
        vec!['f', 'o', 'o', ' ', 'b', 'a', 'r', ' ', 'b', 'a', 'z', ' '],
        3,
        4,
    )
    .unwrap();
    let result = run(
        Value::CharArray(chars),
        vec![row_vector(&[1.0, 2.0]), row_vector(&[2.0, 2.0])],
    )
    .expect("mat2cell");
    let cell = match result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    let second = cell.data[1].clone();
    match second {
        Value::CharArray(slice) => {
            assert_eq!(slice.rows, 1);
            assert_eq!(slice.cols, 2);
            let text: String = slice.data.iter().collect();
            assert_eq!(text, "o ");
        }
        other => panic!("expected CharArray, got {other:?}"),
    }
}
