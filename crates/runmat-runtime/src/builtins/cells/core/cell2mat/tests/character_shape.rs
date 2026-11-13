use super::*;
use runmat_value::{CharArray, Tensor};

#[test]
fn nd_cell_grid_positions_follow_semantic_linear_order() {
    for shape in [vec![2, 2], vec![2, 2, 2], vec![2, 1, 2, 3]] {
        let expected = (1..=shape.iter().product::<usize>())
            .map(|n| n as f64)
            .collect::<Vec<_>>();
        let values = expected.iter().copied().map(Value::Num).collect();
        let cells = runmat_value::CellArray::from_column_major(values, shape.clone()).unwrap();
        let Value::Tensor(result) = run(Value::Cell(cells)).unwrap() else {
            panic!("numeric result")
        };
        assert_eq!(result.shape(), shape.as_slice());
        assert_eq!(result.materialize_f64(), expected);
    }
}

#[test]
fn nd_character_contents_are_rejected_without_truncating_shape() {
    let chars = CharArray::from_unicode_scalars_with_shape(vec!['a', 'b', 'c', 'd'], vec![1, 2, 2])
        .unwrap();
    let cell = crate::make_cell(vec![Value::CharArray(chars)], 1, 1).unwrap();
    let error = run(cell).unwrap_err();
    assert!(error.to_string().contains("2-D character array"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn char_cell() {
    let a =
        Value::CharArray(CharArray::from_unicode_scalars("hi".chars().collect(), 1, 2).unwrap());
    let b =
        Value::CharArray(CharArray::from_unicode_scalars("BY".chars().collect(), 1, 2).unwrap());
    let cell = crate::make_cell(vec![a, b], 2, 1).expect("cell");
    let result = run(cell).expect("cell2mat");
    match result {
        Value::CharArray(arr) => {
            assert_eq!(arr.rows, 2);
            assert_eq!(arr.cols, 2);
            assert_eq!(arr.data(), runmat_value::encode_utf16_code_units("hiBY"));
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
            assert_eq!(t.shape(), &[2, 3, 2]);
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
            assert_eq!(t.shape(), &[0, 0]);
            assert!(t.materialize_f64().is_empty());
        }
        other => panic!("expected empty tensor, got {other:?}"),
    }
}
