use super::super::num2cell_builtin;
use futures::executor::block_on;
use runmat_value::{CellArray, ComplexTensor, Tensor, Value};

#[test]
fn rejects_matrix_duplicate_and_out_of_range_dimensions() {
    let input = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    for dims in [
        Tensor::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).unwrap(),
        Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap(),
        Tensor::new(vec![3.0], vec![1, 1]).unwrap(),
    ] {
        assert!(block_on(num2cell_builtin(input(), vec![Value::Tensor(dims)])).is_err());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn explicit_group_dimension_still_rejects_rank_zero_sources() {
    for input in [
        Value::Tensor(Tensor::new(vec![2.0], vec![]).unwrap()),
        Value::ComplexTensor(ComplexTensor::new(vec![(2.0, 3.0)], vec![]).unwrap()),
        Value::Cell(CellArray::new_with_shape(Vec::new(), vec![]).unwrap()),
    ] {
        let error = block_on(num2cell_builtin(input, vec![Value::Num(1.0)]))
            .expect_err("rank-zero source has no first axis");
        assert_eq!(error.identifier(), Some("RunMat:num2cell:InvalidInput"));
        assert!(error
            .to_string()
            .contains("dimensions must be between 1 and 0"));
    }
}
