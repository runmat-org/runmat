use super::super::num2cell_builtin;
use futures::executor::block_on;
use runmat_value::{CellArray, ComplexTensor, Tensor, Value};

#[test]
fn allocation_free_partition_address_matches_grouped_order() {
    let plan = super::super::plan::PartitionPlan::new(&[2, 3, 4], &[3, 1]).unwrap();
    for (output, grouped) in plan.groups().enumerate() {
        let grouped = grouped.unwrap();
        assert_eq!(grouped.len(), plan.slice_count());
        for (slice, source) in grouped.into_iter().enumerate() {
            assert_eq!(plan.source_index(output, slice).unwrap(), source);
        }
    }
    assert!(plan.source_index(plan.output_count(), 0).is_err());
}

#[test]
fn rejects_matrix_duplicate_and_out_of_range_dimensions() {
    let input = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    for dims in [
        Tensor::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).unwrap(),
        Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap(),
        Tensor::new(vec![3.0], vec![1, 1]).unwrap(),
    ] {
        assert!(block_on(num2cell_builtin(
            None,
            input(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(dims)])
        ))
        .is_err());
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
        let error = block_on(num2cell_builtin(
            None,
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect_err("rank-zero source has no first axis");
        assert_eq!(error.identifier(), Some("RunMat:num2cell:InvalidInput"));
        assert!(error
            .to_string()
            .contains("dimensions must be between 1 and 0"));
    }
}
