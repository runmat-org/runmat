use super::super::{
    assign_tensor_with_plan, build_complex_rhs_view, build_string_rhs_view,
    integer_gpu_rhs_indices_for_plan, materialize_rhs_nd_real, validate_string_rhs_for_plan,
};
use super::validated_assignment_rhs_shape;
use crate::indexing::plan::IndexPlan;
use futures::executor::block_on;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntegerStorage, LogicalArray, StringArray, Tensor, Value,
};

#[test]
fn externally_constructed_complex_plan_cannot_overrun_borrowed_rhs_ordinals() {
    let plan = IndexPlan::new(vec![0, 1], vec![1, 2], vec![1], 1, vec![1, 2]);
    let rhs = Value::ComplexTensor(ComplexTensor::new(vec![(2.0, 3.0)], vec![1, 1]).unwrap());
    let target = ComplexTensor::new(vec![(0.0, 0.0); 2], vec![1, 2]).unwrap();
    let error = super::super::assign_complex_with_plan_host_in_context(None, target, &plan, &rhs)
        .expect_err("mismatched public plan must not reach a second source ordinal");
    assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
}

#[test]
fn empty_string_selection_validates_text_and_cellstr_without_copying_payloads() {
    let selected = [0];
    for rhs in [
        Value::String("scalar".into()),
        Value::CharArray(CharArray::new_row("scalar")),
        Value::StringArray(StringArray::new(vec![], vec![0, 1]).unwrap()),
        Value::Cell(CellArray::new_with_shape(vec![], vec![0, 1]).unwrap()),
    ] {
        validate_string_rhs_for_plan(&rhs, &selected).expect("scalar or compatible empty RHS");
    }
    for (rhs, expected) in [
        (
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
            "RunMat:ShapeMismatch",
        ),
        (
            Value::Cell(
                CellArray::new_with_shape(
                    vec![Value::String("a".into()), Value::String("b".into())],
                    vec![1, 2],
                )
                .unwrap(),
            ),
            "RunMat:ShapeMismatch",
        ),
        (
            Value::Cell(CellArray::new_with_shape(vec![Value::Num(1.0)], vec![1, 1]).unwrap()),
            "RunMat:InvalidSliceAssignmentRhs",
        ),
    ] {
        let error = validate_string_rhs_for_plan(&rhs, &selected)
            .expect_err("incompatible empty-selection RHS");
        assert_eq!(error.identifier(), Some(expected));
    }
}

#[test]
fn singleton_dimensions_align_without_nonscalar_expansion() {
    for (rhs, selected, count) in [
        (vec![1, 2], vec![2, 1], 2),
        (vec![2, 1], vec![1, 2], 2),
        (vec![1, 2, 1, 3], vec![2, 1, 3, 1], 6),
        (vec![0, 1], vec![1, 0], 0),
    ] {
        assert_eq!(
            validated_assignment_rhs_shape(&rhs, count, &selected).expect("singleton alignment"),
            selected
        );
    }
    for (rhs, count, selected) in [
        (vec![3, 2], 6, vec![2, 3]),
        (vec![1, 2], 2, vec![2, 2]),
        (vec![1, 2, 3], 5, vec![2, 3]),
        (vec![usize::MAX, 2], 0, vec![usize::MAX, 2]),
    ] {
        let error = validated_assignment_rhs_shape(&rhs, count, &selected)
            .expect_err("order, count, and overflow remain checked");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }
}

#[test]
fn numeric_logical_complex_and_string_views_share_singleton_alignment() {
    let real = Value::Tensor(Tensor::new(vec![11.0, 12.0], vec![1, 2]).unwrap());
    assert_eq!(
        block_on(materialize_rhs_nd_real(&real, &[2, 1])).unwrap(),
        vec![11.0, 12.0]
    );
    let logical = Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap());
    assert_eq!(
        block_on(materialize_rhs_nd_real(&logical, &[2, 1])).unwrap(),
        vec![1.0, 0.0]
    );
    let complex = Value::ComplexTensor(
        ComplexTensor::new(vec![(11.0, 1.0), (12.0, 2.0)], vec![1, 2]).unwrap(),
    );
    let view = build_complex_rhs_view(&complex, &[2, 1]).unwrap();
    assert_eq!(view.tensor_value(0), (11.0, 1.0));
    assert_eq!(view.tensor_value(1), (12.0, 2.0));
    let strings = Value::StringArray(
        StringArray::new(vec!["first".into(), "second".into()], vec![1, 2]).unwrap(),
    );
    let view = build_string_rhs_view(&strings, &[2, 1]).unwrap();
    assert_eq!(view.count(), 2);
    assert_eq!(
        (view.value_at(0).as_str(), view.value_at(1).as_str()),
        (Some("first"), Some("second"))
    );
    let cells = Value::Cell(
        CellArray::new(
            vec![
                Value::String("first".into()),
                Value::String("second".into()),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let view = build_string_rhs_view(&cells, &[2, 1]).unwrap();
    assert_eq!(view.count(), 2);
    assert_eq!(
        (view.value_at(0).as_str(), view.value_at(1).as_str()),
        (Some("first"), Some("second"))
    );
}

#[test]
fn empty_real_and_logical_materialization_never_synthesizes_elements() {
    let real = Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).unwrap());
    let logical = Value::LogicalArray(LogicalArray::new(Vec::new(), vec![0, 1]).unwrap());
    for rhs in [real, logical, Value::Num(7.0)] {
        assert!(block_on(materialize_rhs_nd_real(&rhs, &[1, 0]))
            .unwrap()
            .is_empty());
    }
    let nonempty = Value::Tensor(Tensor::new(vec![11.0, 12.0], vec![1, 2]).unwrap());
    let error = block_on(materialize_rhs_nd_real(&nonempty, &[1, 0]))
        .expect_err("nonempty nonscalar RHS cannot fill an empty selection");
    assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
}

#[test]
fn native_wide_integer_scatter_and_device_indices_preserve_element_order() {
    let values = vec![9_007_199_254_740_993, u64::MAX];
    let rhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(values.clone()), vec![1, 2]).unwrap(),
    );
    let target = Tensor::new_integer(IntegerStorage::U64(vec![0; 4]), vec![2, 2]).unwrap();
    let plan = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2, 1], 2, vec![2, 2]);
    let Value::Tensor(output) = block_on(assign_tensor_with_plan(target, &plan, &rhs)).unwrap()
    else {
        panic!("native integer output");
    };
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![values[0], values[1], 0, 0]))
    );
    assert_eq!(
        integer_gpu_rhs_indices_for_plan(&[1, 2], &plan).unwrap(),
        vec![0, 1]
    );
}
