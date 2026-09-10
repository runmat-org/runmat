use super::*;
use crate::indexing::plan::{build_assignment_plan, build_index_plan};
use crate::indexing::selectors::SliceSelector;
use runmat_value::{StructValue, Tensor};

fn element(value: f64) -> StructValue {
    let mut structure = StructValue::new();
    structure.insert("value", Value::Num(value));
    structure
}

fn array(shape: Vec<usize>) -> Value {
    let count = shape.iter().product();
    Value::StructArray(
        StructArray::new(
            (0..count)
                .map(|index| element(index as f64 + 1.0))
                .collect(),
            shape,
        )
        .unwrap(),
    )
}

fn empty_rhs() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap())
}

fn shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Struct(_) => vec![1, 1],
        Value::StructArray(value) => value.shape().to_vec(),
        _ => panic!("expected structure value"),
    }
}

#[test]
fn scalar_linear_growth_preserves_schema_and_fills_holes() {
    let plan = build_assignment_plan(&[SliceSelector::Scalar(3)], 1, &[1, 1]).unwrap();
    let result = assign_with_plan(
        Value::Struct(element(1.0)),
        &plan,
        Value::Struct(element(3.0)),
        false,
    )
    .unwrap();
    assert_eq!(shape(&result), vec![1, 3]);
    let Value::StructArray(result) = result else {
        unreachable!()
    };
    assert!(
        matches!(result.get_linear(1).unwrap().fields.get("value"), Some(Value::Tensor(value)) if value.is_empty())
    );
    assert_eq!(
        result.get_linear(2).unwrap().fields.get("value"),
        Some(&Value::Num(3.0))
    );
}

#[test]
fn assignment_requires_the_exact_ordered_schema() {
    let mut reversed = StructValue::new();
    reversed.insert("other", Value::Num(1.0));
    reversed.insert("value", Value::Num(2.0));
    let plan = build_index_plan(&[SliceSelector::Scalar(1)], 1, &[1, 2]).unwrap();
    let error =
        assign_with_plan(array(vec![1, 2]), &plan, Value::Struct(reversed), false).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:DissimilarStructureAssignment")
    );
}

#[test]
fn linear_deletion_preserves_vector_orientation_and_deduplicates() {
    let row_plan = IndexPlan::new(vec![0, 0, 2], vec![1, 3], vec![3], 1, vec![1, 4]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![1, 4]), &row_plan, empty_rhs(), true).unwrap()),
        vec![1, 2]
    );
    let column_plan = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2], 1, vec![4, 1]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![4, 1]), &column_plan, empty_rhs(), true).unwrap()),
        vec![2, 1]
    );
    let all_row = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 2]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![1, 2]), &all_row, empty_rhs(), true).unwrap()),
        vec![1, 0]
    );
    let all_column = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2], 1, vec![2, 1]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 1]), &all_column, empty_rhs(), true).unwrap()),
        vec![0, 1]
    );
}

#[test]
fn scalar_and_empty_selection_deletion_are_normalized() {
    let scalar = IndexPlan::new(vec![0], vec![1, 1], vec![1], 1, vec![1, 1]);
    assert_eq!(
        shape(&assign_with_plan(Value::Struct(element(1.0)), &scalar, empty_rhs(), true).unwrap()),
        vec![0, 0]
    );
    let none = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![1, 2]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![1, 2]), &none, empty_rhs(), true).unwrap()),
        vec![1, 2]
    );
}

#[test]
fn subscript_deletion_supports_one_dimension_and_rejects_two_partial_dimensions() {
    let row = build_index_plan(
        &[SliceSelector::Scalar(1), SliceSelector::Colon],
        2,
        &[2, 3],
    )
    .unwrap();
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 3]), &row, empty_rhs(), true).unwrap()),
        vec![1, 3]
    );
    let column = build_index_plan(
        &[SliceSelector::Colon, SliceSelector::Scalar(2)],
        2,
        &[2, 3],
    )
    .unwrap();
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 3]), &column, empty_rhs(), true).unwrap()),
        vec![2, 2]
    );
    let invalid = build_index_plan(
        &[SliceSelector::Scalar(1), SliceSelector::Scalar(1)],
        2,
        &[2, 2],
    )
    .unwrap();
    assert_eq!(
        assign_with_plan(array(vec![2, 2]), &invalid, empty_rhs(), true)
            .unwrap_err()
            .identifier(),
        Some("RunMat:UnsupportedStructDeletion")
    );
}

#[test]
fn full_and_nd_dimension_deletion_preserve_typed_empty_and_shape() {
    let full = build_index_plan(&[SliceSelector::Colon, SliceSelector::Colon], 2, &[2, 2]).unwrap();
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 2]), &full, empty_rhs(), true).unwrap()),
        vec![0, 0]
    );
    let nd = build_index_plan(
        &[
            SliceSelector::Colon,
            SliceSelector::Scalar(1),
            SliceSelector::Colon,
        ],
        3,
        &[2, 2, 2],
    )
    .unwrap();
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 2, 2]), &nd, empty_rhs(), true).unwrap()),
        vec![2, 1, 2]
    );
}

#[test]
fn deletion_rejects_nonempty_rhs() {
    let plan = build_index_plan(&[SliceSelector::Scalar(1)], 1, &[1, 2]).unwrap();
    assert_eq!(
        assign_with_plan(array(vec![1, 2]), &plan, Value::Num(0.0), true)
            .unwrap_err()
            .identifier(),
        Some("RunMat:InvalidStructDeletionRhs")
    );
}
