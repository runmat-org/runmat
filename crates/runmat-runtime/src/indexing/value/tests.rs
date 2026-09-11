use super::*;
use crate::indexing::plan::{build_assignment_plan, build_index_plan};
use crate::indexing::selectors::SliceSelector;
use futures::executor::block_on;
use runmat_value::CharArray;

#[test]
fn repeated_scalar_reads_preserve_runtime_family_and_shape() {
    let plan = build_index_plan(
        &[SliceSelector::LinearIndices {
            values: vec![1, 1, 1],
            output_shape: vec![1, 3],
        }],
        1,
        &[1, 1],
    )
    .expect("plan");
    assert!(
        matches!(read_with_plan(Value::Num(4.0), &plan).unwrap(), Value::Tensor(v) if v.shape == [1, 3])
    );
    assert!(
        matches!(read_with_plan(Value::Bool(true), &plan).unwrap(), Value::LogicalArray(v) if *v.data == [1, 1, 1])
    );
    assert!(
        matches!(read_with_plan(Value::String("entry".into()), &plan).unwrap(), Value::StringArray(v) if v.data == ["entry", "entry", "entry"])
    );
}

#[test]
fn character_and_logical_scalar_assignment_grow_through_shared_plan() {
    let plan = build_assignment_plan(&[SliceSelector::Scalar(3)], 1, &[1, 1]).unwrap();
    assert!(
        matches!(block_on(assign_with_plan(Value::Bool(true), &plan, Value::Bool(false))).unwrap(), Value::LogicalArray(v) if *v.data == [1, 0, 0])
    );
    assert!(
        matches!(block_on(assign_with_plan(Value::CharArray(CharArray::new_row("a")), &plan, Value::CharArray(CharArray::new_row("z")))).unwrap(), Value::CharArray(v) if v.row_string().as_deref() == Some("a z"))
    );
}

#[test]
fn numeric_assignment_growth_uses_the_shared_checked_plan_shape() {
    let scalar_plan = build_assignment_plan(&[SliceSelector::Scalar(2)], 1, &[1, 1]).unwrap();
    let Value::Tensor(scalar) = block_on(assign_with_plan(
        Value::Num(1.0),
        &scalar_plan,
        Value::Num(3.0),
    ))
    .unwrap() else {
        panic!("scalar growth must materialize a numeric tensor")
    };
    assert_eq!(scalar.shape, vec![1, 2]);
    assert_eq!(scalar.materialize_f64(), vec![1.0, 3.0]);

    let vector = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let vector_plan = build_assignment_plan(&[SliceSelector::Scalar(4)], 1, &[1, 2]).unwrap();
    let Value::Tensor(vector) = block_on(assign_with_plan(
        Value::Tensor(vector),
        &vector_plan,
        Value::Num(9.0),
    ))
    .unwrap() else {
        panic!("vector growth must retain a numeric tensor")
    };
    assert_eq!(vector.shape, vec![1, 4]);
    assert_eq!(vector.materialize_f64(), vec![1.0, 2.0, 0.0, 9.0]);

    let matrix = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let matrix_plan = build_assignment_plan(
        &[SliceSelector::Scalar(1), SliceSelector::Scalar(3)],
        2,
        &[2, 1],
    )
    .unwrap();
    let Value::Tensor(matrix) = block_on(assign_with_plan(
        Value::Tensor(matrix),
        &matrix_plan,
        Value::Num(7.0),
    ))
    .unwrap() else {
        panic!("matrix growth must retain a numeric tensor")
    };
    assert_eq!(matrix.shape, vec![2, 3]);
    assert_eq!(matrix.materialize_f64(), vec![1.0, 2.0, 0.0, 0.0, 7.0, 0.0]);
}
