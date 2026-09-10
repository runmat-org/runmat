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
