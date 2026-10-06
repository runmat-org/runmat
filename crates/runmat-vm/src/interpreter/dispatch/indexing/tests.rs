use super::{
    apply_cell_end_exprs_for_base, apply_cell_end_offsets_for_base, apply_end_offsets_to_numeric,
    assign_integer_scalar_with_plan, integer_scalar_tensor, map_slice_plan_error,
    range_selector_scalar_to_f64, symbolic_scalar_from_value, validate_expr_range_step_metadata,
    IndexContext,
};
use futures::executor::block_on;
use runmat_runtime::indexing::plan::IndexPlan;
use runmat_runtime::indexing::EndExpr;
use runmat_value::{
    CellArray, IntValue, IntegerStorage, ObjectArray, ObjectInstance, SymbolicExpr, Tensor, Value,
};

#[test]
fn integer_scalar_index_assignment_materialization_preserves_all_integer_classes() {
    macro_rules! assert_scalar_storage {
        ($value:expr, $storage:expr) => {{
            let tensor = integer_scalar_tensor($value).expect("integer scalar tensor");
            assert_eq!(tensor.integer_storage(), Some(&$storage));
        }};
    }

    assert_scalar_storage!(IntValue::I8(i8::MIN), IntegerStorage::I8(vec![i8::MIN]));
    assert_scalar_storage!(IntValue::I16(i16::MIN), IntegerStorage::I16(vec![i16::MIN]));
    assert_scalar_storage!(IntValue::I32(i32::MIN), IntegerStorage::I32(vec![i32::MIN]));
    assert_scalar_storage!(IntValue::I64(i64::MIN), IntegerStorage::I64(vec![i64::MIN]));
    assert_scalar_storage!(IntValue::U8(u8::MAX), IntegerStorage::U8(vec![u8::MAX]));
    assert_scalar_storage!(IntValue::U16(u16::MAX), IntegerStorage::U16(vec![u16::MAX]));
    assert_scalar_storage!(IntValue::U32(u32::MAX), IntegerStorage::U32(vec![u32::MAX]));
    assert_scalar_storage!(IntValue::U64(u64::MAX), IntegerStorage::U64(vec![u64::MAX]));
}

#[test]
fn integer_scalar_slice_assignment_preserves_all_integer_classes() {
    macro_rules! assert_slice_assignment {
        ($value:expr, $storage:expr) => {{
            let plan = IndexPlan::new(vec![0], vec![1, 1], vec![1], 1, vec![1, 1]);
            let result = block_on(assign_integer_scalar_with_plan(
                $value.clone(),
                &plan,
                &Value::Int($value),
                false,
            ))
            .expect("integer scalar slice assignment");
            let Value::Tensor(tensor) = result else {
                panic!("integer scalar slice assignment must return a tensor");
            };
            assert_eq!(tensor.integer_storage(), Some(&$storage));
        }};
    }

    assert_slice_assignment!(IntValue::I8(i8::MIN), IntegerStorage::I8(vec![i8::MIN]));
    assert_slice_assignment!(IntValue::I16(i16::MIN), IntegerStorage::I16(vec![i16::MIN]));
    assert_slice_assignment!(IntValue::I32(i32::MIN), IntegerStorage::I32(vec![i32::MIN]));
    assert_slice_assignment!(IntValue::I64(i64::MIN), IntegerStorage::I64(vec![i64::MIN]));
    assert_slice_assignment!(IntValue::U8(u8::MAX), IntegerStorage::U8(vec![u8::MAX]));
    assert_slice_assignment!(IntValue::U16(u16::MAX), IntegerStorage::U16(vec![u16::MAX]));
    assert_slice_assignment!(IntValue::U32(u32::MAX), IntegerStorage::U32(vec![u32::MAX]));
    assert_slice_assignment!(IntValue::U64(u64::MAX), IntegerStorage::U64(vec![u64::MAX]));
}

#[test]
fn map_slice_plan_error_preserves_identifier_and_adds_context() {
    let err = crate::interpreter::errors::mex("ShapeMismatch", "shape mismatch");
    let mapped = map_slice_plan_error("slice assign", err);
    assert_eq!(mapped.identifier(), Some("RunMat:ShapeMismatch"));
    assert_eq!(mapped.message(), "slice assign: shape mismatch");
}

#[test]
fn map_slice_plan_error_keeps_identifier_absent_when_missing() {
    let err = runmat_runtime::build_runtime_error("plain error").build();
    let mapped = map_slice_plan_error("slice assign", err);
    assert_eq!(mapped.identifier(), None);
    assert_eq!(mapped.message(), "slice assign: plain error");
}

#[test]
fn validate_expr_range_step_metadata_rejects_mismatched_arity() {
    let err = validate_expr_range_step_metadata(&[0, 1], &[true])
        .expect_err("mismatched range step metadata should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
}

#[test]
fn range_selector_scalar_to_f64_rejects_non_numeric_scalar() {
    let err = block_on(range_selector_scalar_to_f64(&Value::String(
        "x".to_string(),
    )))
    .expect_err("non-numeric range selector scalar should fail");
    assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
}

#[test]
fn typed_integer_scalar_range_and_symbolic_paths_ignore_f64_mirrors() {
    macro_rules! assert_typed_scalar {
        ($storage:expr, $expected:expr) => {{
            let tensor = Tensor::new_integer($storage, vec![1, 1]).expect("typed integer scalar");
            let value = Value::Tensor(tensor);

            assert_eq!(
                block_on(range_selector_scalar_to_f64(&value)).unwrap(),
                $expected
            );
            assert_eq!(
                symbolic_scalar_from_value(&value).unwrap(),
                SymbolicExpr::constant($expected)
            );
        }};
    }

    assert_typed_scalar!(IntegerStorage::I8(vec![-8]), -8.0);
    assert_typed_scalar!(IntegerStorage::I16(vec![-16]), -16.0);
    assert_typed_scalar!(IntegerStorage::I32(vec![-32]), -32.0);
    assert_typed_scalar!(IntegerStorage::I64(vec![-64]), -64.0);
    assert_typed_scalar!(IntegerStorage::U8(vec![8]), 8.0);
    assert_typed_scalar!(IntegerStorage::U16(vec![16]), 16.0);
    assert_typed_scalar!(IntegerStorage::U32(vec![32]), 32.0);
    assert_typed_scalar!(IntegerStorage::U64(vec![64]), 64.0);
}

#[test]
fn apply_end_offsets_rejects_out_of_bounds_positions() {
    let mut vars = vec![];
    let err = block_on(apply_end_offsets_to_numeric(
        &[Value::Num(1.0)],
        IndexContext::new(1, 0, 0, &[], &[5]),
        &[(1, EndExpr::End)],
        &mut vars,
    ))
    .expect_err("out-of-bounds end-selector position should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidEndSelectorPlan"));
}

#[test]
fn apply_end_offsets_rejects_duplicate_positions() {
    let mut vars = vec![];
    let err = block_on(apply_end_offsets_to_numeric(
        &[Value::Num(1.0)],
        IndexContext::new(1, 0, 0, &[], &[5]),
        &[(0, EndExpr::End), (0, EndExpr::Const(2.0))],
        &mut vars,
    ))
    .expect_err("duplicate end-selector positions should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidEndSelectorPlan"));
}

#[test]
fn apply_end_offsets_rejects_duplicate_range_dims_in_context() {
    let mut vars = vec![];
    let err = block_on(apply_end_offsets_to_numeric(
        &[Value::Num(1.0)],
        IndexContext::new(2, 0, 0, &[1, 1], &[5, 5]),
        &[(0, EndExpr::End)],
        &mut vars,
    ))
    .expect_err("duplicate range dims should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
}

#[test]
fn apply_end_offsets_rejects_out_of_bounds_range_dims_in_context() {
    let mut vars = vec![];
    let err = block_on(apply_end_offsets_to_numeric(
        &[Value::Num(1.0)],
        IndexContext::new(2, 0, 0, &[2], &[5, 5]),
        &[(0, EndExpr::End)],
        &mut vars,
    ))
    .expect_err("out-of-bounds range dims should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorDim"));
}

#[test]
fn apply_end_offsets_rejects_range_dim_conflicting_with_colon_mask_in_context() {
    let mut vars = vec![];
    let err = block_on(apply_end_offsets_to_numeric(
        &[Value::Num(1.0)],
        IndexContext::new(2, 0b01, 0, &[0], &[5, 5]),
        &[(0, EndExpr::End)],
        &mut vars,
    ))
    .expect_err("range/colon conflict should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
}

#[test]
fn apply_end_offsets_rejects_range_dim_conflicting_with_end_mask_in_context() {
    let mut vars = vec![];
    let err = block_on(apply_end_offsets_to_numeric(
        &[Value::Num(1.0)],
        IndexContext::new(2, 0, 0b10, &[1], &[5, 5]),
        &[(0, EndExpr::End)],
        &mut vars,
    ))
    .expect_err("range/end conflict should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
}

#[test]
fn apply_end_offsets_rejects_context_rank_exceeding_mask_width() {
    let mut vars = vec![];
    let err = block_on(apply_end_offsets_to_numeric(
        &[Value::Num(1.0)],
        IndexContext::new(33, 0, 0, &[], &[1; 33]),
        &[(0, EndExpr::End)],
        &mut vars,
    ))
    .expect_err("selector rank beyond mask width should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
}

#[test]
fn apply_cell_end_offsets_rejects_duplicate_positions() {
    let base = Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell base"));
    let err = apply_cell_end_offsets_for_base(&base, &[Value::Num(1.0)], &[(0, 0), (0, 1)], false)
        .expect_err("duplicate cell end-offset positions should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidEndSelectorPlan"));
}

#[test]
fn apply_cell_end_offsets_rejects_out_of_bounds_positions() {
    let base = Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell base"));
    let err = apply_cell_end_offsets_for_base(&base, &[Value::Num(1.0)], &[(1, 0)], false)
        .expect_err("out-of-bounds cell end-offset positions should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidEndSelectorPlan"));
}

#[test]
fn apply_cell_end_exprs_rejects_duplicate_positions() {
    let base = Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell base"));
    let mut vars = vec![];
    let err = block_on(apply_cell_end_exprs_for_base(
        &base,
        &[Value::Num(1.0)],
        &[(0, EndExpr::End), (0, EndExpr::Const(1.0))],
        &mut vars,
        false,
    ))
    .expect_err("duplicate cell end-expression positions should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidEndSelectorPlan"));
}

#[test]
fn apply_cell_end_exprs_rejects_out_of_bounds_positions() {
    let base = Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell base"));
    let mut vars = vec![];
    let err = block_on(apply_cell_end_exprs_for_base(
        &base,
        &[Value::Num(1.0)],
        &[(1, EndExpr::End)],
        &mut vars,
        false,
    ))
    .expect_err("out-of-bounds cell end-expression positions should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidEndSelectorPlan"));
}

#[test]
fn resolve_cell_indices_rejects_fractional_values() {
    let err = block_on(super::resolve_cell_indices(&[Value::Num(1.5)]))
        .expect_err("fractional cell index should fail");
    assert_eq!(err.identifier(), Some("RunMat:CellIndexType"));
}

#[test]
fn resolve_cell_indices_rejects_zero_values() {
    let err = block_on(super::resolve_cell_indices(&[Value::Num(0.0)]))
        .expect_err("zero cell index should fail");
    assert_eq!(err.identifier(), Some("RunMat:CellIndexOutOfBounds"));
}

#[test]
fn resolve_cell_indices_accepts_scalar_tensor_values() {
    let scalar =
        Value::Tensor(runmat_value::Tensor::new(vec![2.0], vec![1, 1]).expect("scalar tensor"));
    let indices =
        block_on(super::resolve_cell_indices(&[scalar])).expect("scalar tensor index should pass");
    assert_eq!(indices, vec![2]);
}

#[test]
fn object_array_scalar_index_returns_object_and_slice_preserves_array() {
    let values = ["first", "second"]
        .into_iter()
        .map(|name| {
            let mut object = ObjectInstance::new("matlab.unittest.TestResult".into());
            object
                .properties
                .insert("Name".into(), Value::String(name.into()));
            Value::Object(object)
        })
        .collect();
    let array = Value::ObjectArray(ObjectArray::row("matlab.unittest.TestResult", values).unwrap());
    let registry = crate::bytecode::FunctionRegistry::default();

    let scalar = block_on(super::paren_index_value(
        array.clone(),
        vec![Value::Num(2.0)],
        1,
        &registry,
    ))
    .unwrap();
    assert!(matches!(
        scalar,
        Value::Object(object)
            if object.properties.get("Name") == Some(&Value::String("second".into()))
    ));

    let selector = Value::Tensor(runmat_value::Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap());
    let slice = block_on(super::paren_index_value(
        array,
        vec![selector],
        1,
        &registry,
    ))
    .unwrap();
    assert!(matches!(
        slice,
        Value::ObjectArray(array)
            if array.shape() == [1, 2] && array.class_name() == "matlab.unittest.TestResult"
    ));
}

#[test]
fn logical_value_from_integer_tensor_uses_native_storage_not_mirror() {
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2])
        .expect("typed integer tensor");

    let value = super::logical_value_from_tensor(tensor).expect("logical conversion");
    assert_eq!(
        value,
        Value::LogicalArray(
            runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical array")
        )
    );
}
