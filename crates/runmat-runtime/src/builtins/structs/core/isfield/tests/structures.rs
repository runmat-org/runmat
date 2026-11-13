use super::run;
use runmat_value::{CellArray, StructArray, Value};

#[test]
fn scalar_structure_queries_are_exact_and_case_sensitive() {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.fields.insert("name".into(), Value::from("entry"));
    let structure = structure.finish();
    assert_eq!(
        run(Value::Struct(structure.clone()), Value::from("name")).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run(Value::Struct(structure), Value::from("Name")).unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn typed_array_uses_its_homogeneous_schema() {
    let mut first = runmat_value::UnboundedStructValueBuilder::new();
    first.fields.insert("name".into(), Value::from("first"));
    first.fields.insert("id".into(), Value::Num(1.0));
    let mut second = runmat_value::UnboundedStructValueBuilder::new();
    second.fields.insert("name".into(), Value::from("second"));
    second.fields.insert("id".into(), Value::Num(2.0));
    let array = StructArray::new(vec![first.finish(), second.finish()], vec![1, 2])
        .expect("structure array");
    assert_eq!(
        run(Value::StructArray(array.clone()), Value::from("name")).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run(Value::StructArray(array), Value::from("missing")).unwrap(),
        Value::Bool(false),
    );
}

#[test]
fn non_structure_and_empty_cell_targets_return_false() {
    assert_eq!(
        run(Value::Num(5.0), Value::from("field")).unwrap(),
        Value::Bool(false)
    );
    let empty = CellArray::new(Vec::new(), 0, 0).expect("empty");
    assert_eq!(
        run(Value::Cell(empty), Value::from("field")).unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn nested_resident_values_are_not_gathered() {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.fields.insert(
        "resident".into(),
        Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("authentic unregistered allocation")
                .handle,
        ),
    );
    assert_eq!(
        run(Value::Struct(structure.finish()), Value::from("resident")).unwrap(),
        Value::Bool(true)
    );
}
