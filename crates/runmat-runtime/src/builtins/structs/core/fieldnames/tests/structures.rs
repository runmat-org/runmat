use super::{run, strings};
use runmat_value::{CellArray, StructArray, StructValue, Value};

#[test]
fn preserves_scalar_insertion_order_and_case() {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.fields.insert("beta".into(), Value::Num(1.0));
    structure.fields.insert("Beta".into(), Value::Num(2.0));
    structure.fields.insert("alpha".into(), Value::Num(3.0));
    let (names, shape) = strings(run(Value::Struct(structure.finish())).expect("fieldnames"));
    assert_eq!(names, ["beta", "Beta", "alpha"]);
    assert_eq!(shape, [3, 1]);
}

#[test]
fn reads_ordered_schema_from_typed_struct_array() {
    let mut first = runmat_value::UnboundedStructValueBuilder::new();
    first.fields.insert("name".into(), Value::from("first"));
    first.fields.insert("id".into(), Value::Num(101.0));
    let mut second = runmat_value::UnboundedStructValueBuilder::new();
    second.fields.insert("name".into(), Value::from("second"));
    second.fields.insert("id".into(), Value::Num(102.0));
    let array =
        StructArray::new(vec![first.finish(), second.finish()], vec![1, 2]).expect("struct array");
    let (names, shape) = strings(run(Value::StructArray(array)).expect("fieldnames"));
    assert_eq!(names, ["name", "id"]);
    assert_eq!(shape, [2, 1]);
}

#[test]
fn empty_typed_array_retains_its_schema() {
    let array = StructArray::empty(vec!["name".into(), "id".into()], vec![0, 3]).unwrap();
    let (names, shape) = strings(run(Value::StructArray(array)).expect("fieldnames"));
    assert_eq!(names, ["name", "id"]);
    assert_eq!(shape, [2, 1]);
}

#[test]
fn ordinary_cell_of_structs_is_not_a_structure_array() {
    let cell = CellArray::new(vec![Value::Struct(StructValue::new())], 1, 1).unwrap();
    assert!(run(Value::Cell(cell)).is_err());
}

#[test]
fn reads_only_outer_metadata() {
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
    let (names, _) = strings(run(Value::Struct(structure.finish())).expect("metadata-only lookup"));
    assert_eq!(names, ["resident"]);
}
