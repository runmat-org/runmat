use super::run;
use runmat_value::{CellArray, StructArray, StructValue, Value};

#[test]
fn scalar_structure_queries_are_exact_and_case_sensitive() {
    let mut structure = StructValue::new();
    structure.fields.insert("name".into(), Value::from("entry"));
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
    let mut first = StructValue::new();
    first.fields.insert("name".into(), Value::from("first"));
    first.fields.insert("id".into(), Value::Num(1.0));
    let mut second = StructValue::new();
    second.fields.insert("name".into(), Value::from("second"));
    second.fields.insert("id".into(), Value::Num(2.0));
    let array = StructArray::new(vec![first, second], vec![1, 2]).expect("structure array");
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
    let mut structure = StructValue::new();
    structure.fields.insert(
        "resident".into(),
        Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        }),
    );
    assert_eq!(
        run(Value::Struct(structure), Value::from("resident")).unwrap(),
        Value::Bool(true)
    );
}
