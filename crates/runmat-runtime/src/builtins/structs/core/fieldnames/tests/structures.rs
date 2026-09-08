use super::{run, strings};
use runmat_value::{CellArray, StructValue, Value};

#[test]
fn preserves_scalar_insertion_order_and_case() {
    let mut structure = StructValue::new();
    structure.fields.insert("beta".into(), Value::Num(1.0));
    structure.fields.insert("Beta".into(), Value::Num(2.0));
    structure.fields.insert("alpha".into(), Value::Num(3.0));
    let (names, shape) = strings(run(Value::Struct(structure)).expect("fieldnames"));
    assert_eq!(names, ["beta", "Beta", "alpha"]);
    assert_eq!(shape, [3, 1]);
}

#[test]
fn collects_sorted_union_from_represented_struct_array() {
    let mut first = StructValue::new();
    first.fields.insert("name".into(), Value::from("Ada"));
    first.fields.insert("id".into(), Value::Num(101.0));
    let mut second = StructValue::new();
    second.fields.insert("name".into(), Value::from("Grace"));
    second
        .fields
        .insert("department".into(), Value::from("Research"));
    let array = CellArray::new_with_shape(
        vec![Value::Struct(first), Value::Struct(second)],
        vec![1, 2],
    )
    .expect("struct array");
    let (names, shape) = strings(run(Value::Cell(array)).expect("fieldnames"));
    assert_eq!(names, ["department", "id", "name"]);
    assert_eq!(shape, [3, 1]);
}

#[test]
fn returns_empty_column_for_empty_represented_struct_array() {
    let array = CellArray::new(Vec::new(), 0, 0).expect("empty struct array");
    let (names, shape) = strings(run(Value::Cell(array)).expect("fieldnames"));
    assert!(names.is_empty());
    assert_eq!(shape, [0, 1]);
}

#[test]
fn reads_only_outer_metadata() {
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
    let (names, _) = strings(run(Value::Struct(structure)).expect("metadata-only lookup"));
    assert_eq!(names, ["resident"]);
}
