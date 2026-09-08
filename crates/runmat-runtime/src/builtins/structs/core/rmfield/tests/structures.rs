use super::{run, structure};
use runmat_value::{CellArray, IntegerStorage, Tensor, Value};

#[test]
fn scalar_removal_preserves_the_original_and_retained_values() {
    let original = structure(&[("name", Value::from("Ada")), ("score", Value::Num(42.0))]);
    let result = run(original.clone(), vec![Value::from("score")]).unwrap();
    assert!(matches!(&original, Value::Struct(value) if value.fields.contains_key("score")));
    assert!(
        matches!(result, Value::Struct(value) if value.fields.len() == 1 && value.fields.contains_key("name"))
    );
}

#[test]
fn represented_array_preserves_shape_and_updates_every_element() {
    let first = structure(&[("name", Value::from("Ada")), ("score", Value::Num(90.0))]);
    let second = structure(&[("name", Value::from("Grace")), ("score", Value::Num(95.0))]);
    let array = CellArray::new_with_shape(vec![first, second], vec![1, 2]).unwrap();
    let result = run(Value::Cell(array), vec![Value::from("score")]).unwrap();
    let Value::Cell(array) = result else {
        panic!("expected represented structure array");
    };
    assert_eq!(array.shape, vec![1, 2]);
    assert!(array.data.iter().all(|value| matches!(value, Value::Struct(structure) if structure.fields.len() == 1 && structure.fields.contains_key("name"))));
}

#[test]
fn represented_array_reports_the_first_requested_name_missing_from_any_element() {
    let first = structure(&[("a", Value::Num(1.0)), ("b", Value::Num(2.0))]);
    let second = structure(&[("b", Value::Num(3.0))]);
    let array = CellArray::new(vec![first, second], 1, 2).unwrap();
    let names = CellArray::new(vec![Value::from("a"), Value::from("missing")], 1, 2).unwrap();
    let error = run(Value::Cell(array), vec![Value::Cell(names)]).unwrap_err();
    assert!(error.message().contains("non-existent field 'a'"));
}

#[test]
fn retained_integer_classes_remain_exact() {
    let storages = [
        IntegerStorage::I8(vec![i8::MIN]),
        IntegerStorage::I16(vec![i16::MIN]),
        IntegerStorage::I32(vec![i32::MIN]),
        IntegerStorage::I64(vec![i64::MIN]),
        IntegerStorage::U8(vec![u8::MAX]),
        IntegerStorage::U16(vec![u16::MAX]),
        IntegerStorage::U32(vec![u32::MAX]),
        IntegerStorage::U64(vec![u64::MAX]),
    ];
    for storage in storages {
        let tensor = Tensor::new_integer(storage.clone(), vec![1, 1]).unwrap();
        let target = structure(&[("keep", Value::Tensor(tensor)), ("drop", Value::Num(1.0))]);
        let result = run(target, vec![Value::from("drop")]).unwrap();
        assert!(
            matches!(result, Value::Struct(value) if matches!(value.fields.get("keep"), Some(Value::Tensor(tensor)) if tensor.integer_storage() == Some(&storage)))
        );
    }
}

#[test]
fn nested_resident_values_are_not_gathered() {
    let handle = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    };
    let target = structure(&[
        ("keep", Value::GpuTensor(handle.clone())),
        ("drop", Value::Num(1.0)),
    ]);
    let result = run(target, vec![Value::from("drop")]).unwrap();
    assert!(
        matches!(result, Value::Struct(value) if matches!(value.fields.get("keep"), Some(Value::GpuTensor(retained)) if retained == &handle))
    );
}
