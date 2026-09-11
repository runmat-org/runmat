use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_scalar_struct() {
    let mut st = StructValue::new();
    st.fields.insert("answer".to_string(), Value::Num(42.0));
    let value = run_getfield(Value::Struct(st), vec![Value::from("answer")]).expect("getfield");
    assert_eq!(value, Value::Num(42.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_nested_structs() {
    let mut inner = StructValue::new();
    inner.fields.insert("depth".to_string(), Value::Num(3.0));
    let mut outer = StructValue::new();
    outer
        .fields
        .insert("inner".to_string(), Value::Struct(inner));
    let result = run_getfield(
        Value::Struct(outer),
        vec![Value::from("inner"), Value::from("depth")],
    )
    .expect("nested getfield");
    assert_eq!(result, Value::Num(3.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_struct_array_element() {
    let mut first = StructValue::new();
    first
        .fields
        .insert("name".to_string(), Value::from("entry-a"));
    let mut second = StructValue::new();
    second
        .fields
        .insert("name".to_string(), Value::from("entry-b"));
    let array = StructArray::new(vec![first, second], vec![1, 2]).unwrap();
    let index = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
    let result = run_getfield(
        Value::StructArray(array),
        vec![Value::Cell(index), Value::from("name")],
    )
    .expect("struct array element");
    assert_eq!(result, Value::from("entry-b"));
}

#[test]
fn getfield_index_selectors_read_typed_integer_storage_exactly() {
    let mut first = StructValue::new();
    first
        .fields
        .insert("name".to_string(), Value::from("entry-a"));
    let mut second = StructValue::new();
    second
        .fields
        .insert("name".to_string(), Value::from("entry-b"));
    let array = StructArray::new(vec![first, second], vec![1, 2]).unwrap();
    let index_tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("index tensor");
    let index = CellArray::new_with_shape(vec![Value::Tensor(index_tensor)], vec![1, 1])
        .expect("index cell");

    let result = run_getfield(
        Value::StructArray(array),
        vec![Value::Cell(index), Value::from("name")],
    )
    .expect("struct array element");
    assert_eq!(result, Value::from("entry-b"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_struct_array_defaults_to_first() {
    let mut first = StructValue::new();
    first
        .fields
        .insert("name".to_string(), Value::from("entry-a"));
    let mut second = StructValue::new();
    second
        .fields
        .insert("name".to_string(), Value::from("entry-b"));
    let array = StructArray::new(vec![first, second], vec![1, 2]).unwrap();
    let result =
        run_getfield(Value::StructArray(array), vec![Value::from("name")]).expect("default index");
    assert_eq!(result, Value::from("entry-a"));
}
