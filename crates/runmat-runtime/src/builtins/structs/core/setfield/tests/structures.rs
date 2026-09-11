use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_creates_scalar_field() {
    let struct_value = StructValue::new();
    let updated = run_setfield(
        Value::Struct(struct_value),
        vec![Value::from("answer"), Value::Num(42.0)],
    )
    .expect("setfield");
    match updated {
        Value::Struct(st) => {
            assert_eq!(
                st.fields.get("answer"),
                Some(&Value::Num(42.0)),
                "field should be inserted"
            );
        }
        other => panic!("expected struct result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_creates_nested_structs() {
    let struct_value = StructValue::new();
    let updated = run_setfield(
        Value::Struct(struct_value),
        vec![
            Value::from("solver"),
            Value::from("name"),
            Value::from("cg"),
        ],
    )
    .expect("setfield");
    match updated {
        Value::Struct(st) => {
            let solver = st.fields.get("solver").expect("solver field");
            match solver {
                Value::Struct(inner) => {
                    assert_eq!(
                        inner.fields.get("name"),
                        Some(&Value::from("cg")),
                        "inner field should exist"
                    );
                }
                other => panic!("expected inner struct, got {other:?}"),
            }
        }
        other => panic!("expected struct result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_updates_struct_array_element() {
    let mut a = StructValue::new();
    a.fields
        .insert("id".to_string(), Value::Int(IntValue::I32(1)));
    let mut b = StructValue::new();
    b.fields
        .insert("id".to_string(), Value::Int(IntValue::I32(2)));
    let array = StructArray::new(vec![a, b], vec![1, 2]).unwrap();
    let indices =
        CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
    let updated = run_setfield(
        Value::StructArray(array),
        vec![
            Value::Cell(indices),
            Value::from("id"),
            Value::Int(IntValue::I32(42)),
        ],
    )
    .expect("setfield");
    match updated {
        Value::StructArray(array) => assert_eq!(
            array.get_linear(1).unwrap().fields.get("id"),
            Some(&Value::Int(IntValue::I32(42)))
        ),
        other => panic!("expected structure array, got {other:?}"),
    }
}

#[test]
fn setfield_rejects_nonscalar_leading_structure_selection() {
    let array = StructArray::new(vec![StructValue::new(), StructValue::new()], vec![1, 2])
        .expect("structure array");
    let indices = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).expect("indices");
    let selector =
        CellArray::new_with_shape(vec![Value::Tensor(indices)], vec![1, 1]).expect("selector");
    let error = run_setfield(
        Value::StructArray(array),
        vec![Value::Cell(selector), Value::from("value"), Value::Num(1.0)],
    )
    .expect_err("leading selection must be scalar");
    assert_eq!(error.identifier(), SETFIELD_ERROR_INDEX_SHAPE.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_struct_array_with_end_index() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut first = StructValue::new();
    first
        .fields
        .insert("id".to_string(), Value::Int(IntValue::I32(1)));
    let mut second = StructValue::new();
    second
        .fields
        .insert("id".to_string(), Value::Int(IntValue::I32(2)));
    let array = StructArray::new(vec![first, second], vec![1, 2]).unwrap();
    let index_cell = CellArray::new_with_shape(vec![Value::from("end")], vec![1, 1]).unwrap();
    let updated = run_setfield(
        Value::StructArray(array),
        vec![
            Value::Cell(index_cell),
            Value::from("id"),
            Value::Int(IntValue::I32(99)),
        ],
    )
    .expect("setfield");
    match updated {
        Value::StructArray(array) => assert_eq!(
            array.get_linear(1).unwrap().fields.get("id"),
            Some(&Value::Int(IntValue::I32(99)))
        ),
        other => panic!("expected structure array result, got {other:?}"),
    }
}
