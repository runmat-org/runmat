use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_rejects_cell_content_traversal() {
    let mut inner1 = StructValue::new();
    inner1.fields.insert("value".to_string(), Value::Num(1.0));
    let mut inner2 = StructValue::new();
    inner2.fields.insert("value".to_string(), Value::Num(2.0));
    let cell = CellArray::new_with_shape(
        vec![Value::Struct(inner1), Value::Struct(inner2)],
        vec![1, 2],
    )
    .unwrap();
    let mut root = StructValue::new();
    root.fields.insert("samples".to_string(), Value::Cell(cell));

    let index_cell =
        CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
    let error = run_setfield(
        Value::Struct(root),
        vec![
            Value::from("samples"),
            Value::Cell(index_cell),
            Value::from("value"),
            Value::Num(10.0),
        ],
    )
    .expect_err("parenthesis indexing must preserve the cell container");
    assert_eq!(
        error.identifier(),
        SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT.identifier
    );
}

#[test]
fn setfield_cell_parentheses_assignment_uses_column_major_subscripts() {
    let cell = CellArray::new_with_shape(
        vec![
            Value::from("r1c1"),
            Value::from("r1c2"),
            Value::from("r2c1"),
            Value::from("r2c2"),
        ],
        vec![2, 2],
    )
    .expect("cell target");
    let mut root = StructValue::new();
    root.fields.insert("samples".into(), Value::Cell(cell));
    let selector = CellArray::new_with_shape(
        vec![Value::Int(IntValue::I32(2)), Value::Int(IntValue::I32(1))],
        vec![1, 2],
    )
    .expect("selector");
    let replacement =
        CellArray::new_with_shape(vec![Value::from("updated")], vec![1, 1]).expect("replacement");

    let updated = run_setfield(
        Value::Struct(root),
        vec![
            Value::from("samples"),
            Value::Cell(selector),
            Value::Cell(replacement),
        ],
    )
    .expect("cell parenthesis assignment");
    let Value::Struct(root) = updated else {
        panic!("expected structure result");
    };
    let Value::Cell(cell) = root.fields.get("samples").expect("samples field") else {
        panic!("expected cell field");
    };
    assert_eq!(cell.data[2], Value::from("updated"));
    assert_eq!(cell.data[1], Value::from("r1c2"));
}
