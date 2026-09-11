use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_errors_when_indexing_missing_field() {
    let struct_value = StructValue::new();
    let index_cell =
        CellArray::new_with_shape(vec![Value::Int(IntValue::I32(1))], vec![1, 1]).unwrap();
    let error = run_setfield(
        Value::Struct(struct_value),
        vec![
            Value::from("missing"),
            Value::Cell(index_cell),
            Value::Num(1.0),
        ],
    )
    .expect_err("setfield should fail when field is missing");
    let err = error.message().to_string();
    assert!(
        err.contains("Reference to non-existent field 'missing'."),
        "unexpected error message: {err}"
    );
    assert_eq!(error.identifier(), SETFIELD_ERROR_MISSING_FIELD.identifier);
}

#[test]
fn setfield_maps_shared_index_bounds_to_its_catalog_error() {
    let structures = StructArray::new(vec![StructValue::new(), StructValue::new()], vec![1, 2])
        .expect("structure array");
    let selector = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(3))], vec![1, 1])
        .expect("selector");
    let error = run_setfield(
        Value::StructArray(structures),
        vec![
            Value::Cell(selector),
            Value::from("values"),
            Value::Num(20.0),
        ],
    )
    .expect_err("out-of-bounds assignment");
    assert_eq!(
        error.identifier(),
        SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_undefined_detection_requires_identifier() {
    let with_identifier = build_runtime_error("missing")
        .with_identifier(crate::IDENT_UNDEFINED_FUNCTION)
        .build();
    assert!(is_undefined_function(&with_identifier));

    let message_only =
        build_runtime_error(format!("{} message only", crate::IDENT_UNDEFINED_FUNCTION)).build();
    assert!(
        !is_undefined_function(&message_only),
        "message-only undefined markers should not trigger setter fallback"
    );
}
