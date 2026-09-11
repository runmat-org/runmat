use super::*;

#[test]
fn getfield_integer_capabilities_cover_payloads_and_structural_indices() {
    assert_eq!(GETFIELD_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(GETFIELD_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    assert_eq!(GETFIELD_INTEGER_CAPABILITIES[1].inputs[0].classes.len(), 8);
    assert_eq!(
        GETFIELD_INTEGER_CAPABILITIES[0].output_class,
        BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert_eq!(
        GETFIELD_INTEGER_CAPABILITIES[1].overload,
        BuiltinIntegerOverloadKind::StructuralParameter
    );
}

#[test]
fn getfield_rejects_nonscalar_intermediate_structure_selection() {
    let mut first = StructValue::new();
    first.fields.insert("name".into(), Value::from("first"));
    let mut second = StructValue::new();
    second.fields.insert("name".into(), Value::from("second"));
    let array = StructArray::new(vec![first, second], vec![1, 2]).expect("structure array");
    let indices = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).expect("indices");
    let selector =
        CellArray::new_with_shape(vec![Value::Tensor(indices)], vec![1, 1]).expect("selector");

    let error = run_getfield(
        Value::StructArray(array),
        vec![Value::Cell(selector), Value::from("name")],
    )
    .expect_err("intermediate selection must be scalar");
    assert_eq!(error.identifier(), GETFIELD_ERROR_INDEX_SHAPE.identifier);
}

#[test]
fn getfield_missing_field_errors() {
    let error = run_getfield(
        Value::Struct(StructValue::new()),
        vec![Value::from("missing")],
    )
    .expect_err("missing field");
    assert_eq!(error.identifier(), GETFIELD_ERROR_MISSING_FIELD.identifier);
}

#[test]
fn getfield_maps_shared_index_bounds_to_its_catalog_error() {
    let mut structure = StructValue::new();
    structure.fields.insert(
        "values".to_string(),
        Value::Tensor(Tensor::new(vec![10.0], vec![1, 1]).expect("tensor")),
    );
    let selector = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1])
        .expect("selector");
    let error = run_getfield(
        Value::Struct(structure),
        vec![Value::from("values"), Value::Cell(selector)],
    )
    .expect_err("out-of-bounds access");
    assert_eq!(
        error.identifier(),
        GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn indexing_missing_field_name_fails() {
    let mut outer = StructValue::new();
    outer.fields.insert("inner".to_string(), Value::Num(1.0));
    let index = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(1))], vec![1, 1]).unwrap();
    let err =
        error_message(run_getfield(Value::Struct(outer), vec![Value::Cell(index)]).unwrap_err());
    assert!(err.contains("expected field name"));
}

#[test]
fn getfield_undefined_detection_requires_identifier() {
    let with_identifier = build_runtime_error("missing")
        .with_identifier(crate::IDENT_UNDEFINED_FUNCTION)
        .build();
    assert!(is_undefined_function(&with_identifier));

    let message_only =
        build_runtime_error(format!("{} message only", crate::IDENT_UNDEFINED_FUNCTION)).build();
    assert!(!is_undefined_function(&message_only));
}
