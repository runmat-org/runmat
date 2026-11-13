use super::*;
use runmat_builtins::{CONTAINERS_MAP_DESCRIPTOR, CONTAINERS_MAP_SUBSREF_DESCRIPTOR};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn construct_empty_map_defaults() {
    let map = containers_map_builtin(Vec::new()).expect("map");
    let count = containers_map_subsref(
        map.clone(),
        ".".to_string(),
        Value::from("Count".to_string()),
    )
    .expect("Count");
    assert_eq!(count, Value::Int(IntValue::U64(0)));

    let key_type = containers_map_subsref(
        map.clone(),
        ".".to_string(),
        Value::from("KeyType".to_string()),
    )
    .expect("KeyType");
    assert_eq!(
        key_type,
        Value::CharArray(CharArray::from_unicode_scalars("char".chars().collect(), 1, 4).unwrap())
    );

    let value_type = containers_map_subsref(
        map.clone(),
        ".".to_string(),
        Value::from("ValueType".to_string()),
    )
    .expect("ValueType");
    assert_eq!(
        value_type,
        Value::CharArray(CharArray::from_unicode_scalars("any".chars().collect(), 1, 3).unwrap())
    );
}

#[test]
fn containers_map_descriptor_includes_constructor_and_method_signatures() {
    let constructor_labels: Vec<&str> = CONTAINERS_MAP_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(constructor_labels.contains(&"M = containers.Map()"));
    assert!(constructor_labels.contains(&"M = containers.Map(keys, values)"));

    let method_labels: Vec<&str> = CONTAINERS_MAP_SUBSREF_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(method_labels.contains(&"value = containers.Map.subsref(M, S)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn constructor_with_cells_lookup() {
    let keys = crate::make_cell(vec![Value::from("apple"), Value::from("pear")], 1, 2).unwrap();
    let values = crate::make_cell(vec![Value::Num(5.0), Value::Num(7.0)], 1, 2).unwrap();
    let map = containers_map_builtin(vec![keys, values]).expect("map");
    let apple = containers_map_subsref(
        map.clone(),
        "()".to_string(),
        crate::make_cell(vec![Value::from("apple")], 1, 1).unwrap(),
    )
    .expect("lookup");
    assert_eq!(apple, Value::Num(5.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn constructor_rejects_duplicate_keys() {
    let keys = crate::make_cell(vec![Value::from("dup"), Value::from("dup")], 1, 2).unwrap();
    let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
    let err = containers_map_builtin(vec![keys, values]).expect_err("duplicate check");
    let message = error_message(err);
    assert!(message.contains("Duplicate key name"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn constructor_errors_when_value_count_mismatch() {
    let keys = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
    let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
    let err = containers_map_builtin(vec![keys, values]).expect_err("count mismatch");
    let message = error_message(err);
    assert!(message.contains("number of keys"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn comparison_method_rejects_unknown_values() {
    let keys = crate::make_cell(vec![Value::from("a")], 1, 1).unwrap();
    let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
    let err = containers_map_builtin(vec![
        keys,
        values,
        Value::from("ComparisonMethod"),
        Value::from("caseinsensitive"),
    ])
    .expect_err("comparison method");
    let message = error_message(err);
    assert!(message.contains("unrecognised option"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn key_type_single_roundtrip() {
    let map = containers_map_builtin(vec![
        Value::from("KeyType"),
        Value::from("single"),
        Value::from("ValueType"),
        Value::from("any"),
    ])
    .expect("map");
    let key_type = containers_map_subsref(map.clone(), ".".to_string(), Value::from("KeyType"))
        .expect("keytype");
    assert_eq!(
        key_type,
        Value::CharArray(
            CharArray::from_unicode_scalars("single".chars().collect(), 1, 6).unwrap()
        )
    );

    let single_key = Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap());
    let payload = crate::make_cell(vec![single_key], 1, 1).unwrap();
    let map = containers_map_subsasgn(map, "()".to_string(), payload.clone(), Value::Num(7.0))
        .expect("assign");
    let value = map
        .with_owned(|map| containers_map_subsref(map, "()".to_string(), payload))
        .expect("lookup");
    assert!(matches!(value.value(), Value::Num(n) if (n - 7.0).abs() < 1e-12));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn value_type_double_converts_integers() {
    let map = containers_map_builtin(vec![
        Value::from("KeyType"),
        Value::from("double"),
        Value::from("ValueType"),
        Value::from("double"),
    ])
    .expect("map");
    let payload = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
    let map = containers_map_subsasgn(
        map,
        "()".to_string(),
        payload.clone(),
        Value::Int(IntValue::I32(7)),
    )
    .expect("assign");
    let value = map
        .with_owned(|map| containers_map_subsref(map, "()".to_string(), payload))
        .expect("lookup");
    assert!(matches!(value.value(), Value::Num(n) if (n - 7.0).abs() < 1e-12));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn value_type_logical_rejects_nonscalar_numeric_arrays() {
    let tensor = Tensor::new(vec![0.0, 2.0, -3.0], vec![3, 1]).unwrap();
    let map = containers_map_builtin(vec![
        Value::from("KeyType"),
        Value::from("char"),
        Value::from("ValueType"),
        Value::from("logical"),
    ])
    .expect("map");
    let payload = crate::make_cell(vec![Value::from("mask")], 1, 1).unwrap();
    let error = containers_map_subsasgn(map, "()".to_string(), payload, Value::Tensor(tensor))
        .expect_err("declared logical values must be scalar");
    assert!(error.to_string().contains("logical scalars"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn uniform_values_enforced_on_assignment() {
    let map = containers_map_builtin(vec![
        crate::make_cell(vec![Value::from("x")], 1, 1).unwrap(),
        crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap(),
    ])
    .expect("map");
    let payload = crate::make_cell(vec![Value::from("x")], 1, 1).unwrap();
    let err = containers_map_subsasgn(map, "()".to_string(), payload, Value::from("text"))
        .expect_err("uniform enforcement");
    let message = error_message(err);
    assert!(message.contains("numeric scalar"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn assignment_updates_and_inserts() {
    let map = containers_map_builtin(Vec::new()).expect("map");
    let payload = crate::make_cell(vec![Value::from("alpha")], 1, 1).unwrap();
    let updated = containers_map_subsasgn(
        map.clone(),
        "()".to_string(),
        payload.clone(),
        Value::Num(1.0),
    )
    .expect("assign");
    let updated = updated
        .with_owned(|map| {
            containers_map_subsasgn(map, "()".to_string(), payload.clone(), Value::Num(5.0))
        })
        .expect("update");
    let beta_payload = crate::make_cell(vec![Value::from("beta")], 1, 1).unwrap();
    let updated = updated
        .with_owned(|map| {
            containers_map_subsasgn(map, "()".to_string(), beta_payload, Value::Num(9.0))
        })
        .expect("insert");
    let value = updated
        .with_owned(|map| containers_map_subsref(map, "()".to_string(), payload))
        .expect("lookup");
    assert_eq!(value, Value::Num(5.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn subsref_rejects_multiple_keys() {
    let keys = crate::make_cell(
        vec![Value::from("a"), Value::from("b"), Value::from("c")],
        1,
        3,
    )
    .unwrap();
    let values = crate::make_cell(
        vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
        1,
        3,
    )
    .unwrap();
    let map = containers_map_builtin(vec![keys, values]).expect("map");
    let request = crate::make_cell(vec![Value::from("a"), Value::from("c")], 1, 2).unwrap();
    let payload = crate::make_cell(vec![request], 1, 1).unwrap();
    let error = containers_map_subsref(map, "()".to_string(), payload).unwrap_err();
    assert!(error.to_string().contains("exactly one scalar key"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn subsref_rejects_empty_key_collection() {
    let keys = crate::make_cell(vec![Value::from("z")], 1, 1).unwrap();
    let values = crate::make_cell(vec![Value::Num(42.0)], 1, 1).unwrap();
    let map = containers_map_builtin(vec![keys, values]).expect("map");
    let empty_keys = crate::make_cell(Vec::new(), 1, 0).unwrap();
    let payload = crate::make_cell(vec![empty_keys], 1, 1).unwrap();
    let error = containers_map_subsref(map, "()".to_string(), payload).unwrap_err();
    assert!(error.to_string().contains("exactly one scalar key"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn subsasgn_rejects_multiple_keys() {
    let keys = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
    let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
    let map = containers_map_builtin(vec![keys, values]).expect("map");
    let key_spec = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
    let payload = crate::make_cell(vec![key_spec], 1, 1).unwrap();
    let new_values = crate::make_cell(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).unwrap();
    let error = containers_map_subsasgn(map, "()".to_string(), payload, new_values).unwrap_err();
    assert!(error.to_string().contains("exactly one scalar key"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn assignment_value_count_mismatch_errors() {
    let keys = crate::make_cell(vec![Value::from("x"), Value::from("y")], 1, 2).unwrap();
    let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
    let map = containers_map_builtin(vec![keys, values]).expect("map");
    let key_spec = crate::make_cell(vec![Value::from("x"), Value::from("y")], 1, 2).unwrap();
    let payload = crate::make_cell(vec![key_spec], 1, 1).unwrap();
    let rhs = crate::make_cell(vec![Value::Num(99.0)], 1, 1).unwrap();
    let err =
        containers_map_subsasgn(map, "()".to_string(), payload, rhs).expect_err("value count");
    let message = error_message(err);
    assert!(message.contains("exactly one scalar key"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn subsasgn_rejects_empty_key_collection() {
    let keys = crate::make_cell(vec![Value::from("root")], 1, 1).unwrap();
    let values = crate::make_cell(vec![Value::Num(7.0)], 1, 1).unwrap();
    let map = containers_map_builtin(vec![keys, values]).expect("map");
    let empty_keys = crate::make_cell(Vec::new(), 1, 0).unwrap();
    let payload = crate::make_cell(vec![empty_keys], 1, 1).unwrap();
    let rhs = crate::make_cell(Vec::new(), 1, 0).unwrap();
    let error = containers_map_subsasgn(map, "()".to_string(), payload, rhs).unwrap_err();
    assert!(error.to_string().contains("exactly one scalar key"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn keys_values_iskey_remove() {
    let keys = crate::make_cell(
        vec![Value::from("a"), Value::from("b"), Value::from("c")],
        1,
        3,
    )
    .unwrap();
    let values = crate::make_cell(
        vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
        1,
        3,
    )
    .unwrap();
    let map = containers_map_builtin(vec![keys, values]).expect("map");
    let key_list = containers_map_keys(map.clone()).expect("keys");
    match key_list {
        Value::Cell(cell) => assert_eq!(cell.data().len(), 3),
        other => panic!("expected cell array, got {other:?}"),
    }
    let mask = containers_map_is_key(
        map.clone(),
        crate::make_cell(vec![Value::from("a"), Value::from("z")], 1, 2).unwrap(),
    )
    .expect("mask");
    match mask {
        Value::LogicalArray(arr) => {
            assert_eq!(arr.data, vec![1, 0]);
        }
        other => panic!("expected logical array, got {:?}", other),
    }
    let removed = containers_map_remove(
        map.clone(),
        crate::make_cell(vec![Value::from("b")], 1, 1).unwrap(),
    )
    .expect("remove");
    let mask = containers_map_is_key(
        removed,
        crate::make_cell(vec![Value::from("b")], 1, 1).unwrap(),
    )
    .expect("mask");
    assert_eq!(mask, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn remove_missing_key_returns_error() {
    let keys = crate::make_cell(vec![Value::from("key")], 1, 1).unwrap();
    let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
    let map = containers_map_builtin(vec![keys, values]).expect("map");
    let err = containers_map_remove(
        map,
        crate::make_cell(vec![Value::from("missing")], 1, 1).unwrap(),
    )
    .expect_err("remove missing");
    assert_eq!(
        err.identifier(),
        CONTAINERS_MAP_ERROR_MISSING_KEY.identifier
    );
    let message = error_message(err);
    assert_eq!(message, CONTAINERS_MAP_ERROR_MISSING_KEY.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn length_delegates_to_map_count() {
    let keys = crate::make_cell(
        vec![Value::from("a"), Value::from("b"), Value::from("c")],
        1,
        3,
    )
    .unwrap();
    let values = crate::make_cell(
        vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
        1,
        3,
    )
    .unwrap();
    let map = containers_map_builtin(vec![keys, values]).expect("map");
    assert_eq!(map_length(&map), Some(3));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn map_handle_rejects_missing_typed_payload() {
    let target = runmat_gc::gc_allocate(Value::Object(ObjectInstance::new(CLASS_IDENTITY)))
        .expect("storage");
    let handle = Value::HandleObject(HandleRef {
        class_name: CLASS_IDENTITY.owned().into(),
        target,
        valid: true,
    });
    let error = with_store(&handle, BUILTIN_CONSTRUCTOR, |store| Ok(store.len()))
        .expect_err("missing typed payload must reject");
    assert_eq!(error.identifier(), CONTAINERS_MAP_ERROR_INTERNAL.identifier);
    assert_eq!(map_length(&handle), None);
}
