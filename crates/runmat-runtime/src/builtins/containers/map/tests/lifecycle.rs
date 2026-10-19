use super::*;

fn key(name: &str) -> Value {
    crate::make_cell(vec![Value::from(name)], 1, 1).unwrap()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn failed_multi_key_remove_preserves_all_entries() {
    let keys = crate::make_cell(vec![Value::from("present"), Value::from("other")], 1, 2).unwrap();
    let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
    let map = containers_map_builtin(vec![keys, values]).unwrap();
    let remove =
        crate::make_cell(vec![Value::from("present"), Value::from("missing")], 1, 2).unwrap();
    assert!(containers_map_remove(map.clone(), remove).is_err());
    assert_eq!(
        containers_map_subsref(map, "()".into(), key("present")).unwrap(),
        Value::Num(1.0)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn duplicate_multi_key_remove_is_atomic() {
    let map = containers_map_builtin(vec![
        key("present"),
        crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap(),
    ])
    .unwrap();
    let remove =
        crate::make_cell(vec![Value::from("present"), Value::from("present")], 1, 2).unwrap();
    assert!(containers_map_remove(map.clone(), remove).is_err());
    assert_eq!(
        containers_map_subsref(map, "()".into(), key("present")).unwrap(),
        Value::Num(1.0)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn assignment_rejects_transient_value_without_changing_map() {
    let map = containers_map_builtin(Vec::new()).unwrap();
    let result = containers_map_subsasgn(
        map.clone(),
        "()".into(),
        key("invalid"),
        Value::output_list(vec![Value::Num(1.0)]),
    );
    assert!(result.is_err());
    assert_eq!(
        containers_map_is_key(map, key("invalid")).unwrap(),
        Value::Bool(false)
    );
}
