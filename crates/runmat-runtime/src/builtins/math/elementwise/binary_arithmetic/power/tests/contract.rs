use super::*;
#[test]
fn power_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = POWER_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"C = power(A, B)"));
    assert!(labels.contains(&"C = power(A, B, \"like\", prototype)"));
}

#[test]
fn power_parser_error_has_stable_identifier() {
    let err = power_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
        .expect_err("expected parser error");
    assert_eq!(err.identifier(), POWER_ERROR_INVALID_ARGUMENT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_like_missing_prototype_errors() {
    let err = power_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
        .expect_err("expected error");
    assert!(err.message().contains("prototype"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_like_extra_arguments_error() {
    let err = power_builtin(
        Value::Num(1.0),
        Value::Num(2.0),
        vec![Value::from("like"), Value::Num(1.0), Value::Num(2.0)],
    )
    .expect_err("expected error");
    assert!(err.message().contains("too many"));
}
