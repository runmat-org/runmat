use super::*;
#[test]
fn rdivide_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = RDIVIDE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"C = rdivide(A, B)"));
    assert!(labels.contains(&"C = rdivide(A, B, \"like\", prototype)"));
}

#[test]
fn rdivide_parser_error_has_stable_identifier() {
    let err = rdivide_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
        .expect_err("expected parser error");
    assert_eq!(err.identifier(), RDIVIDE_ERROR_INVALID_ARGUMENT.identifier);
}
