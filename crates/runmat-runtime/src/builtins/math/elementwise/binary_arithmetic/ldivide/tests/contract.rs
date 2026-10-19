use super::*;
#[test]
fn ldivide_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = LDIVIDE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"C = ldivide(A, B)"));
    assert!(labels.contains(&"C = ldivide(A, B, \"like\", prototype)"));
}

#[test]
fn ldivide_parser_error_has_stable_identifier() {
    let err = ldivide_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
        .expect_err("expected parser error");
    assert_eq!(err.identifier(), LDIVIDE_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn ldivide_like_is_gated_in_matlab_mode() {
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = ldivide_builtin(
        Value::Num(2.0),
        Value::Num(5.0),
        vec![Value::from("like"), Value::Num(0.0)],
    )
    .expect_err("like form must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:LdivideLikePrototypeExtension")
    );
}
