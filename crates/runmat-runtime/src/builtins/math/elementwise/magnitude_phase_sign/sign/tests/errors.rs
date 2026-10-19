use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_string_input_errors() {
    let err = sign_builtin(Value::String("runmat".to_string())).unwrap_err();
    assert!(
        err.message()
            .contains("expected numeric, logical, or character input"),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_complex_with_nan() {
    let result = sign_builtin(Value::Complex(f64::NAN, 1.0)).unwrap();
    match result {
        Value::Complex(re, im) => {
            assert!(re.is_nan());
            assert!(im.is_nan());
        }
        other => panic!("expected complex result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_rejects_string_with_stable_identifier() {
    let err = sign_builtin(Value::from("bad")).expect_err("expected error");
    assert_eq!(err.identifier(), SIGN_ERROR_INVALID_INPUT.identifier);
}
