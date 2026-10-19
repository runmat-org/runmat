use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_converts_string_scalars_and_arrays() {
    assert_eq!(
        double_builtin(Value::String(" 12.5 ".into()), Vec::new()).unwrap(),
        Value::Num(12.5)
    );
    let invalid = double_builtin(Value::String("not a number".into()), Vec::new()).unwrap();
    assert!(matches!(invalid, Value::Num(value) if value.is_nan()));

    let strings = StringArray::new(
        vec!["1".into(), "-2.25".into(), "missing".into(), "Inf".into()],
        vec![2, 2],
    )
    .unwrap();
    let Value::Tensor(output) = double_builtin(Value::StringArray(strings), Vec::new()).unwrap()
    else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape(), &[2, 2]);
    let values = output.materialize_f64();
    assert_eq!(values[0..2], [1.0, -2.25]);
    assert!(values[2].is_nan());
    assert_eq!(values[3], f64::INFINITY);
}
