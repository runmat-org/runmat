use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[path = "tests/native.rs"]
mod native;
#[cfg(target_arch = "wasm32")]
#[path = "tests/wasm.rs"]
mod wasm;

fn run_pyargs(values: Vec<Value>) -> BuiltinResult<Value> {
    let arguments = crate::call::BuiltinArguments::from_borrowed(&values, None)?;
    pyargs_builtin(None, arguments).map(|value| value.into_value_and_lease().0)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pyargs_constructs_local_bundle_without_foreign_service() {
    let bundle = run_pyargs(vec![
        Value::String("count".into()),
        Value::Int(runmat_value::IntValue::I64(7)),
    ])
    .expect("bundle");
    let Value::Object(bundle) = bundle else {
        panic!("expected object")
    };
    assert!(bundle
        .class_name()
        .is(runmat_types::standard::PYTHON_ARGUMENTS));
    let Value::Cell(names) = bundle.properties().get("Names").expect("Names") else {
        panic!("names")
    };
    let Value::Cell(values) = bundle.properties().get("Values").expect("Values") else {
        panic!("values")
    };
    assert_eq!(names.rows, 1);
    assert_eq!(names.cols, 1);
    assert_eq!(names.data(), vec![Value::String("count".into())]);
    assert_eq!(
        values.data(),
        vec![Value::Int(runmat_value::IntValue::I64(7))]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pyargs_accepts_effective_character_rows_and_scalar_string_arrays() {
    let row = runmat_value::CharArray::from_unicode_scalars_with_shape(
        vec!['n', 'a', 'm', 'e'],
        vec![1, 4, 1],
    )
    .expect("trailing singleton row");
    let scalar = runmat_value::StringArray::new(vec!["other".into()], vec![1, 1])
        .expect("scalar string array");
    let bundle = run_pyargs(vec![
        Value::CharArray(row),
        Value::Num(1.0),
        Value::StringArray(scalar),
        Value::Bool(true),
    ])
    .expect("admitted text names");
    let Value::Object(bundle) = bundle else {
        panic!("expected Python arguments object");
    };
    let Value::Cell(names) = bundle.properties().get("Names").expect("Names") else {
        panic!("names cell");
    };
    assert_eq!(
        names.data(),
        vec![Value::String("name".into()), Value::String("other".into())]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pyargs_rejects_invalid_pairs_before_any_python_service() {
    let odd = run_pyargs(vec![Value::String("x".into())]).expect_err("odd pair");
    assert_eq!(odd.identifier(), Some("RunMat:Foreign:InvalidCall"));
    for name in [
        Value::Num(1.0),
        Value::Int(runmat_value::IntValue::I64(1)),
        Value::Bool(true),
        Value::String(String::new().into()),
    ] {
        let nontext = run_pyargs(vec![name, Value::Num(2.0)]).expect_err("nonempty text name");
        assert_eq!(nontext.identifier(), Some("RunMat:Foreign:InvalidCall"));
    }
}
