use super::*;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pyargs_constructs_local_bundle_without_foreign_service() {
    let bundle = pyargs_builtin(vec![
        Value::String("count".into()),
        Value::Int(runmat_value::IntValue::I64(7)),
    ])
    .expect("bundle");
    let Value::Object(bundle) = bundle else {
        panic!("expected object")
    };
    assert!(bundle
        .class_name
        .is(runmat_types::standard::PYTHON_ARGUMENTS));
    let Value::Cell(names) = bundle.properties.get("Names").expect("Names") else {
        panic!("names")
    };
    let Value::Cell(values) = bundle.properties.get("Values").expect("Values") else {
        panic!("values")
    };
    assert_eq!(names.rows, 1);
    assert_eq!(names.cols, 1);
    assert_eq!(names.data, vec![Value::String("count".into())]);
    assert_eq!(
        values.data,
        vec![Value::Int(runmat_value::IntValue::I64(7))]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pyargs_rejects_invalid_pairs_before_any_python_service() {
    let odd = pyargs_builtin(vec![Value::String("x".into())])
        .err()
        .expect("odd pair");
    assert_eq!(odd.identifier(), Some("RunMat:Foreign:InvalidCall"));
    let nontext = pyargs_builtin(vec![Value::Num(1.0), Value::Num(2.0)])
        .err()
        .expect("text name");
    assert_eq!(nontext.identifier(), Some("RunMat:Foreign:InvalidCall"));
}
