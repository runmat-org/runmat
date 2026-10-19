use super::{run, structure};
use runmat_value::Value;

#[test]
fn variadic_field_arguments_follow_compatibility_mode() {
    let target = || structure(&[("a", Value::Num(1.0)), ("b", Value::Num(2.0))]);
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = run(target(), vec![Value::from("a"), Value::from("b")]).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:RmfieldVariadicExtension")
    );
    drop(strict);

    let enabled = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = run(target(), vec![Value::from("a"), Value::from("b")]).unwrap();
    assert!(matches!(result, Value::Struct(value) if value.fields.is_empty()));
    drop(enabled);
}
