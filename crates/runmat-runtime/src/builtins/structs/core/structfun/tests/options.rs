use super::*;

#[test]
fn parses_supported_uniform_controls() {
    for value in [
        Value::Bool(false),
        Value::Num(0.0),
        Value::Int(runmat_value::IntValue::I32(0)),
        Value::String("off".into()),
    ] {
        let Value::Struct(_) = call(
            Value::FunctionHandle("class".into()),
            numbers(),
            vec![Value::String("UniformOutput".into()), value],
        )
        .unwrap() else {
            panic!("expected nonuniform structure")
        };
    }
}

#[test]
fn rejects_incomplete_and_unknown_options() {
    let incomplete = call(
        Value::FunctionHandle("class".into()),
        numbers(),
        vec![Value::String("UniformOutput".into())],
    )
    .unwrap_err();
    assert_eq!(
        incomplete.identifier(),
        runmat_builtins::STRUCTFUN_ERROR_INVALID_INPUT.identifier
    );
    let unknown = call(
        Value::FunctionHandle("class".into()),
        numbers(),
        vec![Value::String("Unknown".into()), Value::Bool(true)],
    )
    .unwrap_err();
    assert_eq!(
        unknown.identifier(),
        runmat_builtins::STRUCTFUN_ERROR_INVALID_INPUT.identifier
    );
}
