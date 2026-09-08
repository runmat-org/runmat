use super::*;
use runmat_builtins::CELLFUN_ERROR_UNDEFINED_FUNCTION;
use runmat_value::Closure;
use std::sync::Arc;

#[test]
fn bound_callback_uses_semantic_identity() {
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, outputs| {
            assert_eq!(function, 417);
            assert_eq!(outputs, 1);
            let [Value::Num(seed), Value::Num(value)] = arguments else {
                panic!("unexpected callback arguments: {arguments:?}");
            };
            let result = seed + value;
            Box::pin(async move { Ok(Value::Num(result)) })
        },
    )));
    let callback = Value::Closure(Closure {
        function_name: "cellfun_bound_fixture".into(),
        bound_function: Some(417),
        captures: vec![Value::Num(10.0)],
    });
    let result = call(
        callback,
        vec![cell(vec![Value::Num(1.0), Value::Num(2.0)], &[1, 2])],
    )
    .unwrap();
    assert_eq!(tensor_values(result), vec![11.0, 12.0]);
}

#[test]
fn name_callback_resolves_to_semantic_function() {
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "project_callback").then_some(418)
        })));
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, _| {
            assert_eq!(function, 418);
            let [Value::Num(value)] = arguments else {
                panic!("unexpected arguments")
            };
            let result = value * 2.0;
            Box::pin(async move { Ok(Value::Num(result)) })
        },
    )));
    let result = call(
        Value::String("project_callback".into()),
        vec![cell(vec![Value::Num(3.0)], &[1, 1])],
    )
    .unwrap();
    assert_eq!(tensor_values(result), vec![6.0]);
}

#[test]
fn error_handler_receives_callback_failure() {
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, _| {
            assert_eq!(function, 419);
            let Value::Struct(context) = &arguments[1] else {
                panic!("expected error context")
            };
            assert!(context.fields.contains_key("identifier"));
            let seed = arguments[0].clone();
            Box::pin(async move { Ok(seed) })
        },
    )));
    let handler = Value::Closure(Closure {
        function_name: "cellfun_handler_fixture".into(),
        bound_function: Some(419),
        captures: vec![Value::Num(7.0)],
    });
    let result = call(
        Value::FunctionHandle("missing_cellfun_callback".into()),
        vec![
            cell(vec![Value::Num(1.0)], &[1, 1]),
            Value::String("ErrorHandler".into()),
            handler,
        ],
    )
    .unwrap();
    assert_eq!(tensor_values(result), vec![7.0]);
}

#[test]
fn unresolved_qualified_callback_reports_typed_external_identity() {
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_| None)));
    let error = call(
        Value::ExternalFunctionHandle("pkg.callback".into()),
        vec![cell(vec![Value::Num(1.0)], &[1, 1])],
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        CELLFUN_ERROR_UNDEFINED_FUNCTION.identifier
    );
    assert!(error.message().contains("ExternalName(QualifiedName"));
}
