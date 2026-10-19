use super::*;
use runmat_builtins::CELLFUN_ERROR_UNDEFINED_FUNCTION;

#[test]
fn error_handler_receives_callback_failure() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 419);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, _| {
            assert_eq!(function, 419);
            let Value::Struct(context) = &arguments[1] else {
                panic!("expected error context")
            };
            assert!(context.fields.contains_key("identifier"));
            let seed = arguments[0].clone();
            Box::pin(async move {
                crate::sequence::single_value_sequence(seed).and_then(rooted_sequence)
            })
        },
    )))
    .expect("call-port scope");
    let handler = Value::Closure(Closure::from_data(runmat_value::ClosureData {
        lexical_environment: None,
        function_name: "cellfun_handler_fixture".into(),
        binding: Some(binding),
        source_role: runmat_types::FunctionSourceRole::Unknown,
        source_file: None,
        captures: vec![Value::Num(7.0)].into(),
    }));
    let result = call(
        Value::FunctionHandle("missing_cellfun_callback".into()),
        vec![
            cell(vec![Value::Num(1.0)], &[1, 1]),
            Value::String("ErrorHandler".into()),
            handler,
        ],
    )
    .unwrap();
    assert_eq!(tensor_values(result.value()), vec![7.0]);
}

#[test]
fn unresolved_qualified_callback_reports_typed_external_identity() {
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_| None)))
            .expect("call-port scope");
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
