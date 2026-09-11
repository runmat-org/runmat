use super::*;
use std::sync::Arc;

#[test]
fn passes_requested_output_count_to_bound_callbacks() {
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_, arguments, outputs| {
            assert_eq!(outputs, 2);
            let Value::Num(value) = arguments[0] else {
                panic!("expected numeric field")
            };
            Box::pin(async move {
                crate::sequence::ValueSequence::comma_separated(vec![
                    Value::Num(value),
                    Value::Num(value + 10.0),
                ])
                .map_err(crate::sequence::sequence_error_to_runtime)
            })
        },
    )));
    let _outputs = crate::output_count::push_output_count(Some(2));
    let result = call(
        Value::BoundFunctionHandle {
            name: "two_outputs".into(),
            function: 91,
        },
        numbers(),
        Vec::new(),
    )
    .unwrap();
    let Value::OutputList(outputs) = result else {
        panic!("expected output list")
    };
    assert_eq!(outputs.len(), 2);
    let Value::Tensor(first) = &outputs[0] else {
        panic!("expected tensor")
    };
    let Value::Tensor(second) = &outputs[1] else {
        panic!("expected tensor")
    };
    assert_eq!(first.materialize_f64(), vec![1.0, 2.0]);
    assert_eq!(second.materialize_f64(), vec![11.0, 12.0]);
}

#[test]
fn callback_errors_are_wrapped_without_text_rewriting() {
    let error = call(
        Value::FunctionHandle("missing_structfun_callback".into()),
        numbers(),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::STRUCTFUN_ERROR_FUNCTION_ERROR.identifier
    );
    assert!(error.message().contains("Undefined function"));
    assert!(!error.message().contains("cellfun:"));
}

#[test]
fn error_handler_receives_the_original_callback_identity() {
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, _| {
            let context = arguments.first().cloned();
            crate::sequence::single_value_future(async move {
                match function {
                    101 => Err(crate::build_runtime_error("field callback failed")
                        .with_identifier("RunMat:Test:OriginalCallback")
                        .build()),
                    102 => {
                        let Some(Value::Struct(context)) = context else {
                            panic!("expected callback error context")
                        };
                        assert_eq!(
                            context.fields.get("identifier"),
                            Some(&Value::String("RunMat:Test:OriginalCallback".into()))
                        );
                        assert_eq!(
                            context.fields.get("field"),
                            Some(&Value::String("a".into()))
                        );
                        Ok(Value::Num(7.0))
                    }
                    _ => panic!("unexpected semantic callback"),
                }
            })
        },
    )));
    let mut structure = StructValue::new();
    structure.insert("a", Value::Num(1.0));
    let result = call(
        Value::BoundFunctionHandle {
            name: "fails".into(),
            function: 101,
        },
        structure,
        vec![
            Value::String("ErrorHandler".into()),
            Value::BoundFunctionHandle {
                name: "recovers".into(),
                function: 102,
            },
        ],
    )
    .unwrap();
    let Value::Tensor(result) = result else {
        panic!("expected uniform output")
    };
    assert_eq!(result.materialize_f64(), vec![7.0]);
}
