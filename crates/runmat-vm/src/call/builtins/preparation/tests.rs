use super::*;
use std::rc::Rc;

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
}

#[test]
fn prepared_call_keeps_outputs_and_runtime_errors() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            let runtime = runtime();
            let args = [Value::Tensor(
                runmat_value::Tensor::new(vec![1.; 6], vec![2, 3]).unwrap(),
            )];
            let mut prepared = prepare_builtin_args("size", &args, &runtime).await.unwrap();
            let result = call_prepared_builtin("size", &mut prepared, 2)
                .await
                .unwrap();
            let Value::OutputList(outputs) = result else {
                panic!("two outputs")
            };
            assert_eq!(outputs, vec![Value::Num(2.), Value::Num(3.)]);
            let mut prepared = prepare_builtin_args("__missing_observed_builtin__", &[], &runtime)
                .await
                .unwrap();
            let error = call_prepared_builtin("__missing_observed_builtin__", &mut prepared, 1)
                .await
                .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
        });
    });
}

#[test]
fn imported_call_uses_the_same_prepared_invocation_boundary() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            let runtime = runtime();
            let args = [Value::Tensor(
                runmat_value::Tensor::new(vec![1.; 6], vec![2, 3]).unwrap(),
            )];
            let result = crate::call::builtins::resolve_imported_builtin(
                "numel",
                &[(vec!["numel".into()], false)],
                &args,
                1,
                &runtime,
            )
            .await
            .unwrap();
            let crate::call::builtins::ImportedBuiltinResolution::Resolved(value) = result else {
                panic!("resolved import")
            };
            assert_eq!(value, Value::Num(6.));
        });
    });
}
