use super::*;
use std::rc::Rc;

fn runtime() -> runmat_runtime::context::RuntimeContext {
    runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
}

#[test]
fn direct_dispatch_keeps_outputs_and_runtime_errors() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            let runtime = runtime();
            let _scope = runtime.enter();
            let args = [Value::Tensor(
                runmat_value::Tensor::new(vec![1.; 6], vec![2, 3]).unwrap(),
            )];
            let result = runmat_runtime::call_builtin_async_with_outputs("size", &args, 2)
                .await
                .unwrap();
            let Value::OutputList(outputs) = result else {
                panic!("two outputs")
            };
            assert_eq!(outputs, vec![Value::Num(2.), Value::Num(3.)]);
            let error = runmat_runtime::call_builtin_async_with_outputs(
                "__missing_observed_builtin__",
                &[],
                1,
            )
            .await
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
        });
    });
}

#[test]
fn imported_call_uses_runtime_dispatch_with_original_arguments() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            let runtime = runtime();
            let _scope = runtime.enter();
            let args = [Value::Tensor(
                runmat_value::Tensor::new(vec![1.; 6], vec![2, 3]).unwrap(),
            )];
            let result = crate::call::builtins::resolve_imported_builtin(
                "numel",
                &[(vec!["numel".into()], false)],
                &args,
                1,
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
