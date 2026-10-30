use super::*;
use runmat_runtime::execution::{DeferredCall, DeferredInvocation, DeferredProgram};
use runmat_value::Value;

#[test]
fn retained_builtin_selects_real_numeric_worker_program_and_keeps_runtime_failure() {
    futures::executor::block_on(async {
        #[cfg(target_arch = "wasm32")]
        runmat_runtime::builtins::wasm_registry::register_all();
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let registry = FunctionRegistry::new(std::collections::HashMap::new());
        for (name, arguments) in [
            ("logical", vec![Value::Num(1.)]),
            (
                "error",
                vec![
                    Value::from("RunMat:WorkerBuiltinProbe"),
                    Value::from("originating failure"),
                ],
            ),
        ] {
            let target = runmat_runtime::execution::RootedValue::in_context(
                Value::function_handle(name.into()),
                &runtime,
            )
            .unwrap();
            let selected = select_worker_callable(&runtime, target, &registry);
            assert!(
                selected.is_ok(),
                "normal builtin must reach worker admission"
            );
            let call = DeferredCall {
                spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
                invocation: DeferredInvocation::Program {
                    callable: runmat_execution::ProgramCallable::builtin(name).unwrap(),
                    context: runmat_execution::ProgramInvocationContext::Direct,
                    arguments,
                    requested_outputs: 1,
                },
                retry: runmat_execution::RetryPolicy::Never,
                program_revision: None,
                environment: Default::default(),
                console: None,
                capabilities: runmat_builtins::builtin_required_capabilities(name),
                program: DeferredProgram::InterpreterV2(
                    capture_builtin_program(&runtime, &registry).unwrap(),
                ),
            };
            let result = crate::program_execution::execute_deferred_program_in_context(
                call,
                runtime.clone(),
            )
            .await;
            if name == "logical" {
                assert_eq!(result.unwrap().as_slice(), &[Value::Bool(true)]);
            } else {
                let error = result.unwrap_err();
                assert!(matches!(
                    error,
                    runmat_runtime::execution::ExecutionServiceError::RuntimeFailure(_)
                ));
                let error = error.into_runtime_error_in_context(Some(&runtime));
                assert_eq!(error.identifier(), Some("RunMat:WorkerBuiltinProbe"));
            }
        }
    });
}
