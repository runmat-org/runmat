use runmat_builtins::{
    RETHROW_ACTIVE_CATCH_EXTENSION, RETHROW_ERROR_INVALID_ARITY, RETHROW_ERROR_NO_ACTIVE_CATCH,
    RETHROW_TEXT_EXTENSION,
};
use runmat_value::Value;

#[runmat_macros::runtime_builtin(
    name = "rethrow",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::diagnostics::rethrow"
)]
pub async fn rethrow_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &RETHROW_ACTIVE_CATCH_EXTENSION,
                "rethrow",
            )?;
            let exception = runtime
                .as_ref()
                .and_then(|runtime| runtime.active_catch_exception())
                .ok_or_else(|| {
                    crate::build_runtime_error(RETHROW_ERROR_NO_ACTIVE_CATCH.message)
                        .with_identifier(
                            RETHROW_ERROR_NO_ACTIVE_CATCH
                                .identifier
                                .expect("identifier"),
                        )
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?;
            Err(crate::runtime_error::error_from_exception(exception))
        }
        1 => match args.into_iter().next().expect("one rethrow argument") {
            e @ Value::MException(_) => crate::rethrow_builtin(e).await,
            e @ Value::String(_) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &RETHROW_TEXT_EXTENSION,
                    "rethrow",
                )?;
                crate::rethrow_builtin(e).await
            }
            other => Err(
                crate::build_runtime_error(format!("RunMat:error: {other:?}"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build(),
            ),
        },
        _ => Err(
            crate::build_runtime_error(RETHROW_ERROR_INVALID_ARITY.message)
                .with_identifier(RETHROW_ERROR_INVALID_ARITY.identifier.expect("identifier"))
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use futures::executor::block_on;
    use runmat_value::MException;
    use std::rc::Rc;

    async fn rethrow_builtin_registered(
        runtime: Option<crate::context::RuntimeContext>,
        values: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        let values = match runtime.as_ref() {
            Some(runtime) => crate::execution::RootedValues::in_context(values, runtime)?,
            None => crate::execution::RootedValues::new(values)
                .map_err(|error| error.into_runtime_error())?,
        };
        super::rethrow_builtin_registered(runtime, values.into()).await
    }

    fn runtime() -> crate::context::RuntimeContext {
        crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn shorthand_uses_active_catch_and_rejects_stale_context() {
        let runtime = runtime();
        runtime.set_runmat_extensions_enabled(true);
        block_on(runtime.scope(async {
            let original = MException::new("RunMat:original".into(), "original".into());
            let guard = runtime.enter_active_catch(Some(original));
            let error = rethrow_builtin_registered(Some(runtime.clone()), vec![])
                .await
                .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:original"));
            drop(guard);
            let stale = rethrow_builtin_registered(Some(runtime.clone()), vec![])
                .await
                .unwrap_err();
            assert_eq!(stale.identifier(), Some("RunMat:rethrow:NoActiveCatch"));
        }));
    }

    #[test]
    fn extension_forms_are_gated_and_arity_is_stable() {
        let runtime = runtime();
        block_on(runtime.scope(async {
            let shorthand = rethrow_builtin_registered(Some(runtime.clone()), vec![])
                .await
                .unwrap_err();
            assert_eq!(
                shorthand.identifier(),
                Some("RunMat:compatibility:RethrowActiveCatchExtension")
            );
            let text = rethrow_builtin_registered(
                Some(runtime.clone()),
                vec![Value::String("new".into())],
            )
            .await
            .unwrap_err();
            assert_eq!(
                text.identifier(),
                Some("RunMat:compatibility:RethrowTextExtension")
            );
            let arity = rethrow_builtin_registered(
                Some(runtime.clone()),
                vec![Value::Num(1.0), Value::Num(2.0)],
            )
            .await
            .unwrap_err();
            assert_eq!(arity.identifier(), Some("RunMat:rethrow:InvalidArity"));
        }));
    }
}
