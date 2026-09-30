//! Execution after implementation selection, shared by normal and admitted calls.
use super::*;

/// The old inventory remains separate until its L23 removal. Canonical
/// implementations have exactly one context-bearing ABI.
#[derive(Clone, Copy)]
pub(super) enum SelectedImplementation {
    Canonical(crate::builtin::RuntimeBuiltinImplementation),
    Legacy(fn(&[Value]) -> runmat_builtins::BuiltinFuture),
}

impl SelectedImplementation {
    fn invoke(
        self,
        args: &[Value],
        context: Option<crate::context::RuntimeContext>,
        requested_outputs: Option<usize>,
    ) -> crate::builtin::RuntimeBuiltinFuture {
        match self {
            Self::Canonical(implementation) => implementation(
                crate::builtin::RuntimeBuiltinInvocation::new(args, context, requested_outputs),
            ),
            Self::Legacy(implementation) => implementation(args),
        }
    }
}

pub(super) async fn execute(
    implementation: SelectedImplementation,
    execution: &[Value],
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
) -> Result<Value, RuntimeError> {
    let result = match implementation
        .invoke(execution, context.clone(), requested_outputs)
        .await
    {
        Ok(result) => result,
        Err(error) if should_retry_with_gpu_gather(&error, execution) => {
            if let Some(recorder) =
                runmat_accelerate_api::observation::current_provider_observation()
            {
                recorder.execution_disposition(runmat_accelerate_api::observation::ProviderExecutionDisposition::SharedRuntimeFallback);
            }
            match gather_args_for_retry_async(execution).await? {
                Some(gathered) => {
                    implementation
                        .invoke(&gathered, context, requested_outputs)
                        .await?
                }
                None => return Err(error),
            }
        }
        Err(error) => return Err(error),
    };
    Ok(result)
}
