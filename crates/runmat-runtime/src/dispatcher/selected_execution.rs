//! Execution after implementation selection, shared by normal and admitted calls.
use super::*;

pub(super) async fn execute(
    implementation: crate::builtin::RuntimeBuiltinImplementation,
    execution: &[Value],
) -> Result<Value, RuntimeError> {
    let result = match implementation(execution).await {
        Ok(result) => result,
        Err(error) if should_retry_with_gpu_gather(&error, execution) => {
            if let Some(recorder) =
                runmat_accelerate_api::observation::current_provider_observation()
            {
                recorder.execution_disposition(runmat_accelerate_api::observation::ProviderExecutionDisposition::SharedRuntimeFallback);
            }
            match gather_args_for_retry_async(execution).await? {
                Some(gathered) => implementation(&gathered).await?,
                None => return Err(error),
            }
        }
        Err(error) => return Err(error),
    };
    Ok(result)
}
