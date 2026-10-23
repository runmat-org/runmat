use runmat_value::Value;

use crate::context::{PreferenceServiceError, RuntimeContext};
use crate::execution::value_codec::{encode_durable_value, validate_durable_preparation};
use crate::execution::RootedValueHandles;
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};

fn after_prior_gather(mut error: RuntimeError, prior_gather: bool) -> RuntimeError {
    if prior_gather {
        error.gpu_gather_retry = GpuGatherRetry::Never;
    }
    error
}

/// Finish all public value preparation before an application-store mutation.
pub(super) async fn durable_values(
    runtime: &RuntimeContext,
    builtin: &'static str,
    values: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    // Direct runtime embedding has no VM frame roots. Keep caller-managed
    // identities alive across every preparation await, including abandonment.
    let _roots = RootedValueHandles::for_values(values.iter()).map_err(|source| {
        super::service::service_error(
            builtin,
            PreferenceServiceError::Storage(
                crate::build_runtime_error("could not retain preference input")
                    .with_source(source)
                    .build(),
            ),
        )
    })?;
    // Reject known-invalid host values throughout the batch before a resident
    // sibling can cause a provider download.
    for value in &values {
        validate_durable_preparation(value).map_err(|source| {
            super::service::service_error(builtin, PreferenceServiceError::ValueEncoding(source))
        })?;
    }
    let mut cooperation = runtime.cpu_cooperation();
    cooperation
        .checkpoint(0)
        .await
        .map_err(|source| source.into_runtime_error())?;
    let mut prepared = Vec::with_capacity(values.len());
    let mut prior_gather = false;
    for value in values {
        let resident = crate::dispatcher::value_contains_gpu(&value);
        let host = cooperation
            .await_cancellable(crate::dispatcher::gather_if_needed_async(&value))
            .await
            .map_err(|source| after_prior_gather(source.into_runtime_error(), prior_gather))?
            .map_err(|source| {
                after_prior_gather(
                    super::controls::map_control_flow(builtin, source),
                    prior_gather,
                )
            })?;
        prior_gather |= resident;
        encode_durable_value(&host).map_err(|source| {
            after_prior_gather(
                super::service::service_error(
                    builtin,
                    PreferenceServiceError::ValueEncoding(source),
                ),
                prior_gather,
            )
        })?;
        prepared.push(host);
        cooperation
            .checkpoint(1)
            .await
            .map_err(|source| after_prior_gather(source.into_runtime_error(), prior_gather))?;
    }
    Ok(prepared)
}
