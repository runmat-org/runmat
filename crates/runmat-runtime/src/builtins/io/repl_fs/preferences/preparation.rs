use runmat_value::Value;

use crate::context::{PreferenceServiceError, RuntimeContext};
use crate::execution::value_codec::encode_durable_value;
use crate::BuiltinResult;

/// Finish all public value preparation before an application-store mutation.
pub(super) async fn durable_values(
    runtime: &RuntimeContext,
    builtin: &'static str,
    values: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    // Reject known-invalid host values throughout the batch before a resident
    // sibling can cause a provider download.
    for value in &values {
        if !crate::dispatcher::value_contains_gpu(value) {
            encode_durable_value(value).map_err(|source| {
                super::service::service_error(
                    builtin,
                    PreferenceServiceError::ValueEncoding(source),
                )
            })?;
        }
    }
    let mut cooperation = runtime.cpu_cooperation();
    cooperation
        .checkpoint(0)
        .await
        .map_err(|source| source.into_runtime_error())?;
    let mut prepared = Vec::with_capacity(values.len());
    for value in values {
        let host = cooperation
            .await_cancellable(crate::dispatcher::gather_if_needed_async(&value))
            .await
            .map_err(|source| source.into_runtime_error())?
            .map_err(|source| super::controls::map_control_flow(builtin, source))?;
        encode_durable_value(&host).map_err(|source| {
            super::service::service_error(builtin, PreferenceServiceError::ValueEncoding(source))
        })?;
        prepared.push(host);
        cooperation
            .checkpoint(1)
            .await
            .map_err(|source| source.into_runtime_error())?;
    }
    Ok(prepared)
}
