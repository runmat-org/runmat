//! Complex values must not enter real-only inverse-trigonometric hooks.
use crate::builtins::common::{gpu_helpers, provider_restore};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

#[cfg(test)]
mod tests;

pub(crate) async fn compute(
    input: GpuTensorHandle,
    builtin: &str,
    evaluate: impl FnOnce(Value) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    compute_protecting(input, builtin, &[], evaluate).await
}

pub(crate) async fn compute_protecting(
    input: GpuTensorHandle,
    builtin: &str,
    protected: &[GpuTensorHandle],
    evaluate: impl FnOnce(Value) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    if !gpu_helpers::expected_handle_numeric_element_type(&input).is_ok_and(|element| {
        gpu_helpers::numeric_descriptor_matches(
            &input,
            element,
            GpuTensorStorage::ComplexInterleaved,
        )
    }) {
        return Err(build_runtime_error(format!(
            "{builtin}: malformed or unowned complex GPU input"
        ))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:Internal"))
        .build());
    }
    gpu_helpers::require_resident_compute_owner(&input, || {
        build_runtime_error(format!("{builtin}: malformed or unowned complex GPU input"))
            .with_builtin(builtin)
            .with_identifier(format!("RunMat:{builtin}:Internal"))
            .build()
    })?;
    provider_restore::gather_value_compute_restore_protecting(input, builtin, protected, evaluate)
        .await
}
