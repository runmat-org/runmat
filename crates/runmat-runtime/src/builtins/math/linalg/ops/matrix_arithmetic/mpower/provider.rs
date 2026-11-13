use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::gpu_helpers::{
    BinaryGpuOutputContract, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};
use crate::BuiltinResult;

pub(super) async fn try_power(base: &Value, exponent: &Value) -> BuiltinResult<Option<Value>> {
    let Value::GpuTensor(base) = base else {
        return Ok(None);
    };
    let provider = gpu_helpers::require_resident_compute_owner(base, || {
        super::errors::internal("mpower: resident base has no exact provider owner")
    })?;
    if !compatible_base(base) {
        return Ok(None);
    }
    let Some(exponent) = super::exponent::parse(exponent)? else {
        return Ok(None);
    };
    if exponent < 0 {
        // The shared inverse kernel owns negative integer powers. The ordinary
        // fallback preserves the selected provider owner when restoring output.
        return Ok(None);
    }
    let [rows, columns] = base.shape() else {
        return Ok(None);
    };
    if rows != columns {
        return Err(super::errors::invalid_input(format!(
            "Matrix must be square for matrix power: {rows}x{columns}"
        )));
    }
    match exponent {
        0 => identity(provider, base)?
            .map(gpu_helpers::resident_gpu_value)
            .transpose(),
        1 => Ok(Some(Value::GpuTensor(base.clone()))),
        exponent => binary_power(provider, base, exponent as u32)
            .await?
            .map(gpu_helpers::resident_gpu_value)
            .transpose(),
    }
}

fn identity(
    provider: &'static dyn AccelProvider,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let output = match provider.eye_like(prototype) {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
            let identity = crate::builtins::common::matrix::matrix_eye(prototype.shape()[0]);
            gpu_helpers::upload_tensor_protecting(provider, &identity, &[prototype])?
        }
        Err(error) => return Err(provider_failure("identity construction", error)),
    };
    let contract = UnaryGpuOutputContract {
        storage: GpuTensorStorage::Real,
        precision: Some(provider.precision()),
        integer: None,
        logical: false,
        alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if gpu_helpers::unary_gpu_output_matches(&output, prototype, provider, contract) {
        let output = gpu_helpers::propagate_owned_output_provenance(output, [prototype])?;
        return Ok(Some(output));
    }
    Err(gpu_helpers::reject_gpu_output(
        output,
        super::errors::internal("mpower: provider returned invalid identity metadata"),
    ))
}

async fn binary_power(
    provider: &'static dyn AccelProvider,
    base: &GpuTensorHandle,
    mut exponent: u32,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let mut base_state = base.clone();
    let mut result: Option<GpuTensorHandle> = None;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = Some(match result {
                Some(current) => {
                    let Some(product) = multiply(provider, &current, &base_state).await? else {
                        return Ok(None);
                    };
                    product
                }
                None => base_state.clone(),
            });
        }
        exponent >>= 1;
        if exponent > 0 {
            let Some(square) = multiply(provider, &base_state, &base_state).await? else {
                return Ok(None);
            };
            base_state = square;
        }
    }
    result
        .map(Some)
        .ok_or_else(|| super::errors::internal("mpower: empty provider power result"))
}

async fn multiply(
    provider: &'static dyn AccelProvider,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let output = match provider.matmul(lhs, rhs).await {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
            return Ok(None);
        }
        Err(error) => return Err(provider_failure("matrix multiplication", error)),
    };
    let contract = BinaryGpuOutputContract {
        shape: lhs.shape().to_vec(),
        storage: GpuTensorStorage::Real,
        precision: Some(provider.precision()),
        integer: None,
        logical: false,
        alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if gpu_helpers::binary_gpu_output_matches(&output, lhs, rhs, provider, &contract) {
        let output = gpu_helpers::propagate_owned_output_provenance(output, [lhs, rhs])?;
        return Ok(Some(output));
    }
    Err(gpu_helpers::reject_gpu_output(
        output,
        super::errors::internal("mpower: provider returned invalid matrix-product metadata"),
    ))
}

fn compatible_base(handle: &GpuTensorHandle) -> bool {
    let Some(provider) = gpu_helpers::exact_provider_for_handle(handle) else {
        return false;
    };
    runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle) == Some(provider.precision())
}

fn provider_failure(operation: &str, source: anyhow::Error) -> crate::RuntimeError {
    let mut error =
        super::errors::internal(format!("mpower: provider {operation} failed: {source}"));
    error = crate::runtime_error::with_provider_execution_source(error, source.into());
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

#[cfg(test)]
#[path = "provider_owning_tests.rs"]
mod owning_tests;
