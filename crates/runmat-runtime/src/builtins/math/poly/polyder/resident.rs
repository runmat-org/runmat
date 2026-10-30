//! Validate whole polynomial operations before exact-owner dispatch.
use super::{ensure_vector_shape, orientation_from_shape, polyder_error, PolyderEval};
use crate::builtins::common::gpu_helpers;
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderExecutionError,
};
use runmat_value::Value;

#[cfg(test)]
#[path = "resident_tests.rs"]
mod tests;

fn owner(inputs: &[&GpuTensorHandle]) -> BuiltinResult<Option<&'static dyn AccelProvider>> {
    for input in inputs {
        ensure_vector_shape("polyder", "coefficient", input.shape())?;
    }
    gpu_helpers::require_resident_compute_owners(inputs, false, || {
        polyder_error("polyder: no exact owner for resident input")
    })?;
    let provider = gpu_helpers::require_resident_compute_owner(inputs[0], || {
        polyder_error("polyder: no exact owner for resident input")
    })?;
    for input in inputs {
        let actual = gpu_helpers::require_resident_compute_owner(input, || {
            polyder_error("polyder: no exact owner for resident input")
        })?;
        if !runmat_accelerate_api::same_provider_instance(actual, provider)
            || input.physical().storage() != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(input)
            || runmat_accelerate_api::handle_precision(input) != Some(provider.precision())
        {
            return Ok(None);
        }
    }
    Ok(Some(provider))
}

fn optional<T>(result: anyhow::Result<T>) -> BuiltinResult<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error)
            if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(&error) =>
        {
            Ok(None)
        }
        Err(error) => Err(crate::runtime_error::with_provider_execution_source(
            polyder_error(format!("polyder: provider execution failed: {error}")),
            ProviderExecutionError::from(error),
        )),
    }
}

fn length(handle: &GpuTensorHandle) -> BuiltinResult<usize> {
    if handle.shape().contains(&0) {
        return Ok(0);
    }
    handle
        .shape()
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| polyder_error("polyder: polynomial size overflow"))
}

fn combined_length(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    derivative: bool,
) -> BuiltinResult<usize> {
    length(left)?
        .max(1)
        .checked_add(length(right)?.max(1))
        .map(|n| n.saturating_sub(if derivative { 2 } else { 1 }).max(1))
        .ok_or_else(|| polyder_error("polyder: output size overflow"))
}

fn valid(output: &GpuTensorHandle, inputs: &[&GpuTensorHandle], max_len: usize) -> bool {
    let Ok(len) = length(output) else {
        return false;
    };
    len > 0
        && len <= max_len
        && ensure_vector_shape("polyder", "provider output", output.shape()).is_ok()
        && output.physical().storage() == GpuTensorStorage::Real
        && output.physical().element_type() == inputs[0].physical().element_type()
        && !runmat_accelerate_api::handle_is_logical(output)
        && !runmat_accelerate_api::handle_is_transposed(output)
        && gpu_helpers::exact_provider_for_handle(output)
            .zip(gpu_helpers::exact_provider_for_handle(inputs[0]))
            .is_some_and(|(left, right)| runmat_accelerate_api::same_provider_instance(left, right))
        && inputs
            .iter()
            .all(|input| !output.shares_allocation_with(input))
}

fn adopt(
    mut output: GpuTensorHandle,
    inputs: &[&GpuTensorHandle],
    orientation: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    let shape = orientation_from_shape(orientation.shape()).shape_for_len(length(&output)?);
    if output.shape() != shape {
        output = output
            .with_view(output.view().clone().with_shape(shape))
            .map_err(|error| {
                crate::build_runtime_error(format!("polyder: invalid result view: {error}"))
                    .with_source(error)
                    .build()
            })?;
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, inputs.iter().copied())?;
    gpu_helpers::resident_gpu_value(output)
}

fn invalid(output: GpuTensorHandle) -> RuntimeError {
    gpu_helpers::reject_gpu_output(output, polyder_error("polyder: invalid provider output"))
}

pub(super) async fn try_gpu_derivative_single(value: &Value) -> BuiltinResult<Option<Value>> {
    let Value::GpuTensor(input) = value else {
        return Ok(None);
    };
    let Some(provider) = owner(&[input])? else {
        return Ok(None);
    };
    let max_len = length(input)?.saturating_sub(1).max(1);
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(input);
    let Some(output) = optional(provider.polyder_single(input).await)? else {
        return Ok(None);
    };
    if !valid(&output, &[input], max_len) {
        return Err(invalid(output));
    }
    adopt(output, &[input], input).map(Some)
}

pub(super) async fn try_gpu_derivative_product(
    left: &Value,
    right: &Value,
) -> BuiltinResult<Option<Value>> {
    let (Value::GpuTensor(left), Value::GpuTensor(right)) = (left, right) else {
        return Ok(None);
    };
    let Some(provider) = owner(&[left, right])? else {
        return Ok(None);
    };
    let max_len = combined_length(left, right, true)?;
    let _residency = [left, right].map(gpu_helpers::ResidencyRestoreGuard::new);
    let Some(output) = optional(provider.polyder_product(left, right).await)? else {
        return Ok(None);
    };
    if !valid(&output, &[left, right], max_len) {
        return Err(invalid(output));
    }
    adopt(output, &[left, right], left).map(Some)
}

pub(super) async fn try_gpu_quotient(
    left: &Value,
    right: &Value,
) -> BuiltinResult<Option<PolyderEval>> {
    let (Value::GpuTensor(left), Value::GpuTensor(right)) = (left, right) else {
        return Ok(None);
    };
    let Some(provider) = owner(&[left, right])? else {
        return Ok(None);
    };
    let numerator_len = combined_length(left, right, true)?;
    let denominator_len = combined_length(right, right, false)?;
    let _residency = [left, right].map(gpu_helpers::ResidencyRestoreGuard::new);
    let Some(result) = optional(provider.polyder_quotient(left, right).await)? else {
        return Ok(None);
    };
    if !valid(&result.numerator, &[left, right], numerator_len)
        || !valid(&result.denominator, &[left, right], denominator_len)
        || result.numerator.shares_allocation_with(&result.denominator)
    {
        return Err(gpu_helpers::reject_gpu_output(
            result.denominator,
            invalid(result.numerator),
        ));
    }
    Ok(Some(PolyderEval {
        numerator: adopt(result.numerator, &[left, right], left)?,
        denominator: adopt(result.denominator, &[left, right], right)?,
    }))
}
