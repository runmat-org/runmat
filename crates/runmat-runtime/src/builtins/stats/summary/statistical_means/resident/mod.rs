//! Resident reductions use the input's exact owner and retain all intermediates
//! on that provider. These four functions already documented host
//! materialization of GPU inputs; an unsupported hook preserves that admitted
//! boundary, while an execution failure remains visible.
mod numerics;
mod reduction;
mod rmse;

pub(super) use rmse::try_rmse;

use super::options::{Axes, Options, ReductionKind};
use crate::builtins::common::gpu_helpers;
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_value::{NumericDType, NumericScalar, Tensor, Value};

#[derive(Debug)]
enum Failure {
    Unsupported,
    Execution(Box<RuntimeError>),
}

fn provider_failure(kind: ReductionKind, source: anyhow::Error) -> Failure {
    if gpu_helpers::provider_hook_is_unsupported(&source) {
        return Failure::Unsupported;
    }
    let mut error = super::options::invalid(kind, "resident provider execution failed");
    error = crate::runtime_error::with_provider_execution_source(error, source.into());
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    Failure::Execution(Box::new(error))
}

fn contract_failure(kind: ReductionKind, message: impl Into<String>) -> Failure {
    let message = message.into();
    let mut error = super::options::invalid(kind, &message);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    Failure::Execution(Box::new(error))
}

fn expected_shape(input: &[usize], axes: &Axes) -> Vec<usize> {
    if matches!(axes, Axes::All) {
        return vec![1, 1];
    }
    let mut output = input.to_vec();
    if output.len() < 2 {
        output.resize(2, 1);
    }
    for axis in reduction::selected_axes(input, axes) {
        output[axis] = 1;
    }
    output
}

fn equivalent_shape(left: &[usize], right: &[usize]) -> bool {
    let rank = left.len().max(right.len());
    (0..rank)
        .all(|axis| left.get(axis).copied().unwrap_or(1) == right.get(axis).copied().unwrap_or(1))
}

fn validate_output(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    expected: &[usize],
    precision: ProviderPrecision,
) -> Result<(), Failure> {
    let owner = gpu_helpers::exact_provider_for_handle(output);
    if owner.is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        || output.allocation_lease().validate_live().is_err()
        || output.device_id() != input.device_id()
        || !equivalent_shape(output.shape(), expected)
        || runmat_accelerate_api::handle_storage(output) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_precision(output) != Some(precision)
        || runmat_accelerate_api::handle_integer_type(output).is_some()
        || runmat_accelerate_api::handle_is_logical(output)
        || !gpu_helpers::gpu_class_metadata_matches(output, Some(precision), None, false)
    {
        return Err(contract_failure(
            kind,
            "resident provider returned an invalid output class, shape, or owner",
        ));
    }
    Ok(())
}

async fn scalar_summary(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
) -> Result<f64, Failure> {
    if !input.shape().iter().all(|extent| *extent == 1)
        || input.allocation_lease().validate_live().is_err()
        || gpu_helpers::exact_provider_for_handle(input)
            .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
    {
        return Err(contract_failure(
            kind,
            "resident scalar summary has an invalid shape or owner",
        ));
    }
    let summary = gpu_helpers::gather_tensor_async(input)
        .await
        .map_err(|error| Failure::Execution(Box::new(error)))?;
    if summary.len() != 1
        || input.allocation_lease().validate_live().is_err()
        || gpu_helpers::exact_provider_for_handle(input)
            .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
    {
        return Err(contract_failure(
            kind,
            "resident summary was malformed or changed owner",
        ));
    }
    Ok(
        match summary.numeric_value_at(0).expect("one summary element") {
            NumericScalar::F64(value) => value,
            NumericScalar::F32(value) => f64::from(value),
            _ => {
                return Err(contract_failure(
                    kind,
                    "resident summary changed numeric class",
                ))
            }
        },
    )
}

async fn usable_scale(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    scale: GpuTensorHandle,
) -> Result<GpuTensorHandle, Failure> {
    let owner = gpu_helpers::exact_provider_for_handle(&scale);
    if owner.is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        || scale.allocation_lease().validate_live().is_err()
        || scale.device_id() != input.device_id()
        || runmat_accelerate_api::handle_storage(&scale) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_precision(&scale)
            != runmat_accelerate_api::handle_precision(input)
    {
        return Err(contract_failure(
            kind,
            "resident scale has an invalid class or owner",
        ));
    }
    let summary = gpu_helpers::gather_tensor_async(&scale)
        .await
        .map_err(|error| Failure::Execution(Box::new(error)))?;
    let mut values = summary.materialize_f64();
    if values.is_empty() {
        return Err(Failure::Unsupported);
    }
    let mut changed = false;
    for value in &mut values {
        if !value.is_finite() || *value == 0.0 {
            *value = 1.0;
            changed = true;
        }
    }
    if !changed {
        return Ok(scale);
    }
    let dtype = match runmat_accelerate_api::handle_precision(input) {
        Some(ProviderPrecision::F32) => NumericDType::F32,
        Some(ProviderPrecision::F64) => NumericDType::F64,
        None => return Err(Failure::Unsupported),
    };
    let replacement = Tensor::new_with_dtype(values, summary.shape, dtype)
        .map_err(|source| contract_failure(kind, source))?;
    gpu_helpers::upload_tensor_protecting(provider, &replacement, &[input, &scale])
        .map_err(|error| Failure::Execution(Box::new(error)))
}

pub(super) async fn try_reduce(
    kind: ReductionKind,
    input: &GpuTensorHandle,
    options: &Options,
) -> BuiltinResult<Option<Value>> {
    if input.allocation_lease().validate_live().is_err() {
        return Err(super::options::invalid(
            kind,
            "resident input allocation is not live",
        ));
    }
    let Some(provider) = gpu_helpers::exact_provider_for_handle(input) else {
        return Err(super::options::invalid(
            kind,
            "resident input has no exact owner",
        ));
    };
    let outcome = numerics::run(kind, provider, input, options).await;
    match outcome {
        Ok(mut output) => {
            let precision = runmat_accelerate_api::handle_precision(input).ok_or_else(|| {
                super::options::invalid(kind, "resident input has no floating precision")
            })?;
            let target = expected_shape(input.shape(), &options.axes);
            if output.shape() != target && equivalent_shape(output.shape(), &target) {
                let view = output.view().clone().with_shape(target.clone());
                output = output
                    .with_view(view)
                    .map_err(|source| super::options::invalid(kind, &source.to_string()))?;
            }
            if let Err(Failure::Execution(error)) =
                validate_output(kind, provider, input, &output, &target, precision)
            {
                return Err(gpu_helpers::reject_gpu_output(output, *error));
            }
            if output.shares_allocation_with(input) {
                let error = super::options::invalid(
                    kind,
                    "resident reduction aliased its input allocation",
                );
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            gpu_helpers::propagate_output_provenance(&mut output, [input])?;
            gpu_helpers::resident_gpu_value(output).map(Some)
        }
        // These four APIs historically admit host materialization for GPU
        // inputs, including explicit gpuArray values. Unsupported provider
        // hooks preserve that documented boundary; execution faults do not.
        Err(Failure::Unsupported) => Ok(None),
        Err(Failure::Execution(error)) => Err(*error),
    }
}
