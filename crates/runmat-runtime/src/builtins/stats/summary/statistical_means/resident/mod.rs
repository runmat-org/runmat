//! Resident reductions use the input's exact owner and retain all intermediates
//! on that provider. These four functions already documented host
//! materialization of GPU inputs; an unsupported hook preserves that admitted
//! boundary, while an execution failure remains visible.
mod numerics;
mod progress;
mod reduction;
mod rmse;
#[cfg(test)]
mod progress_tests;

use progress::{ObserveResidentHook, ResidentStageProgress};

pub(super) use rmse::try_rmse;

use super::options::{Axes, Options, ReductionKind};
use crate::builtins::common::gpu_helpers;
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_value::Value;

#[derive(Debug)]
enum Failure {
    Unsupported,
    Execution(Box<RuntimeError>),
}

fn provider_failure(
    kind: ReductionKind,
    source: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> Failure {
    let source = source.into();
    if source.is_unsupported() {
        return Failure::Unsupported;
    }
    let mut error = super::options::invalid(kind, "resident provider execution failed");
    error = crate::runtime_error::with_provider_execution_source(error, source);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    Failure::Execution(Box::new(error))
}

fn contract_failure(kind: ReductionKind, message: impl Into<String>) -> Failure {
    let message = message.into();
    let mut error = super::options::invalid(kind, &message);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    Failure::Execution(Box::new(error))
}

fn reject_output(
    kind: ReductionKind,
    output: GpuTensorHandle,
    message: impl Into<String>,
) -> Failure {
    let mut error = super::options::invalid(kind, &message.into());
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    Failure::Execution(Box::new(gpu_helpers::reject_gpu_output(output, error)))
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

fn validate_mask(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    mask: &GpuTensorHandle,
    expected: &[usize],
) -> Result<(), Failure> {
    // A validated logical view may use U8, F32, or F64 physical lanes.
    let descriptor_valid =
        gpu_helpers::expected_handle_numeric_element_type(mask).is_ok_and(|element| {
            matches!(
                element,
                runmat_accelerate_api::NumericElementType::U8
                    | runmat_accelerate_api::NumericElementType::F32
                    | runmat_accelerate_api::NumericElementType::F64
            ) && gpu_helpers::numeric_descriptor_matches(mask, element, GpuTensorStorage::Real)
        }) && mask.view().validate(mask.physical()).is_ok()
            && gpu_helpers::gpu_class_metadata_matches(mask, None, None, true);
    if mask.allocation_lease().validate_live().is_err()
        || mask.device_id() != input.device_id()
        || gpu_helpers::exact_provider_for_handle(mask)
            .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        || !equivalent_shape(mask.shape(), expected)
        || !runmat_accelerate_api::handle_is_logical(mask)
        || !descriptor_valid
        || mask.shares_allocation_with(input)
    {
        return Err(contract_failure(
            kind,
            "resident provider returned an invalid logical mask",
        ));
    }
    Ok(())
}

fn admitted_real(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    expected: &[usize],
    precision: ProviderPrecision,
) -> Result<GpuTensorHandle, Failure> {
    match validate_output(kind, provider, input, &output, expected, precision) {
        Ok(()) => Ok(output),
        Err(Failure::Execution(error)) => Err(Failure::Execution(Box::new(
            gpu_helpers::reject_gpu_output(output, *error),
        ))),
        Err(Failure::Unsupported) => Err(Failure::Unsupported),
    }
}

fn admitted_mask(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    expected: &[usize],
) -> Result<GpuTensorHandle, Failure> {
    match validate_mask(kind, provider, input, &output, expected) {
        Ok(()) => Ok(output),
        Err(Failure::Execution(error)) => Err(Failure::Execution(Box::new(
            gpu_helpers::reject_gpu_output(output, *error),
        ))),
        Err(Failure::Unsupported) => Err(Failure::Unsupported),
    }
}

async fn usable_scale(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    scale: GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    progress: &mut ResidentStageProgress,
) -> Result<GpuTensorHandle, Failure> {
    let precision = runmat_accelerate_api::handle_precision(input)
        .ok_or_else(|| progress.unavailable(kind, "resident scale has no floating precision"))?;
    let scale_shape = scale.shape().to_vec();
    let scale = admitted_real(kind, provider, input, scale, &scale_shape, precision)?;
    let finite = provider.logical_isfinite(&scale).observe(progress, kind)?;
    let finite = admitted_mask(kind, provider, input, finite, scale.shape())?;
    let zero = progress.observe_optional(
        kind,
        provider.zeros_like(&scale, scale.shape()),
        "resident constant constructor unavailable",
    )?;
    let zero = admitted_real(kind, provider, input, zero, scale.shape(), precision)?;
    let zero = reduction::protect_live_operands(
        kind,
        zero,
        &[input, &scale, &finite],
        "resident zero constructor aliased a live operand",
    )?;
    let zero = reduction::protect_live_operands(
        kind,
        zero,
        protected,
        "resident zero constructor aliased a surrounding operand",
    )?;
    let nonzero = provider
        .elem_ne(&scale, &zero)
        .await
        .observe(progress, kind)?;
    let nonzero = admitted_mask(kind, provider, input, nonzero, scale.shape())?;
    let usable = provider
        .logical_and(&finite, &nonzero)
        .observe(progress, kind)?;
    let usable = admitted_mask(kind, provider, input, usable, scale.shape())?;
    let one = progress.observe_optional(
        kind,
        provider.ones_like(&scale),
        "resident constant constructor unavailable",
    )?;
    let one = admitted_real(kind, provider, input, one, scale.shape(), precision)?;
    let one = reduction::protect_live_operands(
        kind,
        one,
        &[input, &scale, &finite, &zero, &nonzero, &usable],
        "resident one constructor aliased a live operand",
    )?;
    let one = reduction::protect_live_operands(
        kind,
        one,
        protected,
        "resident one constructor aliased a surrounding operand",
    )?;
    let result = provider
        .select(&usable, &scale, &one)
        .await
        .observe(progress, kind)?;
    if result.shares_allocation_with(&scale) || result.shares_allocation_with(&one) {
        return Err(reject_output(
            kind,
            result,
            "resident selection aliased a scale operand",
        ));
    }
    admitted_real(kind, provider, input, result, scale.shape(), precision)
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
    gpu_helpers::require_resident_compute_owner(input, || {
        super::options::invalid(kind, "resident input has no exact owner")
    })?;
    let Some(provider) = gpu_helpers::exact_provider_for_handle(input) else {
        return Err(super::options::invalid(
            kind,
            "resident input has no exact owner",
        ));
    };
    let mut progress = ResidentStageProgress::default();
    let outcome = numerics::run(kind, provider, input, options, &mut progress).await;
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
            let identity = kind == ReductionKind::Harmonic
                && reduction::selected_axes(input.shape(), &options.axes)
                    .iter()
                    .all(|axis| input.shape()[*axis] == 1);
            if output.shares_allocation_with(input) && !identity {
                let error = super::options::invalid(
                    kind,
                    "resident reduction aliased its input allocation",
                );
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
            gpu_helpers::resident_gpu_value(output).map(Some)
        }
        // These four APIs historically admit host materialization for GPU
        // inputs, including explicit gpuArray values. Unsupported provider
        // hooks preserve that documented boundary; execution faults do not.
        Err(Failure::Unsupported) => Ok(None),
        Err(Failure::Execution(error)) => Err(*error),
    }
}
