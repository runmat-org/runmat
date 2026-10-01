//! Admission of each resident `sum` stage before the next provider effect.

use super::{gpu_helpers, sum_internal_error, OutputTemplate};
use crate::{BuiltinResult, GpuGatherRetry};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

fn malformed(detail: &str) -> crate::RuntimeError {
    let mut error = sum_internal_error(format!(
        "sum: provider returned malformed {detail} class, shape, owner, or alias"
    ));
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

fn physical_contract(
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    alias: gpu_helpers::GpuOutputAliasPolicy,
    output_class: OutputTemplate,
) -> gpu_helpers::UnaryGpuOutputContract {
    if runmat_accelerate_api::handle_is_logical(input) {
        // A logical reduction counts numeric values. Its provider intermediate
        // uses the owner's floating precision; public double adaptation follows.
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: Some(provider.precision()),
            integer: None,
            logical: false,
            alias,
        }
    } else if runmat_accelerate_api::handle_integer_type(input).is_some()
        && !matches!(output_class, OutputTemplate::Native)
    {
        // Ordinary integer reduction accumulates in double. An integer result
        // here has already lost information to native saturation; casting it
        // later cannot repair the numerical result.
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: Some(runmat_accelerate_api::ProviderPrecision::F64),
            integer: None,
            logical: false,
            alias,
        }
    } else {
        gpu_helpers::UnaryGpuOutputContract {
            storage: runmat_accelerate_api::handle_storage(input),
            precision: runmat_accelerate_api::handle_precision(input),
            integer: runmat_accelerate_api::handle_integer_type(input),
            logical: false,
            alias,
        }
    }
}

fn admit(
    provider: &'static dyn AccelProvider,
    original: &GpuTensorHandle,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    expected_shape: &[usize],
    alias: gpu_helpers::GpuOutputAliasPolicy,
    detail: &str,
    output_class: OutputTemplate,
) -> BuiltinResult<GpuTensorHandle> {
    // A provider may use a one-element scalar view with fewer singleton axes.
    // Normalize only that equivalent view; never reshape a nonscalar result.
    let output = if output.shape() != expected_shape
        && output.shape().iter().all(|&extent| extent == 1)
        && expected_shape.iter().all(|&extent| extent == 1)
    {
        match output.with_view(output.view().clone().with_shape(expected_shape.to_vec())) {
            Ok(normalized) => normalized,
            Err(source) => {
                let mut error = malformed(detail);
                error.source = Some(Box::new(source));
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
        }
    } else {
        output
    };
    let contract = physical_contract(input, provider, alias, output_class);
    let original_alias_allowed = matches!(alias, gpu_helpers::GpuOutputAliasPolicy::AllowInput)
        && original.shares_allocation_with(input);
    let valid = gpu_helpers::unary_gpu_output_matches_shape(
        &output,
        input,
        provider,
        expected_shape,
        contract,
    ) && (!output.shares_allocation_with(original) || original_alias_allowed)
        && gpu_helpers::expected_handle_numeric_element_type(&output).is_ok()
        && gpu_helpers::numeric_descriptor_matches(
            &output,
            output.physical().element_type(),
            output.physical().storage(),
        );
    if !valid {
        return Err(gpu_helpers::reject_gpu_output(output, malformed(detail)));
    }
    Ok(output)
}

pub(super) fn admit_nan_map(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
) -> BuiltinResult<GpuTensorHandle> {
    admit(
        provider,
        input,
        input,
        output,
        input.shape(),
        gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        "NaN-map",
        OutputTemplate::Native,
    )
}

pub(super) fn admit_reduction(
    provider: &'static dyn AccelProvider,
    original: &GpuTensorHandle,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    dimension: Option<usize>,
    output_class: OutputTemplate,
) -> BuiltinResult<GpuTensorHandle> {
    let shape = if let Some(dimension) = dimension {
        let mut shape = input.shape().to_vec();
        shape[dimension] = 1;
        shape
    } else {
        vec![1, 1]
    };
    let alias = if dimension.is_some_and(|dim| input.shape()[dim] == 1) {
        gpu_helpers::GpuOutputAliasPolicy::AllowInput
    } else {
        gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation
    };
    admit(
        provider,
        original,
        input,
        output,
        &shape,
        alias,
        "reduction",
        output_class,
    )
}

pub(super) fn advance_reduction(
    provider: &'static dyn AccelProvider,
    original: &GpuTensorHandle,
    current: GpuTensorHandle,
    next: GpuTensorHandle,
    dimension: usize,
    output_class: OutputTemplate,
) -> BuiltinResult<GpuTensorHandle> {
    match admit_reduction(
        provider,
        original,
        &current,
        next,
        Some(dimension),
        output_class,
    ) {
        Ok(next) => {
            if let Err(source) = current.release() {
                let mut error = sum_internal_error("sum: failed to retire reduction intermediate");
                error.source = Some(Box::new(source));
                error.gpu_gather_retry = GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(next, error));
            }
            Ok(next)
        }
        Err(error) => Err(gpu_helpers::reject_gpu_output(current, error)),
    }
}

pub(super) fn fail_stage(
    current: GpuTensorHandle,
    error: crate::RuntimeError,
) -> crate::RuntimeError {
    gpu_helpers::reject_gpu_output(current, error)
}

pub(super) fn retire_before_fallback(current: GpuTensorHandle) -> BuiltinResult<()> {
    current.release().map_err(|source| {
        let mut error = sum_internal_error("sum: failed to retire reduction intermediate");
        error.source = Some(Box::new(source));
        error.gpu_gather_retry = GpuGatherRetry::Never;
        error
    })
}

pub(super) fn publish(
    mut output: GpuTensorHandle,
    input: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Err(source) = output.allocation_lease().validate_live() {
        let mut error = malformed("published reduction");
        error.source = Some(Box::new(source));
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    if let Err(error) = gpu_helpers::propagate_output_provenance(&mut output, [input]) {
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    gpu_helpers::resident_gpu_value(output)
}

#[cfg(test)]
#[path = "provider_output/tests.rs"]
mod tests;
