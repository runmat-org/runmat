//! Admission of an image-normalization result under its selected provider.
use anyhow::anyhow;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderExecutionError,
    ProviderExecutionResult,
};
use runmat_runtime::builtins::common::gpu_helpers::{
    exact_provider_for_handle, gpu_class_metadata_matches,
};

/// Only a unique provider-created output can be released on rejection.
/// A malformed alias of input, scratch or a public checkpoint is borrowed.
pub(super) fn reject_untrusted_output<'a>(
    output: GpuTensorHandle,
    protected: impl Iterator<Item = &'a GpuTensorHandle>,
    message: &'static str,
) -> ProviderExecutionError {
    if protected
        .into_iter()
        .any(|borrowed| output.shares_allocation_with(borrowed))
    {
        ProviderExecutionError::from(anyhow!(message)).after_effect()
    } else {
        super::outputs::reject_outputs(vec![output], message)
    }
}

pub(super) fn admit(
    output: GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
) -> ProviderExecutionResult<GpuTensorHandle> {
    admit_stage(output, provider, input, input.shape(), &[])
}

/// One owner for the final-only and staged image result contracts. Published
/// moments are protected alongside the borrowed image during later stages.
pub(super) fn admit_stage(
    output: GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    expected_shape: &[usize],
    prior: &[&GpuTensorHandle],
) -> ProviderExecutionResult<GpuTensorHandle> {
    let element = input.physical().element_type();
    let precision = match element {
        NumericElementType::F32 => Some(runmat_accelerate_api::ProviderPrecision::F32),
        NumericElementType::F64 => Some(runmat_accelerate_api::ProviderPrecision::F64),
        _ => None,
    };
    let valid = precision.is_some()
        && output.shape() == expected_shape
        && output.physical().storage() == GpuTensorStorage::Real
        && output.physical().element_type() == element
        && output.device_id() == provider.device_id()
        && !runmat_accelerate_api::handle_is_logical(&output)
        && !runmat_accelerate_api::handle_is_transposed(&output)
        && gpu_class_metadata_matches(&output, precision, None, false)
        && exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && output.allocation_lease().validate_live().is_ok()
        && !output.shares_allocation_with(input)
        && prior
            .iter()
            .all(|protected| !output.shares_allocation_with(protected));
    if valid {
        Ok(output)
    } else {
        Err(reject_untrusted_output(
            output,
            std::iter::once(input).chain(prior.iter().copied()),
            "image normalize provider returned an invalid result",
        ))
    }
}

#[cfg(test)]
#[path = "image_output/tests.rs"]
mod tests;
