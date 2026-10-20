//! Exact owner and Double result admission for resident sub2ind.
use super::*;
use crate::builtins::common::gpu_helpers;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};

pub(super) fn source(
    dims: &Value,
    subs: crate::call::FunctionArgumentView<'_>,
) -> Option<GpuTensorHandle> {
    // Mixed owners still admit the host numerical computation, but there is
    // no authority for restoring its result to any one of them.
    gpu_helpers::select_resident_output_source(
        std::iter::once(dims)
            .chain(subs)
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
        BUILTIN_NAME,
    )
    .ok()
    .flatten()
}

pub(super) fn direct_owner(
    source: &GpuTensorHandle,
    inputs: &[&GpuTensorHandle],
) -> Option<&'static dyn AccelProvider> {
    let provider = gpu_helpers::exact_provider_for_handle(source)?;
    if provider.precision() != ProviderPrecision::F64
        || inputs.iter().any(|input| {
            gpu_helpers::exact_provider_for_handle(input)
                .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
                || input.device_id() != source.device_id()
                || input.allocation_lease().validate_live().is_err()
                || runmat_accelerate_api::handle_storage(input) != GpuTensorStorage::Real
                || runmat_accelerate_api::handle_precision(input) != Some(ProviderPrecision::F64)
                || runmat_accelerate_api::handle_integer_type(input).is_some()
                || runmat_accelerate_api::handle_is_logical(input)
                || !gpu_helpers::gpu_class_metadata_matches(
                    input,
                    Some(ProviderPrecision::F64),
                    None,
                    false,
                )
        })
    {
        return None;
    }
    Some(provider)
}

pub(super) fn publish_direct(
    mut output: GpuTensorHandle,
    shape: &[usize],
    source: &GpuTensorHandle,
    inputs: &[&GpuTensorHandle],
    provider: &'static dyn AccelProvider,
) -> crate::BuiltinResult<Value> {
    let valid = output.allocation_lease().validate_live().is_ok()
        && output.shape() == shape
        && output.device_id() == source.device_id()
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(&output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&output) == Some(ProviderPrecision::F64)
        && runmat_accelerate_api::handle_integer_type(&output).is_none()
        && !runmat_accelerate_api::handle_is_logical(&output)
        && gpu_helpers::gpu_class_metadata_matches(
            &output,
            Some(ProviderPrecision::F64),
            None,
            false,
        )
        && !output.shares_allocation_with(source)
        && inputs
            .iter()
            .all(|input| !output.shares_allocation_with(input));
    if !valid {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            sub2ind_provider_error("sub2ind: provider returned an invalid Double result"),
        ));
    }
    gpu_helpers::propagate_output_provenance(
        &mut output,
        std::iter::once(source).chain(inputs.iter().copied()),
    )?;
    gpu_helpers::resident_gpu_value(output)
}

pub(super) fn restore_host(
    source: Option<&GpuTensorHandle>,
    value: Value,
    protected: &[&GpuTensorHandle],
) -> crate::BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    if gpu_helpers::exact_provider_for_handle(source)
        .is_none_or(|owner| owner.precision() != ProviderPrecision::F64)
    {
        return Ok(value);
    }
    gpu_helpers::restore_class_preserving_value_protecting(source, value, BUILTIN_NAME, protected)
}

#[cfg(test)]
#[path = "resident_owner/tests.rs"]
mod tests;
