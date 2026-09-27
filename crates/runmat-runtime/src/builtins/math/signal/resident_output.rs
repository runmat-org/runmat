//! Ownership and publication at signal-provider boundaries.
//!
//! Geometry and numeric class remain the calling operation's contract. These
//! helpers compose the existing common validators and restoration policy.
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderExecutionError};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers::{self, BinaryGpuOutputContract};
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};

pub(super) fn owner(
    input: &GpuTensorHandle,
    mut error: RuntimeError,
) -> BuiltinResult<&'static dyn AccelProvider> {
    error.gpu_gather_retry = GpuGatherRetry::Never;
    if let Err(cause) = input.allocation_lease().validate_live() {
        return Err(crate::runtime_error::with_provider_execution_source(
            error,
            ProviderExecutionError::from_source(cause),
        ));
    }
    gpu_helpers::exact_provider_for_handle(input).ok_or(error)
}

pub(super) fn admit(
    mut output: GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    contract: &BinaryGpuOutputContract,
    protected: &[&GpuTensorHandle],
    mut error: RuntimeError,
) -> BuiltinResult<GpuTensorHandle> {
    error.gpu_gather_retry = GpuGatherRetry::Never;
    if let Err(cause) = output.allocation_lease().validate_live() {
        return Err(crate::runtime_error::with_provider_execution_source(
            error,
            ProviderExecutionError::from_source(cause),
        ));
    }
    if !gpu_helpers::binary_gpu_output_matches(&output, input, input, provider, contract)
        || protected
            .iter()
            .any(|value| output.shares_allocation_with(value))
    {
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    gpu_helpers::propagate_output_provenance(
        &mut output,
        std::iter::once(input).chain(protected.iter().copied()),
    )
    .map_err(|mut error| {
        error.gpu_gather_retry = GpuGatherRetry::Never;
        error
    })?;
    Ok(output)
}

/// Restore each requested output while protecting borrowed operands and every
/// preceding result. A failed later upload drops owned siblings, never callers.
pub(super) fn restore(
    input: &GpuTensorHandle,
    value: Value,
    builtin: &str,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    let Value::OutputList(values) = value else {
        return gpu_helpers::restore_class_preserving_value_protecting(
            input, value, builtin, protected,
        );
    };
    let mut outputs = Vec::with_capacity(values.len());
    for value in values {
        let borrowed = protected
            .iter()
            .copied()
            .chain(outputs.iter().filter_map(|value| {
                if let Value::GpuTensor(handle) = value {
                    Some(handle)
                } else {
                    None
                }
            }))
            .collect::<Vec<_>>();
        let output = gpu_helpers::restore_class_preserving_value_protecting(
            input, value, builtin, &borrowed,
        )?;
        outputs.push(output);
    }
    Ok(Value::OutputList(outputs))
}
