//! Ownership and publication at signal-provider boundaries.
//!
//! Geometry and numeric class remain the calling operation's contract. These
//! helpers compose the existing common validators and restoration policy.
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderExecutionError};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers::{self, BinaryGpuOutputContract};
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};

fn terminal(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

pub(super) fn owner(
    input: &GpuTensorHandle,
    error: impl FnOnce() -> RuntimeError,
) -> BuiltinResult<&'static dyn AccelProvider> {
    if let Err(cause) = input.allocation_lease().validate_live() {
        let mut error = error();
        error.gpu_gather_retry = GpuGatherRetry::Never;
        return Err(crate::runtime_error::with_provider_execution_source(
            error,
            ProviderExecutionError::from_source(cause),
        ));
    }
    gpu_helpers::exact_provider_for_handle(input).ok_or_else(|| {
        let mut error = error();
        error.gpu_gather_retry = GpuGatherRetry::Never;
        error
    })
}

/// Caller operands determine placement; provider-produced siblings only exclude aliases.
pub(super) struct OutputContext<'a> {
    pub input: &'a GpuTensorHandle,
    pub provider: &'static dyn AccelProvider,
    pub caller_inputs: &'a [&'a GpuTensorHandle],
}

impl OutputContext<'_> {
    /// Publish coordinates in signal precision through the same typed restoration
    /// policy as fallback outputs. No legacy preferred-precision upload is used.
    pub(super) fn coordinate(
        &self,
        values: Vec<f64>,
        shape: Vec<usize>,
        siblings: &[&GpuTensorHandle],
        builtin: &str,
        error: impl FnOnce() -> RuntimeError,
    ) -> BuiltinResult<Value> {
        let tensor = super::common::output_precision::SignalPrecision::of_handle(self.input)
            .real(values, shape)
            .map_err(|_| terminal(error()))?;
        let protected = self
            .caller_inputs
            .iter()
            .copied()
            .chain(siblings.iter().copied())
            .collect::<Vec<_>>();
        gpu_helpers::restore_class_preserving_value_protecting(
            self.input,
            Value::Tensor(tensor),
            builtin,
            &protected,
        )
    }

    pub(super) fn admit(
        &self,
        output: GpuTensorHandle,
        contract: &BinaryGpuOutputContract,
        siblings: &[&GpuTensorHandle],
        error: impl FnOnce() -> RuntimeError,
    ) -> BuiltinResult<GpuTensorHandle> {
        if let Err(cause) = output.allocation_lease().validate_live() {
            let mut error = error();
            error.gpu_gather_retry = GpuGatherRetry::Never;
            let error = crate::runtime_error::with_provider_execution_source(
                error,
                ProviderExecutionError::from_source(cause),
            );
            return Err(gpu_helpers::reject_gpu_output(output, error));
        }
        if !gpu_helpers::binary_gpu_output_matches(
            &output,
            self.input,
            self.input,
            self.provider,
            contract,
        ) || self
            .caller_inputs
            .iter()
            .chain(siblings.iter())
            .any(|value| output.shares_allocation_with(value))
        {
            let mut error = error();
            error.gpu_gather_retry = GpuGatherRetry::Never;
            return Err(gpu_helpers::reject_gpu_output(output, error));
        }
        // Protected siblings are alias exclusions, not caller intent authorities.
        let output = gpu_helpers::propagate_owned_output_provenance(
            output,
            std::iter::once(self.input).chain(self.caller_inputs.iter().copied()),
        )
        .map_err(|mut error| {
            error.gpu_gather_retry = GpuGatherRetry::Never;
            error
        })?;
        Ok(output)
    }
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
    let values = values
        .try_into_owned()
        .map_err(|_| {
            crate::sequence::sequence_error_to_runtime(
                runmat_value::ValueSequenceError::SharedOutputStorage,
            )
        })?
        .try_into_unbounded_vec()
        .map_err(|_| {
            crate::sequence::sequence_error_to_runtime(
                runmat_value::ValueSequenceError::SharedOutputStorage,
            )
        })?;
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
    Ok(Value::output_list(outputs))
}

/// Surplus ABI padding is not a computed spectral output. Keep its existing
/// host Double scalar representation instead of uploading it as signal data.
pub(super) fn restore_named(
    input: &GpuTensorHandle,
    value: Value,
    named_outputs: usize,
    builtin: &str,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    let Value::OutputList(values) = value else {
        return restore(input, value, builtin, protected);
    };
    let mut values = values
        .try_into_owned()
        .map_err(|_| {
            crate::runtime_error::semantic_error(
                "SharedOutputStorage",
                "resident signal output storage is shared and cannot be rewritten implicitly",
            )
        })?
        .try_into_unbounded_vec()
        .map_err(|_| {
            crate::runtime_error::semantic_error(
                "SharedOutputStorage",
                "resident signal output storage is shared and cannot be rewritten implicitly",
            )
        })?;
    let padding = values.split_off(values.len().min(named_outputs));
    let Value::OutputList(outputs) =
        restore(input, Value::output_list(values), builtin, protected)?
    else {
        unreachable!("list restoration preserves its carrier")
    };
    let mut outputs = outputs
        .try_into_owned()
        .map_err(|_| {
            crate::runtime_error::semantic_error(
                "SharedOutputStorage",
                "restored signal output storage is shared and cannot be extended implicitly",
            )
        })?
        .try_into_unbounded_vec()
        .map_err(|_| {
            crate::runtime_error::semantic_error(
                "SharedOutputStorage",
                "restored signal output storage is shared and cannot be extended implicitly",
            )
        })?;
    outputs.extend(padding);
    Ok(Value::output_list(outputs))
}

/// Validate both members before any consumer can reshape, reduce, or publish
/// either one. Result-reported dimensions are not a geometry authority.
pub(super) fn spectral(
    result: runmat_accelerate_api::ProviderSpectralResult,
    request: &runmat_accelerate_api::ProviderSpectralRequest<'_>,
    provider: &'static dyn AccelProvider,
    protected: &[&GpuTensorHandle],
    error: impl Fn() -> RuntimeError,
) -> BuiltinResult<runmat_accelerate_api::ProviderSpectralResult> {
    use runmat_accelerate_api::{GpuTensorStorage, ProviderSpectralRange};
    let rows = match request.range {
        ProviderSpectralRange::Onesided => request.nfft / 2 + 1,
        ProviderSpectralRange::Twosided | ProviderSpectralRange::Centered => request.nfft,
    };
    if result.rows != rows || result.cols != request.frame_count {
        let mut error = error();
        error.gpu_gather_retry = GpuGatherRetry::Never;
        return Err(gpu_helpers::reject_gpu_output(
            result.ps,
            gpu_helpers::reject_gpu_output(result.s, error),
        ));
    }
    let mut contract = floating_contract(
        request.input,
        vec![rows, request.frame_count],
        GpuTensorStorage::ComplexInterleaved,
    );
    let context = OutputContext {
        input: request.input,
        provider,
        caller_inputs: protected,
    };
    let s = context.admit(result.s, &contract, &[&result.ps], &error)?;
    contract.storage = GpuTensorStorage::Real;
    let ps = context.admit(result.ps, &contract, &[&s], error)?;
    Ok(runmat_accelerate_api::ProviderSpectralResult {
        s,
        ps,
        rows,
        cols: request.frame_count,
    })
}

pub(super) fn floating_contract(
    input: &GpuTensorHandle,
    shape: Vec<usize>,
    storage: runmat_accelerate_api::GpuTensorStorage,
) -> BinaryGpuOutputContract {
    let precision = if runmat_accelerate_api::handle_is_logical(input)
        || runmat_accelerate_api::handle_integer_type(input).is_some()
    {
        Some(runmat_accelerate_api::ProviderPrecision::F64)
    } else {
        runmat_accelerate_api::handle_precision(input)
    };
    BinaryGpuOutputContract {
        shape,
        storage,
        precision,
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
    }
}
