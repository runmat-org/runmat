//! Ownership and physical-result checks for executed learned convolution.

use super::*;
use crate::builtins::common::gpu_helpers::{
    reject_gpu_output, unary_gpu_output_matches_shape, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};
use runmat_accelerate_api::{AccelProvider, GpuTensorLayout, GpuTensorStorage};

fn invalid(function: &'static str) -> crate::RuntimeError {
    crate::runtime_error::build_runtime_error(format!(
        "{function}: provider returned an invalid resident convolution descriptor"
    ))
    .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

pub(super) fn result(
    output: GpuTensorHandle,
    input: &GpuTensorHandle,
    weights: &GpuTensorHandle,
    bias: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let valid = output.view().layout() == GpuTensorLayout::Contiguous
        && !output.shares_allocation_with(weights)
        && !output.shares_allocation_with(bias)
        && unary_gpu_output_matches_shape(
            &output,
            input,
            provider,
            shape,
            UnaryGpuOutputContract {
                storage: GpuTensorStorage::Real,
                precision: runmat_accelerate_api::handle_precision(input),
                integer: None,
                logical: false,
                alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
            },
        );
    if valid {
        Ok(output)
    } else {
        Err(reject_gpu_output(output, invalid(function)))
    }
}

pub(super) fn distinct_gradients(
    input: &GpuTensorHandle,
    weights: &GpuTensorHandle,
    bias: &GpuTensorHandle,
    upstream: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<()> {
    if input.shares_allocation_with(weights)
        || input.shares_allocation_with(bias)
        || weights.shares_allocation_with(bias)
        || input.shares_allocation_with(upstream)
        || weights.shares_allocation_with(upstream)
        || bias.shares_allocation_with(upstream)
    {
        Err(invalid(function))
    } else {
        Ok(())
    }
}
