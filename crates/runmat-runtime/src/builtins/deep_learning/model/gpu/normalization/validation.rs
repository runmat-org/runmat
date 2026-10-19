use super::super::*;
use crate::builtins::common::gpu_helpers::{
    reject_gpu_output, unary_gpu_output_matches_shape, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};

pub(super) fn validate_output(
    output: GpuTensorHandle,
    input: &GpuTensorHandle,
    scale: &GpuTensorHandle,
    offset: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let precision = runmat_accelerate_api::handle_precision(input);
    if !unary_gpu_output_matches_shape(
        &output,
        input,
        provider,
        shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision,
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) || output.shares_allocation_with(scale)
        || output.shares_allocation_with(offset)
    {
        return Err(reject_gpu_output(
            output,
            crate::build_runtime_error(format!(
                "{function}: provider returned an invalid normalization output descriptor"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ));
    }
    Ok(output)
}

pub(super) fn validate_statistics(
    statistics: GpuTensorHandle,
    input: &GpuTensorHandle,
    scale: &GpuTensorHandle,
    offset: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    groups: usize,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let shape = [3, groups];
    if !unary_gpu_output_matches_shape(
        &statistics,
        input,
        provider,
        &shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) || statistics.shares_allocation_with(scale)
        || statistics.shares_allocation_with(offset)
        || statistics.shares_allocation_with(output)
    {
        return Err(reject_gpu_output(
            statistics,
            crate::build_runtime_error(format!(
                "{function}: provider returned invalid normalization statistics"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ));
    }
    Ok(statistics)
}
