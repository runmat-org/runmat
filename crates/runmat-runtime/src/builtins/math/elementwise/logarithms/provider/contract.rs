use super::super::operation::LogarithmOperation;
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

pub(in crate::builtins::math::elementwise::logarithms) fn validate(
    operation: LogarithmOperation,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if !valid(&output, input, provider) {
        let mut error = super::super::errors::internal(
            operation,
            "provider returned malformed logarithm output",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}

pub(in crate::builtins::math::elementwise::logarithms) fn valid(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> bool {
    let contract = gpu_helpers::UnaryGpuOutputContract {
        storage: GpuTensorStorage::Real,
        precision: runmat_accelerate_api::handle_precision(input),
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    gpu_helpers::unary_gpu_output_matches(output, input, provider, contract)
        && gpu_helpers::expected_handle_numeric_element_type(output).is_ok_and(|element| {
            gpu_helpers::numeric_descriptor_matches(output, element, GpuTensorStorage::Real)
        })
}
