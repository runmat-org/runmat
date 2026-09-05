use super::super::operation::LogarithmOperation;
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

pub(super) fn validate(
    operation: LogarithmOperation,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    mut output: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if !valid(&output, input, provider) {
        gpu_helpers::free_unprotected_exact_owner(&output, &[input]);
        return Err(super::super::errors::internal(
            operation,
            "provider returned malformed logarithm output",
        ));
    }
    let provenance = runmat_accelerate_api::handle_provenance(input)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
    Ok(gpu_helpers::resident_gpu_value(output))
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
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
    };
    gpu_helpers::unary_gpu_output_matches(output, input, provider, contract)
}
