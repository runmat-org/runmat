use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};

use crate::builtins::common::gpu_helpers::{
    self, BinaryGpuOutputContract, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};

pub(super) fn valid_unary(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::ComplexInterleaved,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}

pub(super) fn valid_binary(
    output: &GpuTensorHandle,
    real: &GpuTensorHandle,
    imaginary: &GpuTensorHandle,
    shape: Vec<usize>,
    precision: Option<ProviderPrecision>,
    provider: &'static dyn AccelProvider,
) -> bool {
    gpu_helpers::binary_gpu_output_matches(
        output,
        real,
        imaginary,
        provider,
        &BinaryGpuOutputContract {
            shape,
            storage: GpuTensorStorage::ComplexInterleaved,
            precision,
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}
