use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};

use crate::builtins::common::gpu_helpers::{
    self, BinaryGpuOutputContract, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};

pub(super) fn compatible_shape(real: &[usize], imaginary: &[usize]) -> Option<Vec<usize>> {
    if real == imaginary || element_count(imaginary) == Some(1) {
        Some(real.to_vec())
    } else if element_count(real) == Some(1) {
        Some(imaginary.to_vec())
    } else {
        None
    }
}

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
            alias: GpuOutputAliasPolicy::RequireDistinct,
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
            alias: GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}

fn element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
}
