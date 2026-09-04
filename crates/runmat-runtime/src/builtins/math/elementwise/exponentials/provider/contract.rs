use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};

use crate::builtins::common::gpu_helpers;

pub(in crate::builtins::math::elementwise::exponentials) fn valid_real(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    expected_precision: Option<ProviderPrecision>,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: expected_precision,
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}

pub(super) fn precision(handle: &GpuTensorHandle) -> Option<ProviderPrecision> {
    runmat_accelerate_api::handle_precision(handle)
}
