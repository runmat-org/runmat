mod binary;
mod unary;

pub(super) use binary::try_direct as try_binary_direct;
pub(super) use unary::evaluate as evaluate_unary;

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};

use crate::builtins::common::gpu_helpers;

use super::errors;

fn precision_for(dtype: runmat_value::NumericDType) -> Option<ProviderPrecision> {
    match dtype {
        runmat_value::NumericDType::F32 => Some(ProviderPrecision::F32),
        runmat_value::NumericDType::F64 => Some(ProviderPrecision::F64),
        _ => None,
    }
}

fn preserve_unary_provenance(output: &mut GpuTensorHandle, input: &GpuTensorHandle) {
    let provenance = runmat_accelerate_api::handle_provenance(input)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(output, provenance);
}

fn unary_contract(
    storage: GpuTensorStorage,
    precision: Option<ProviderPrecision>,
) -> gpu_helpers::UnaryGpuOutputContract {
    gpu_helpers::UnaryGpuOutputContract {
        storage,
        precision,
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
    }
}

fn unsupported_output_kind() -> crate::RuntimeError {
    errors::internal("host fallback produced an unsupported resident output kind")
}
