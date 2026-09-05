use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, AccelProvider, GpuTensorHandle,
    ProviderPrecision,
};
use runmat_builtins::{COMPLEX_ERROR_INTERNAL, COMPLEX_ERROR_INVALID_INPUT};
use runmat_value::{NumericDType, Value};

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::super::{error, host};

pub(super) struct RealGpuOperand {
    pub(super) handle: GpuTensorHandle,
    owned: bool,
}

impl RealGpuOperand {
    pub(super) fn from_value(
        value: &Value,
        provider: &'static dyn AccelProvider,
    ) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => {
                let owner = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
                    error(
                        &COMPLEX_ERROR_INVALID_INPUT,
                        "GPU input provider is unavailable",
                    )
                })?;
                if !std::ptr::eq(owner, provider) || handle.device_id != provider.device_id() {
                    return Err(error(
                        &COMPLEX_ERROR_INVALID_INPUT,
                        "GPU inputs must belong to the same provider",
                    ));
                }
                Ok(Self {
                    handle: handle.clone(),
                    owned: false,
                })
            }
            value => {
                let input = host::from_value(value.clone())?;
                let handle = gpu_helpers::upload_tensor(provider, &input.tensor)
                    .map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))?;
                runmat_accelerate_api::set_handle_logical(&handle, false);
                Ok(Self {
                    handle,
                    owned: true,
                })
            }
        }
    }
}

impl Drop for RealGpuOperand {
    fn drop(&mut self) {
        if self.owned {
            gpu_helpers::free_unprotected_exact_owner(&self.handle, &[]);
        }
    }
}

pub(super) fn requires_exact_host_path(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some() || handle_is_logical(handle))
}

pub(super) fn floating_result_precision(
    real: &Value,
    imaginary: &Value,
) -> Option<ProviderPrecision> {
    if is_single(real) || is_single(imaginary) {
        Some(ProviderPrecision::F32)
    } else {
        Some(ProviderPrecision::F64)
    }
}

fn is_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if handle_precision(handle) == Some(ProviderPrecision::F32))
}
