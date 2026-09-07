use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::{Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::BuiltinResult;

use super::super::{errors, SolveOrientation};

pub(super) struct PreparedOperand {
    provider: &'static dyn AccelProvider,
    handle: GpuTensorHandle,
    owned: bool,
}

impl PreparedOperand {
    pub(super) fn new(
        orientation: SolveOrientation,
        value: &Value,
        provider: &'static dyn AccelProvider,
    ) -> BuiltinResult<Option<Self>> {
        match value {
            Value::GpuTensor(handle) if compatible_handle(handle, provider) => Ok(Some(Self {
                provider,
                handle: handle.clone(),
                owned: false,
            })),
            Value::GpuTensor(_) => Ok(None),
            Value::Tensor(value) if !tensor::is_scalar_tensor(value) => {
                Self::upload(orientation, provider, value).map(Some)
            }
            Value::LogicalArray(value) if value.data.len() != 1 => {
                let value = tensor::logical_to_tensor(value)
                    .map_err(|error| errors::invalid_input(orientation, error))?;
                Self::upload(orientation, provider, &value).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn upload(
        orientation: SolveOrientation,
        provider: &'static dyn AccelProvider,
        tensor: &Tensor,
    ) -> BuiltinResult<Self> {
        let handle = gpu_helpers::upload_tensor(provider, tensor).map_err(|error| {
            errors::internal(
                orientation,
                format!("{}: provider upload failed: {error}", orientation.name()),
            )
        })?;
        Ok(Self {
            provider,
            handle,
            owned: true,
        })
    }

    pub(super) fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
}

impl Drop for PreparedOperand {
    fn drop(&mut self) {
        if self.owned {
            let _ = self.provider.free(&self.handle);
        }
    }
}

fn compatible_handle(handle: &GpuTensorHandle, provider: &dyn AccelProvider) -> bool {
    gpu_helpers::exact_provider_for_handle(handle)
        .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle) == Some(provider.precision())
}
