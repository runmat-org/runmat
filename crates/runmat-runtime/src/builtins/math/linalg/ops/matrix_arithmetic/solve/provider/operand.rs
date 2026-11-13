use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::{Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::BuiltinResult;

use super::super::{errors, SolveOrientation};

#[cfg(test)]
#[path = "operand_tests.rs"]
mod owning_tests;

pub(super) struct PreparedOperand {
    handle: GpuTensorHandle,
}

impl PreparedOperand {
    pub(super) fn new(
        orientation: SolveOrientation,
        value: &Value,
        provider: &'static dyn AccelProvider,
        protected: &[&GpuTensorHandle],
    ) -> BuiltinResult<Option<Self>> {
        match value {
            Value::GpuTensor(handle) if compatible_handle(handle, provider) => Ok(Some(Self {
                handle: handle.clone(),
            })),
            Value::GpuTensor(_) => Ok(None),
            Value::Tensor(value) if !tensor::is_scalar_tensor(value) => {
                Self::upload(provider, value, protected).map(Some)
            }
            Value::LogicalArray(value) if value.data.len() != 1 => {
                let value = tensor::logical_to_tensor(value)
                    .map_err(|error| errors::invalid_input(orientation, error))?;
                Self::upload(provider, &value, protected).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn upload(
        provider: &'static dyn AccelProvider,
        tensor: &Tensor,
        protected: &[&GpuTensorHandle],
    ) -> BuiltinResult<Self> {
        let handle = gpu_helpers::upload_tensor_protecting(provider, tensor, protected)?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
}

fn compatible_handle(handle: &GpuTensorHandle, provider: &dyn AccelProvider) -> bool {
    gpu_helpers::exact_provider_for_handle(handle)
        .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle) == Some(provider.precision())
}
