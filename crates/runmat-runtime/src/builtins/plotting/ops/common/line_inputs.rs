use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};

use crate::builtins::plotting::common::{gather_tensor_from_gpu, gather_tensor_from_gpu_async};
use crate::builtins::plotting::plotting_error;
use crate::BuiltinResult;

#[derive(Clone, Debug)]
pub enum NumericInput {
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl NumericInput {
    pub fn from_value(value: Value, builtin: &'static str) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = Tensor::try_from(&other)
                    .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?;
                Ok(Self::Host(tensor))
            }
        }
    }

    pub fn gpu_handle(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Gpu(handle) => Some(handle),
            Self::Host(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Host(tensor) => tensor.data.len(),
            Self::Gpu(handle) => handle.shape.iter().product(),
        }
    }

    pub fn into_tensor(self, builtin: &'static str) -> BuiltinResult<Tensor> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Gpu(handle) => gather_tensor_from_gpu(handle, builtin),
        }
    }

    pub async fn into_tensor_async(self, builtin: &'static str) -> BuiltinResult<Tensor> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Gpu(handle) => gather_tensor_from_gpu_async(handle, builtin).await,
        }
    }
}
