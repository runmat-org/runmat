use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{IntegerStorage, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
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
            Value::Num(v) => Ok(Self::Host(scalar_tensor(v))),
            Value::Int(v) => Ok(Self::Host(
                Tensor::new_integer(IntegerStorage::from_scalar(v), vec![1, 1])
                    .expect("integer scalar tensor shape"),
            )),
            Value::Bool(v) => Ok(Self::Host(scalar_tensor(if v { 1.0 } else { 0.0 }))),
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
            Self::Host(tensor) => tensor_utils::tensor_element_len(tensor),
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

fn scalar_tensor(value: f64) -> Tensor {
    Tensor::new(vec![value], vec![1]).expect("scalar tensor shape")
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn numeric_input_wraps_scalar_num() {
        let NumericInput::Host(tensor) = NumericInput::from_value(Value::Num(2.5), "plot").unwrap()
        else {
            panic!("expected host tensor")
        };
        assert_eq!(tensor.materialize_f64(), vec![2.5]);
        assert_eq!(tensor.shape, vec![1]);
    }

    #[test]
    fn numeric_input_len_reads_typed_integer_storage_without_mirror() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]).expect("tensor");

        let input = NumericInput::from_value(Value::Tensor(tensor), "plot").unwrap();

        assert_eq!(input.len(), 3);
    }

    #[test]
    fn numeric_input_preserves_wide_typed_integer_scalar() {
        let wide = 9_007_199_254_740_993_u64;
        let NumericInput::Host(tensor) =
            NumericInput::from_value(Value::Int(runmat_builtins::IntValue::U64(wide)), "plot")
                .unwrap()
        else {
            panic!("expected host tensor")
        };

        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide]))
        );
    }
}
