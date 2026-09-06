use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::BuiltinResult;

use super::OutputPrototypeContext;

#[async_recursion::async_recursion(?Send)]
pub(in crate::builtins::math::elementwise::binary_arithmetic) async fn real_to_complex(
    context: OutputPrototypeContext,
    value: Value,
) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(number) => Ok(Value::Complex(number, 0.0)),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let storage = tensor
                .into_numeric_storage()
                .map_err(|error| context.internal_error(error))?;
            let storage = match storage {
                NumericStorage::F64(values) => {
                    ComplexStorage::F64(values.into_iter().map(|value| (value, 0.0)).collect())
                }
                NumericStorage::F32(values) => {
                    ComplexStorage::F32(values.into_iter().map(|value| (value, 0.0)).collect())
                }
                storage => promote_integer_storage(storage),
            };
            let tensor = ComplexTensor::from_complex_storage(storage, shape)
                .map_err(|error| context.internal_error(error))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|error| context.internal_error(error))?;
            real_to_complex(context, Value::Tensor(tensor)).await
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(context, &chars)?;
            real_to_complex(context, Value::Tensor(tensor)).await
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, context.identity.name))?;
            real_to_complex(context, gathered).await
        }
        other => Err(context.described_error(
            context.invalid_input,
            format!("cannot convert value {other:?} to complex output"),
        )),
    }
}

fn promote_integer_storage(storage: NumericStorage) -> ComplexStorage {
    ComplexStorage::F64(
        storage
            .materialize_f64()
            .into_iter()
            .map(|value| (value, 0.0))
            .collect(),
    )
}

pub(super) fn char_array_to_tensor(
    context: OutputPrototypeContext,
    chars: &runmat_value::CharArray,
) -> BuiltinResult<Tensor> {
    Tensor::new(
        chars
            .data
            .iter()
            .map(|&value| value as u32 as f64)
            .collect(),
        chars.shape.clone(),
    )
    .map_err(|error| context.internal_error(error))
}
