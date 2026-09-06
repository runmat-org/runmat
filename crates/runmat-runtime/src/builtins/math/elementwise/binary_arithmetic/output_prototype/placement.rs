use runmat_value::{IntegerStorage, Tensor, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::BuiltinResult;

use super::analysis::DevicePreference;
use super::conversion::char_array_to_tensor;
use super::OutputPrototypeContext;

pub(super) async fn ensure(
    context: OutputPrototypeContext,
    value: Value,
    device: DevicePreference,
) -> BuiltinResult<Value> {
    match device {
        DevicePreference::Host => convert_to_host(context, value).await,
        DevicePreference::Gpu => convert_to_gpu(context, value),
    }
}

async fn convert_to_host(context: OutputPrototypeContext, value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = value {
        gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, context.identity.name))
    } else {
        Ok(value)
    }
}

fn convert_to_gpu(context: OutputPrototypeContext, value: Value) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(context.described_error(
            context.invalid_argument,
            "GPU output requested via 'like' but no acceleration provider is active",
        ));
    };
    match value {
        Value::GpuTensor(handle) => Ok(gpu_helpers::resident_gpu_value(handle)),
        Value::Tensor(tensor) => {
            let handle = gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
                context.internal_error(format!("failed to upload GPU result: {error}"))
            })?;
            Ok(gpu_helpers::resident_gpu_value(handle))
        }
        Value::Num(number) => {
            let tensor = Tensor::new(vec![number], vec![1, 1])
                .map_err(|error| context.internal_error(error))?;
            convert_to_gpu(context, Value::Tensor(tensor))
        }
        Value::Int(integer) => {
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(integer), vec![1, 1])
                .map_err(|error| context.internal_error(error))?;
            convert_to_gpu(context, Value::Tensor(tensor))
        }
        Value::Bool(boolean) => {
            convert_to_gpu(context, Value::Num(if boolean { 1.0 } else { 0.0 }))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|error| context.internal_error(error))?;
            convert_to_gpu(context, Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(context, &chars)?;
            convert_to_gpu(context, Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(context.described_error(
            context.invalid_argument,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        Value::String(_)
        | Value::StringArray(_)
        | Value::SparseTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::OutputList(_) => Err(context.described_error(
            context.invalid_argument,
            "unsupported prototype conversion to GPU output",
        )),
    }
}
