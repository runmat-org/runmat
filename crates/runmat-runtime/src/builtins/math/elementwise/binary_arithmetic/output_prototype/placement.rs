use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::{IntegerStorage, Tensor, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::BuiltinResult;

use super::analysis::DevicePreference;
use super::conversion::char_array_to_tensor;
use super::OutputPrototypeContext;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::resident_output_from_sources;

pub(super) async fn ensure(
    context: OutputPrototypeContext,
    value: Value,
    device: &DevicePreference,
) -> BuiltinResult<Value> {
    match device {
        DevicePreference::Host => convert_to_host(context, value).await,
        DevicePreference::LikeGpu(prototype) => convert_to_gpu(context, value, prototype).await,
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

async fn convert_to_gpu(
    context: OutputPrototypeContext,
    value: Value,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    let Some(provider) = gpu_helpers::exact_provider_for_handle(prototype) else {
        return Err(context.described_error(
            context.invalid_argument,
            "GPU output requested via 'like' but no acceleration provider owns the prototype",
        ));
    };
    let host_value = match value {
        Value::GpuTensor(handle) if same_placement(provider, prototype, &handle) => {
            return Ok(resident_output_from_sources(handle, [prototype]));
        }
        Value::GpuTensor(handle) => gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, context.identity.name))?,
        value => value,
    };
    let tensor = host_value_to_tensor(context, host_value)?;
    let handle = gpu_helpers::upload_tensor(provider, &tensor)
        .map_err(|error| context.internal_error(format!("failed to upload GPU result: {error}")))?;
    Ok(resident_output_from_sources(handle, [prototype]))
}

fn same_placement(
    prototype_provider: &dyn AccelProvider,
    prototype: &GpuTensorHandle,
    output: &GpuTensorHandle,
) -> bool {
    prototype.device_id == output.device_id
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|output_provider| std::ptr::eq(output_provider, prototype_provider))
}

fn host_value_to_tensor(context: OutputPrototypeContext, value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::Num(number) => {
            Tensor::new(vec![number], vec![1, 1]).map_err(|error| context.internal_error(error))
        }
        Value::Int(integer) => {
            Tensor::new_integer(IntegerStorage::from_scalar(integer), vec![1, 1])
                .map_err(|error| context.internal_error(error))
        }
        Value::Bool(boolean) => Tensor::new(vec![if boolean { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|error| context.internal_error(error)),
        Value::LogicalArray(logical) => {
            tensor::logical_to_tensor(&logical).map_err(|error| context.internal_error(error))
        }
        Value::CharArray(chars) => char_array_to_tensor(context, &chars),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(context.described_error(
            context.invalid_argument,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(context.described_error(
            context.invalid_argument,
            format!("unsupported prototype conversion to GPU output ({other:?})"),
        )),
    }
}
