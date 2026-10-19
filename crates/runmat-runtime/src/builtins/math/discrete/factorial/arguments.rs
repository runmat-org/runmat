use runmat_builtins::FACTORIAL_LIKE_EXTENSION;
use runmat_value::{Tensor, Value};

use super::{factorial_error, FactorialError, BUILTIN_NAME};
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::BuiltinResult;

#[derive(Clone)]
pub(super) enum OutputTemplate {
    Default,
    Like(Value),
}

pub(super) fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    match args {
        [] => Ok(OutputTemplate::Default),
        [keyword, prototype] if matches!(keyword_of(keyword).as_deref(), Some("like")) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FACTORIAL_LIKE_EXTENSION,
                BUILTIN_NAME,
            )?;
            Ok(OutputTemplate::Like(prototype.clone()))
        }
        [keyword] if matches!(keyword_of(keyword).as_deref(), Some("like")) => {
            Err(factorial_error(
                FactorialError::InvalidArgument,
                "expected prototype after 'like'",
            ))
        }
        [.., _, _] if args.len() > 2 => Err(factorial_error(
            FactorialError::InvalidArgument,
            "too many input arguments",
        )),
        _ => Err(factorial_error(
            FactorialError::InvalidArgument,
            "unrecognised option; only 'like' is supported",
        )),
    }
}

pub(super) async fn apply_output_template(
    value: Value,
    template: &OutputTemplate,
) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(prototype) => match analyse_prototype(prototype).await? {
            DevicePreference::Host => convert_to_host(value).await,
            DevicePreference::Gpu => convert_to_gpu(value),
        },
    }
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

#[async_recursion::async_recursion(?Send)]
async fn analyse_prototype(prototype: &Value) -> BuiltinResult<DevicePreference> {
    match prototype {
        Value::GpuTensor(_) => Ok(DevicePreference::Gpu),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_) => Ok(DevicePreference::Host),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(factorial_error(
            FactorialError::InvalidInput,
            "complex prototypes for 'like' are not supported; results are always real",
        )),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(factorial_error(
            FactorialError::InvalidInput,
            "prototype must be numeric or a gpuArray",
        )),
        other => {
            let gathered = gpu_helpers::gather_value_async(other)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            analyse_prototype(&gathered).await
        }
    }
}

async fn convert_to_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME)),
        other => Ok(other),
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        factorial_error(
            FactorialError::GpuUnsupported,
            "GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => upload(provider, tensor),
        Value::Num(number) => {
            let tensor = Tensor::new(vec![number], vec![1, 1])
                .map_err(|error| factorial_error(FactorialError::Internal, error))?;
            upload(provider, tensor)
        }
        Value::Int(integer) => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, Value::Int(integer))
                .map_err(|error| factorial_error(FactorialError::Internal, error))?;
            upload(provider, tensor)
        }
        Value::Bool(boolean) => convert_to_gpu(Value::Num(if boolean { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|error| factorial_error(FactorialError::Internal, error))?;
            upload(provider, tensor)
        }
        other => Err(factorial_error(
            FactorialError::InvalidInput,
            format!("cannot place value {other:?} on the GPU via 'like'"),
        )),
    }
}

fn upload(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: Tensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
        factorial_error(
            FactorialError::Internal,
            format!("failed to upload GPU result: {error}"),
        )
    })?;
    Ok(Value::GpuTensor(handle))
}
