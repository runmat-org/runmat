//! RunMat `gpuDevice` provider-metadata facade (not MATLAB's GPUDevice object).

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericScalar, Value};

use super::device_metadata::active_device_metadata;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "gpuDevice";

use runmat_builtins::{
    gpu_device_index_from_float, gpu_device_index_from_integer, GPU_DEVICE_ACTIVE_INDEX,
    GPU_DEVICE_ERROR_DEVICE_NOT_AVAILABLE, GPU_DEVICE_ERROR_INVALID_INDEX,
    GPU_DEVICE_ERROR_NO_PROVIDER, GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED,
    GPU_DEVICE_ERROR_TOO_MANY_INPUTS, GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT,
    GPU_DEVICE_LOGICAL_SELECTOR_EXTENSION, GPU_DEVICE_PROVIDER_INFO_EXTENSION,
};

fn gpu_device_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    gpu_device_error_with_message(error.message, error)
}

fn gpu_device_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::gpudevice")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gpuDevice",
    op_kind: GpuOpKind::Custom("device-info"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Pure metadata query; does not enqueue GPU kernels. Returns an error when no provider is registered.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::acceleration::gpu::gpudevice"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gpuDevice",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; the builtin returns a host-resident struct.",
};

/// Query the active provider and return a metadata struct describing the GPU device.
#[runtime_builtin(
    name = "gpuDevice",
    binding_variant = "default",
    builtin_path = "crate::builtins::acceleration::gpu::gpudevice"
)]
async fn gpu_device_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args
        .first()
        .is_some_and(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GPU_DEVICE_LOGICAL_SELECTOR_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let result = match args.as_slice() {
        [] => active_device_metadata()
            .ok_or_else(|| gpu_device_error(&GPU_DEVICE_ERROR_NO_PROVIDER))
            .map(|metadata| Value::Struct(metadata.into_struct())),
        [arg] => handle_single_argument(arg),
        _ => return Err(gpu_device_error(&GPU_DEVICE_ERROR_TOO_MANY_INPUTS).into()),
    }?;
    crate::compatibility::ensure_builtin_extension_enabled(
        &GPU_DEVICE_PROVIDER_INFO_EXTENSION,
        BUILTIN_NAME,
    )?;
    Ok(result)
}

fn is_keyword(value: &Value, keyword: &str) -> bool {
    match value {
        Value::String(s) => s.trim().eq_ignore_ascii_case(keyword),
        Value::CharArray(ca) if ca.rows == 1 => {
            let collected: String = ca.data.iter().collect();
            collected.trim().eq_ignore_ascii_case(keyword)
        }
        _ => false,
    }
}

fn handle_single_argument(arg: &Value) -> BuiltinResult<Value> {
    if is_reset_arg(arg) {
        return Err(gpu_device_error(&GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED).into());
    }

    match parse_device_index(arg)? {
        Some(index) => {
            let info = active_device_metadata()
                .ok_or_else(|| gpu_device_error(&GPU_DEVICE_ERROR_NO_PROVIDER))?;
            if index == GPU_DEVICE_ACTIVE_INDEX {
                Ok(Value::Struct(info.into_struct()))
            } else {
                Err(gpu_device_error_with_message(
                    format!("gpuDevice: GPU device with index {} not available", index),
                    &GPU_DEVICE_ERROR_DEVICE_NOT_AVAILABLE,
                )
                .into())
            }
        }
        None => Err(gpu_device_error(&GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT).into()),
    }
}

fn is_reset_arg(value: &Value) -> bool {
    if is_keyword(value, "reset") {
        return true;
    }
    match value {
        Value::Tensor(t) => tensor::tensor_element_len(t) == 0,
        Value::LogicalArray(la) => la.data.is_empty(),
        _ => false,
    }
}

fn parse_device_index(value: &Value) -> BuiltinResult<Option<u32>> {
    match value {
        Value::Int(i) => integer_to_index(i),
        Value::Num(n) => float_to_index(*n),
        Value::Bool(b) => {
            if *b {
                Ok(Some(GPU_DEVICE_ACTIVE_INDEX))
            } else {
                Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into())
            }
        }
        Value::Tensor(t) => match tensor::tensor_element_len(t) {
            0 => Ok(None),
            1 => match t
                .numeric_value_at(0)
                .expect("tensor storage length matches shape")
            {
                NumericScalar::F64(value) => float_to_index(value),
                NumericScalar::F32(value) => float_to_index(f64::from(value)),
                value => integer_to_index(
                    &value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer"),
                ),
            },
            _ => Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into()),
        },
        Value::LogicalArray(la) => match la.data.len() {
            0 => Ok(None),
            1 => {
                if la.data[0] != 0 {
                    Ok(Some(GPU_DEVICE_ACTIVE_INDEX))
                } else {
                    Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into())
                }
            }
            _ => Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into()),
        },
        _ => Ok(None),
    }
}

fn integer_to_index(value: &IntValue) -> BuiltinResult<Option<u32>> {
    value
        .try_to_u64()
        .and_then(|raw| gpu_device_index_from_integer(i128::from(raw)).ok())
        .map(Some)
        .ok_or_else(|| gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into())
}

fn float_to_index(raw: f64) -> BuiltinResult<Option<u32>> {
    gpu_device_index_from_float(raw)
        .map(Some)
        .map_err(|_| gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX))
}

#[cfg(test)]
mod tests;
