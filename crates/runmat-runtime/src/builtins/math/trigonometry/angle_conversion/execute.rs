use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_value::{ComplexTensor, NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct AngleConversion {
    pub name: &'static str,
    pub scale_f64: f64,
    pub scale_f32: f32,
    pub invalid_input: &'static BuiltinErrorDescriptor,
    pub internal_error: &'static BuiltinErrorDescriptor,
    pub integer_extension: &'static BuiltinExtensionDescriptor,
    pub logical_extension: &'static BuiltinExtensionDescriptor,
}

pub(super) async fn apply(
    conversion: &'static AngleConversion,
    value: Value,
) -> BuiltinResult<Value> {
    crate::builtins::math::trigonometry::inverse_helpers::reject_excess_outputs(conversion.name)?;
    ensure_extensions(conversion, &value)?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &value,
        conversion.name,
    )?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, conversion.name)?;
    match value {
        Value::GpuTensor(handle) => apply_gpu(conversion, handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(
            re * conversion.scale_f64,
            im * conversion.scale_f64,
        )),
        Value::ComplexTensor(tensor) => apply_complex_tensor(conversion, tensor),
        Value::String(_) | Value::StringArray(_) => {
            Err(error(conversion, conversion.invalid_input))
        }
        other => apply_real(conversion, other),
    }
}

async fn apply_gpu(
    conversion: &'static AngleConversion,
    handle: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        error_with_detail(
            conversion,
            conversion.internal_error,
            "GPU input has no owning provider",
        )
    })?;
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
    let gathered =
        crate::builtins::math::trigonometry::inverse_helpers::align_floating_value_precision(
            gathered,
            &handle,
            conversion.name,
        )?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &gathered,
        conversion.name,
    )?;
    let output = match gathered {
        Value::Complex(re, im) => {
            Value::Complex(re * conversion.scale_f64, im * conversion.scale_f64)
        }
        Value::ComplexTensor(tensor) => apply_complex_tensor(conversion, tensor)?,
        Value::Tensor(tensor) => tensor::tensor_into_value(apply_tensor(conversion, tensor)?),
        Value::Num(value) => Value::Num(value * conversion.scale_f64),
        other => apply_real(conversion, other)?,
    };
    crate::builtins::math::trigonometry::inverse_helpers::upload_value_like(
        provider,
        output,
        conversion.name,
        &handle,
    )
}

fn ensure_extensions(conversion: &AngleConversion, value: &Value) -> BuiltinResult<()> {
    let is_integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if is_integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            conversion.integer_extension,
            conversion.name,
        )?;
    }
    let is_logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if is_logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            conversion.logical_extension,
            conversion.name,
        )?;
    }
    Ok(())
}

fn apply_real(conversion: &AngleConversion, value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(conversion.name, value)
        .map_err(|detail| error_with_detail(conversion, conversion.invalid_input, detail))?;
    apply_tensor(conversion, tensor).map(tensor::tensor_into_value)
}

fn apply_tensor(conversion: &AngleConversion, tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let result = if tensor.numeric_dtype() == NumericDType::F32 {
        let storage = tensor
            .into_numeric_storage()
            .map_err(|detail| error_with_detail(conversion, conversion.invalid_input, detail))?;
        let NumericStorage::F32(values) = storage else {
            unreachable!("F32 dtype must have F32 storage")
        };
        Tensor::from_numeric_storage(
            NumericStorage::F32(
                values
                    .into_iter()
                    .map(|value| value * conversion.scale_f32)
                    .collect(),
            ),
            shape,
        )
    } else {
        Tensor::new(
            tensor::tensor_values_f64_cow(&tensor)
                .iter()
                .map(|value| value * conversion.scale_f64)
                .collect(),
            shape,
        )
    };
    result.map_err(|detail| error_with_detail(conversion, conversion.internal_error, detail))
}

fn apply_complex_tensor(
    conversion: &AngleConversion,
    tensor: ComplexTensor,
) -> BuiltinResult<Value> {
    let dtype = tensor.numeric_dtype();
    let values = tensor
        .materialize_f64()
        .iter()
        .map(|&(re, im)| (re * conversion.scale_f64, im * conversion.scale_f64))
        .collect();
    let converted = ComplexTensor::from_f64_values_with_dtype(values, tensor.shape.clone(), dtype)
        .map_err(|detail| error_with_detail(conversion, conversion.internal_error, detail))?;
    Ok(complex_tensor_into_value(converted))
}

fn error(
    conversion: &AngleConversion,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(descriptor.message).with_builtin(conversion.name);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn error_with_detail(
    conversion: &AngleConversion,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
        .with_builtin(conversion.name);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
