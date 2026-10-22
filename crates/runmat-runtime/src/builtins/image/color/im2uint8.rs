//! MATLAB-compatible `im2uint8` image class conversion.

use runmat_builtins::{
    BuiltinErrorDescriptor, ImageIndexMode, IM2UINT8_DESCRIPTOR, IM2UINT8_ERROR_INTERNAL,
    IM2UINT8_ERROR_INVALID_INPUT, IM2UINT8_ERROR_TOO_MANY_INPUTS, IM2UINT8_ERROR_TOO_MANY_OUTPUTS,
    IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::image::color::common;
use crate::builtins::introspection::class_identity::class_name_for_value;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "im2uint8";

fn im2uint8_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    im2uint8_error_with_message(error.message, error)
}

fn im2uint8_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn im2uint8_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    im2uint8_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn im2uint8_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        build_runtime_error(err.message().to_string())
            .with_builtin(NAME)
            .with_identifier(fallback.identifier.expect("closed image-conversion error"))
            .with_source(err)
            .build()
    }
}

#[runtime_builtin(
    name = "im2uint8",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::color::im2uint8"
)]
async fn im2uint8_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    if rest.len() > 1 {
        return Err(im2uint8_error(&IM2UINT8_ERROR_TOO_MANY_INPUTS));
    }
    crate::output_count::validate_fixed_outputs(
        &IM2UINT8_DESCRIPTOR,
        &IM2UINT8_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let mode = parse_indexed_mode(rest.view())?;
    ensure_resident_integer_class_supported(&value, mode)?;
    let resident_source = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    super::conversion_mode::execute(
        runtime.as_ref(),
        NAME,
        value,
        mode,
        NumericDType::U8,
        rest.len() + 1,
        move |prepared| async move { im2uint8_execute(prepared, mode, resident_source).await },
    )
    .await
}

async fn im2uint8_execute(
    value: Value,
    mode: ImageIndexMode,
    resident_source: Option<runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let value = if matches!(value, Value::GpuTensor(_)) {
        common::gather_value(NAME, &value)
            .await
            .map_err(|err| im2uint8_map_error(err, &IM2UINT8_ERROR_INVALID_INPUT))?
    } else {
        value
    };
    let result = match value {
        Value::Tensor(tensor) => Ok(common::image_value_from_tensor(
            im2uint8_tensor(tensor, mode)
                .map_err(|err| im2uint8_map_error(err, &IM2UINT8_ERROR_INTERNAL))?,
        )),
        Value::LogicalArray(array) => {
            if mode == ImageIndexMode::Indexed {
                let shape = array.shape.clone();
                let tensor =
                    Tensor::from_numeric_storage(NumericStorage::U8(array.data.into_vec()), shape)
                        .map_err(|err| im2uint8_error_with_detail(&IM2UINT8_ERROR_INTERNAL, err))?;
                Ok(common::image_value_from_tensor(tensor))
            } else {
                let tensor = tensor::logical_to_tensor(&array)
                    .map_err(|err| im2uint8_error_with_detail(&IM2UINT8_ERROR_INTERNAL, err))?;
                Ok(common::image_value_from_tensor(
                    im2uint8_tensor(tensor, ImageIndexMode::Intensity)
                        .map_err(|err| im2uint8_map_error(err, &IM2UINT8_ERROR_INTERNAL))?,
                ))
            }
        }
        Value::Int(IntValue::U8(v)) => Ok(Value::Int(IntValue::U8(v))),
        Value::Int(IntValue::U16(v)) if mode == ImageIndexMode::Indexed && v <= u8::MAX.into() => {
            Ok(Value::Int(IntValue::U8(v as u8)))
        }
        Value::Int(IntValue::U16(_)) if mode == ImageIndexMode::Indexed => {
            Err(im2uint8_error_with_detail(
                &IM2UINT8_ERROR_INVALID_INPUT,
                "indexed uint16 values must not exceed 255",
            ))
        }
        Value::Int(IntValue::U16(v)) => Ok(Value::Int(IntValue::U8(quantize_u16_to_u8(v)))),
        Value::Int(IntValue::I16(v)) if mode == ImageIndexMode::Indexed => {
            Err(im2uint8_error_with_detail(
                &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
                format!("class {} for indexed image", IntValue::I16(v).class_name()),
            ))
        }
        Value::Int(IntValue::I16(v)) => Ok(Value::Int(IntValue::U8(quantize_u16_to_u8(
            (i32::from(v) - i32::from(i16::MIN)) as u16,
        )))),
        Value::Int(v) => Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!("class {}", v.class_name()),
        )),
        Value::Num(v) if mode == ImageIndexMode::Indexed && v <= 256.0 => Ok(Value::Int(
            IntValue::U8(common::clamp_round(v - 1.0, 255.0) as u8),
        )),
        Value::Num(_) if mode == ImageIndexMode::Indexed => Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_INVALID_INPUT,
            "indexed double values must not exceed 256",
        )),
        Value::Num(v) => Ok(Value::Int(IntValue::U8(common::unit_to_dtype(
            common::clamp01(v),
            NumericDType::U8,
        ) as u8))),
        Value::Bool(v) => Ok(Value::Int(IntValue::U8(
            if mode == ImageIndexMode::Indexed {
                u8::from(v)
            } else if v {
                255
            } else {
                0
            },
        ))),
        other => Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!("type {}", class_name_for_value(&other)),
        )),
    }?;
    let output = common::restore_resident_numeric_result(resident_source.as_ref(), result, NAME)
        .map_err(|err| im2uint8_map_error(err, &IM2UINT8_ERROR_INTERNAL))?;
    crate::execution::RootedValue::new(output)
        .map_err(|err| im2uint8_map_error(err.into_runtime_error(), &IM2UINT8_ERROR_INTERNAL))
}

fn ensure_resident_integer_class_supported(
    value: &Value,
    mode: ImageIndexMode,
) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        return Ok(());
    };
    let Some(dtype) = runmat_accelerate_api::handle_integer_type(handle) else {
        return Ok(());
    };
    let supported = if mode == ImageIndexMode::Indexed {
        matches!(
            dtype,
            runmat_accelerate_api::IntegerElementType::U8
                | runmat_accelerate_api::IntegerElementType::U16
        )
    } else {
        matches!(
            dtype,
            runmat_accelerate_api::IntegerElementType::I16
                | runmat_accelerate_api::IntegerElementType::U8
                | runmat_accelerate_api::IntegerElementType::U16
        )
    };
    if supported {
        Ok(())
    } else {
        Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!(
                "unsupported{} resident integer image class {dtype:?}",
                if mode == ImageIndexMode::Indexed {
                    " indexed"
                } else {
                    ""
                }
            ),
        ))
    }
}

fn parse_indexed_mode(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ImageIndexMode> {
    let Some(value) = rest.first() else {
        return Ok(ImageIndexMode::Intensity);
    };
    crate::builtins::image::color::conversion_mode::decode(value)
        .map_err(|source| {
            let mut error =
                im2uint8_error_with_detail(&IM2UINT8_ERROR_INVALID_INPUT, source.to_string());
            error.source = Some(Box::new(source));
            error
        })?
        .ok_or_else(|| {
            im2uint8_error_with_detail(
                &IM2UINT8_ERROR_INVALID_INPUT,
                "second input must be the text \"indexed\"",
            )
        })
}

fn quantize_u16_to_u8(value: u16) -> u8 {
    ((u32::from(value) + 128) >> 8).min(u32::from(u8::MAX)) as u8
}

fn im2uint8_tensor(tensor: Tensor, mode: ImageIndexMode) -> BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| im2uint8_error_with_detail(&IM2UINT8_ERROR_INTERNAL, err))?;
    let data = match (storage, mode) {
        (NumericStorage::U8(values), ImageIndexMode::Indexed) => values,
        (NumericStorage::U16(values), ImageIndexMode::Indexed) => {
            if values.iter().any(|value| *value > u16::from(u8::MAX)) {
                return Err(im2uint8_error_with_detail(
                    &IM2UINT8_ERROR_INVALID_INPUT,
                    "indexed uint16 values must not exceed 255",
                ));
            }
            values.into_iter().map(|value| value as u8).collect()
        }
        (NumericStorage::F64(values), ImageIndexMode::Indexed) => {
            if values.iter().any(|value| *value > 256.0) {
                return Err(im2uint8_error_with_detail(
                    &IM2UINT8_ERROR_INVALID_INPUT,
                    "indexed double values must not exceed 256",
                ));
            }
            values
                .into_iter()
                .map(|value| common::clamp_round(value - 1.0, 255.0) as u8)
                .collect()
        }
        (unsupported, ImageIndexMode::Indexed) => {
            return Err(im2uint8_error_with_detail(
                &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
                format!(
                    "unsupported indexed image class {}",
                    unsupported.class_name()
                ),
            ));
        }
        (NumericStorage::U8(values), ImageIndexMode::Intensity) => values,
        (NumericStorage::I16(values), ImageIndexMode::Intensity) => values
            .into_iter()
            .map(|value| quantize_u16_to_u8((i32::from(value) - i32::from(i16::MIN)) as u16))
            .collect(),
        (NumericStorage::U16(values), ImageIndexMode::Intensity) => {
            values.into_iter().map(quantize_u16_to_u8).collect()
        }
        (NumericStorage::F32(values), ImageIndexMode::Intensity) => values
            .into_iter()
            .map(|value| {
                common::unit_to_dtype(common::clamp01(f64::from(value)), NumericDType::U8) as u8
            })
            .collect(),
        (NumericStorage::F64(values), ImageIndexMode::Intensity) => values
            .into_iter()
            .map(|value| common::unit_to_dtype(common::clamp01(value), NumericDType::U8) as u8)
            .collect(),
        (unsupported, ImageIndexMode::Intensity) => {
            return Err(im2uint8_error_with_detail(
                &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
                format!("unsupported image class {}", unsupported.class_name()),
            ));
        }
    };
    Tensor::from_numeric_storage(NumericStorage::U8(data), shape)
        .map_err(|err| im2uint8_error_with_detail(&IM2UINT8_ERROR_INTERNAL, err))
}

#[cfg(test)]
mod tests;
