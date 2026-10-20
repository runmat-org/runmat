//! MATLAB-compatible `ind2rgb` indexed-image conversion.

use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinErrorDescriptor,
    BuiltinPreparationFormId, IND2RGB_DESCRIPTOR, IND2RGB_ERROR_INTERNAL,
    IND2RGB_ERROR_INVALID_COLORMAP, IND2RGB_ERROR_INVALID_INDEX, IND2RGB_ERROR_INVALID_INPUT,
    IND2RGB_ERROR_TOO_MANY_INPUTS, IND2RGB_ERROR_TOO_MANY_OUTPUTS,
};
#[cfg(test)]
use runmat_builtins::{BuiltinIntegerInputAvailability, IND2RGB_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, Value};

use crate::builtins::image::color::color_presentation::{
    self, ColorHostRoute, ColorHostUpload, SelectedColorPresentation,
};
use crate::builtins::image::color::common;
use crate::context::preparation::{
    execute_request, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "ind2rgb";

fn ind2rgb_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ind2rgb_error_with_message(error.message, error)
}

fn ind2rgb_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ind2rgb_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        build_runtime_error(err.message().to_string())
            .with_builtin(NAME)
            .with_identifier(
                fallback
                    .identifier
                    .expect("closed indexed conversion error"),
            )
            .with_source(err)
            .build()
    }
}

#[runtime_builtin(
    name = "ind2rgb",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::color::ind2rgb"
)]
async fn ind2rgb_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    indexed: Value,
    map: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    if !rest.is_empty() {
        return Err(ind2rgb_error(&IND2RGB_ERROR_TOO_MANY_INPUTS));
    }
    crate::output_count::validate_fixed_outputs(
        &IND2RGB_DESCRIPTOR,
        &IND2RGB_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let admitted = admit_inputs(&indexed, &map)?;
    let resident_sources: Vec<_> = [&indexed, &map]
        .into_iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let selected = if admitted && !resident_sources.is_empty() {
        let indexed_shape = match &indexed {
            Value::Tensor(tensor) => tensor.shape().to_vec(),
            Value::GpuTensor(handle) => handle.shape().to_vec(),
            Value::Num(_) | Value::Int(_) => vec![1, 1],
            _ => unreachable!("admitted indexed input"),
        };
        let mut output_shape = indexed_shape.clone();
        output_shape.push(3);
        color_presentation::select(
            runtime.as_ref(),
            ColorHostUpload {
                route: ColorHostRoute::IndexedToRgb,
                sources: &resident_sources,
                output_shape,
                output_dtype: NumericDType::F64,
                host_compute_scratch: color_presentation::checked_scratch(
                    color_presentation::checked_elements(&indexed_shape),
                    8,
                ),
                control_facts: runmat_execution::Digest::sha256(
                    serde_json::to_vec(&indexed_shape).expect("typed indexed geometry serializes"),
                ),
            },
        )?
    } else {
        None
    };
    if !admitted {
        return ind2rgb_execute(indexed, map, resident_sources, selected).await;
    }
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: BuiltinCatalogIdentity { name: NAME },
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 2,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![indexed, map],
    )?;
    let execute = |mut operands: Vec<Value>| async move {
        let map = operands.pop().expect("admitted colormap operand");
        let indexed = operands.pop().expect("admitted indexed operand");
        ind2rgb_execute(indexed, map, resident_sources, selected).await
    };
    execute_request(runtime.as_ref(), request, execute).await
}

fn admit_inputs(indexed: &Value, map: &Value) -> BuiltinResult<bool> {
    let indexed_admitted = match indexed {
        Value::Num(value) => {
            if !value.is_finite() || value.fract() != 0.0 {
                return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_INDEX));
            }
            true
        }
        Value::Int(IntValue::U8(_) | IntValue::U16(_)) => true,
        Value::Int(value) => {
            return Err(ind2rgb_error_with_message(
                format!("ind2rgb: {} indexed image is not supported; expected single, double, uint8, or uint16", value.class_name()),
                &IND2RGB_ERROR_INVALID_INPUT,
            ));
        }
        Value::Tensor(tensor) => {
            if tensor.shape().len() != 2 {
                return Err(ind2rgb_error_with_message(
                    "ind2rgb: indexed image must be an MxN matrix",
                    &IND2RGB_ERROR_INVALID_INPUT,
                ));
            }
            if !matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
            ) {
                return Err(ind2rgb_error_with_message(
                    format!("ind2rgb: {} indexed image is not supported; expected single, double, uint8, or uint16", tensor.numeric_dtype().class_name()),
                    &IND2RGB_ERROR_INVALID_INPUT,
                ));
            }
            true
        }
        Value::GpuTensor(handle) => {
            let dtype = common::resident_numeric_dtype(handle, NAME)
                .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INPUT))?;
            if !matches!(
                dtype,
                NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
            ) {
                return Err(ind2rgb_error_with_message(
                    "ind2rgb: resident numeric indexed image is not supported; expected single, double, uint8, or uint16",
                    &IND2RGB_ERROR_INVALID_INPUT,
                ));
            }
            if handle.shape().len() != 2 {
                return Err(ind2rgb_error_with_message(
                    "ind2rgb: indexed image must be an MxN matrix",
                    &IND2RGB_ERROR_INVALID_INPUT,
                ));
            }
            true
        }
        Value::LogicalArray(_) | Value::Bool(_) => {
            return Err(ind2rgb_error_with_message(
                "ind2rgb: logical indexed image is not supported; expected single, double, uint8, or uint16",
                &IND2RGB_ERROR_INVALID_INPUT,
            ));
        }
        _ => false,
    };
    let map_admitted = match map {
        Value::Tensor(tensor) => {
            if tensor.numeric_dtype() != NumericDType::F64 {
                return Err(ind2rgb_error_with_message(
                    format!(
                        "ind2rgb: {} colormap is not supported; expected double",
                        tensor.numeric_dtype().class_name()
                    ),
                    &IND2RGB_ERROR_INVALID_COLORMAP,
                ));
            }
            let layout = common::color_layout(tensor, NAME)
                .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_COLORMAP))?;
            if !matches!(layout, common::ColorLayout::Colormap { .. }) {
                return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_COLORMAP));
            }
            true
        }
        Value::GpuTensor(handle) => {
            let dtype = common::resident_numeric_dtype(handle, NAME)
                .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_COLORMAP))?;
            if dtype != NumericDType::F64 {
                return Err(ind2rgb_error_with_message(
                    "ind2rgb: resident colormap is not supported; expected double",
                    &IND2RGB_ERROR_INVALID_COLORMAP,
                ));
            }
            if handle.shape().len() != 2
                || handle.shape().get(1) != Some(&3)
                || handle.shape()[0] == 0
            {
                return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_COLORMAP));
            }
            true
        }
        Value::LogicalArray(_) | Value::Bool(_) => {
            return Err(ind2rgb_error_with_message(
                "ind2rgb: logical colormap is not supported; expected double",
                &IND2RGB_ERROR_INVALID_COLORMAP,
            ));
        }
        _ => false,
    };
    if let Value::Tensor(tensor) = indexed {
        match tensor.numeric_dtype() {
            NumericDType::F64 => {
                if tensor.as_f64_slice().is_some_and(|values| {
                    values
                        .iter()
                        .any(|value| !value.is_finite() || value.fract() != 0.0)
                }) {
                    return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_INDEX));
                }
            }
            NumericDType::F32 => {
                if tensor.as_f32_slice().is_some_and(|values| {
                    values
                        .iter()
                        .any(|value| !value.is_finite() || value.fract() != 0.0)
                }) {
                    return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_INDEX));
                }
            }
            _ => {}
        }
    }
    Ok(indexed_admitted && map_admitted)
}

async fn ind2rgb_execute(
    indexed: Value,
    map: Value,
    resident_sources: Vec<runmat_accelerate_api::GpuTensorHandle>,
    mut selected: Option<SelectedColorPresentation>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let indexed_is_logical = match &indexed {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    };
    let map_is_logical = match &map {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    };
    if let Value::GpuTensor(handle) = &indexed {
        let dtype = common::resident_numeric_dtype(handle, NAME)
            .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INPUT))?;
        if indexed_is_logical
            || !matches!(
                dtype,
                NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
            )
        {
            return Err(ind2rgb_error_with_message(
                format!(
                    "ind2rgb: {} indexed image is not supported; expected single, double, uint8, or uint16",
                    if indexed_is_logical { "logical" } else { "resident numeric" }
                ),
                &IND2RGB_ERROR_INVALID_INPUT,
            ));
        }
        if handle.shape().len() != 2 {
            return Err(ind2rgb_error_with_message(
                "ind2rgb: indexed image must be an MxN matrix",
                &IND2RGB_ERROR_INVALID_INPUT,
            ));
        }
    }
    if let Value::GpuTensor(handle) = &map {
        let dtype = common::resident_numeric_dtype(handle, NAME)
            .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_COLORMAP))?;
        if map_is_logical || dtype != NumericDType::F64 {
            return Err(ind2rgb_error_with_message(
                "ind2rgb: resident colormap is not supported; expected double",
                &IND2RGB_ERROR_INVALID_COLORMAP,
            ));
        }
        if handle.shape().len() != 2 || handle.shape().get(1) != Some(&3) || handle.shape()[0] == 0
        {
            return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_COLORMAP));
        }
    }
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    let indexed = match indexed {
        Value::Tensor(tensor) => tensor,
        other => common::gather_tensor(NAME, other)
            .await
            .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INPUT))?,
    };
    let map = match map {
        Value::Tensor(tensor) => tensor,
        other => common::gather_tensor(NAME, other)
            .await
            .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INPUT))?,
    };
    resident_guard.restore();
    let indexed_dtype = indexed.numeric_dtype();
    if indexed.shape().len() != 2 {
        return Err(ind2rgb_error_with_message(
            "ind2rgb: indexed image must be an MxN matrix",
            &IND2RGB_ERROR_INVALID_INPUT,
        ));
    }
    if indexed_is_logical
        || !matches!(
            indexed_dtype,
            NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
        )
    {
        return Err(ind2rgb_error_with_message(
            format!(
                "ind2rgb: {} indexed image is not supported; expected single, double, uint8, or uint16",
                if indexed_is_logical {
                    "logical"
                } else {
                    indexed_dtype.class_name()
                }
            ),
            &IND2RGB_ERROR_INVALID_INPUT,
        ));
    }
    let map_dtype = map.numeric_dtype();
    if map_is_logical || map_dtype != NumericDType::F64 {
        return Err(ind2rgb_error_with_message(
            format!(
                "ind2rgb: {} colormap is not supported; expected double",
                if map_is_logical {
                    "logical"
                } else {
                    map_dtype.class_name()
                }
            ),
            &IND2RGB_ERROR_INVALID_COLORMAP,
        ));
    }
    let layout = common::color_layout(&map, NAME)
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_COLORMAP))?;
    let common::ColorLayout::Colormap { rows: map_rows } = layout else {
        return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_COLORMAP));
    };

    let pixels = indexed.len();
    let mut shape = indexed.shape().to_vec();
    shape.push(3);
    let indexed_values = common::tensor_values_f64(&indexed);
    let map_values = common::tensor_values_f64(&map);
    let mut data = vec![0.0; pixels * 3];
    for (pixel, raw_index) in indexed_values.iter().copied().enumerate() {
        let map_index = map_index(raw_index, indexed_dtype, map_rows)
            .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INVALID_INDEX))?;
        for channel in 0..3 {
            data[pixel + pixels * channel] = map_values[layout.index(map_index, channel)];
        }
    }

    let out = common::tensor_with_dtype(data, shape, NumericDType::F64, NAME)
        .map_err(|err| ind2rgb_map_error(err, &IND2RGB_ERROR_INTERNAL))?;
    let output = color_presentation::publish(&mut selected, || {
        common::restore_resident_numeric_result_for_sources(
            &resident_sources,
            common::image_value_from_tensor(out),
            NAME,
        )
    })?;
    crate::execution::RootedValue::new(output)
        .map_err(|err| ind2rgb_map_error(err.into_runtime_error(), &IND2RGB_ERROR_INTERNAL))
}

fn map_index(value: f64, dtype: NumericDType, map_rows: usize) -> BuiltinResult<usize> {
    if !value.is_finite()
        || (matches!(dtype, NumericDType::F32 | NumericDType::F64) && value.fract() != 0.0)
    {
        return Err(ind2rgb_error(&IND2RGB_ERROR_INVALID_INDEX));
    }
    let index = if matches!(dtype, NumericDType::U8 | NumericDType::U16) {
        (value as isize).clamp(0, map_rows as isize - 1)
    } else {
        (value as isize).clamp(1, map_rows as isize) - 1
    };
    Ok(index as usize)
}

#[cfg(test)]
mod tests;
