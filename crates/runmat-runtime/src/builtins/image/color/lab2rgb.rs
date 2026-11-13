//! MATLAB-compatible default-D65 `lab2rgb` conversion.

use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinErrorDescriptor,
    BuiltinPreparationFormId, LabOutputType, LAB2RGB_DESCRIPTOR, LAB2RGB_ERROR_INTERNAL,
    LAB2RGB_ERROR_INVALID_INPUT, LAB2RGB_ERROR_INVALID_OPTION, LAB2RGB_ERROR_TOO_MANY_INPUTS,
    LAB2RGB_ERROR_TOO_MANY_OUTPUTS, LAB2RGB_EXTENSIONS,
};
use runmat_macros::runtime_builtin;
use runmat_value::NumericStorage;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::image::color::color_presentation::{
    self, ColorHostRoute, ColorHostUpload, SelectedColorPresentation,
};
use crate::builtins::image::color::common;
use crate::context::preparation::{
    execute_request, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lab2rgb";
const XN: f64 = 0.95047;
const YN: f64 = 1.0;
const ZN: f64 = 1.08883;
const EPSILON: f64 = 216.0 / 24389.0;
const KAPPA: f64 = 24389.0 / 27.0;

fn lab2rgb_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    lab2rgb_error_with_message(error.message, error)
}

fn lab2rgb_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn lab2rgb_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        build_runtime_error(err.message().to_string())
            .with_builtin(NAME)
            .with_identifier(fallback.identifier.expect("closed Lab conversion error"))
            .with_source(err)
            .build()
    }
}

#[runtime_builtin(
    name = "lab2rgb",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::color::lab2rgb"
)]
async fn lab2rgb_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    lab: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let output_type = parse_output_type(rest.view())?;
    crate::output_count::validate_fixed_outputs(
        &LAB2RGB_DESCRIPTOR,
        &LAB2RGB_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let admitted = admit_lab(&lab)?;
    let resident_sources = match &lab {
        Value::GpuTensor(handle) => vec![handle.clone()],
        _ => Vec::new(),
    };
    let selected = match &lab {
        Value::GpuTensor(handle) => {
            let input_dtype = match runmat_accelerate_api::handle_precision(handle) {
                Some(runmat_accelerate_api::ProviderPrecision::F32) => NumericDType::F32,
                _ => NumericDType::F64,
            };
            let output_dtype = output_type.map_or(input_dtype, |output| match output {
                LabOutputType::Double => NumericDType::F64,
                LabOutputType::Single => NumericDType::F32,
                LabOutputType::UInt8 => NumericDType::U8,
                LabOutputType::UInt16 => NumericDType::U16,
            });
            color_presentation::select(
                runtime.as_ref(),
                ColorHostUpload {
                    route: ColorHostRoute::LabToRgb,
                    sources: &resident_sources,
                    output_shape: handle.shape().to_vec(),
                    output_dtype,
                    host_compute_scratch: color_presentation::checked_scratch(
                        color_presentation::checked_elements(handle.shape()),
                        16,
                    ),
                    control_facts: runmat_execution::Digest::sha256(
                        serde_json::to_vec(&(input_dtype, output_dtype))
                            .expect("typed Lab output facts serialize"),
                    ),
                },
            )?
        }
        _ => None,
    };
    if !admitted {
        return lab2rgb_execute(lab, output_type, resident_sources, selected).await;
    }
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: BuiltinCatalogIdentity { name: NAME },
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: rest.len() + 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![lab],
    )?;
    let execute = |mut operands: Vec<Value>| async move {
        lab2rgb_execute(
            operands.pop().expect("admitted Lab operand"),
            output_type,
            resident_sources,
            selected,
        )
        .await
    };
    execute_request(runtime.as_ref(), request, execute).await
}

fn admit_lab(lab: &Value) -> BuiltinResult<bool> {
    match lab {
        Value::Tensor(tensor) => {
            common::color_layout(tensor, NAME)
                .map_err(|err| lab2rgb_map_error(err, &LAB2RGB_ERROR_INVALID_INPUT))?;
            if !matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) {
                return Err(lab2rgb_error_with_message(
                    format!(
                        "lab2rgb: {} input is not supported; expected single or double",
                        tensor.numeric_dtype().class_name()
                    ),
                    &LAB2RGB_ERROR_INVALID_INPUT,
                ));
            }
            Ok(true)
        }
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_explicit(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LAB2RGB_EXTENSIONS[0],
                    NAME,
                )?;
            }
            if runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
            {
                return Err(lab2rgb_error_with_message(
                    "lab2rgb: integer and logical LAB input is not supported; expected single or double",
                    &LAB2RGB_ERROR_INVALID_INPUT,
                ));
            }
            let dtype = common::resident_numeric_dtype(handle, NAME)
                .map_err(|err| lab2rgb_map_error(err, &LAB2RGB_ERROR_INVALID_INPUT))?;
            if !matches!(dtype, NumericDType::F32 | NumericDType::F64)
                || !((handle.shape().len() == 3
                    && handle.shape()[2] == 3
                    && handle.shape()[0] > 0
                    && handle.shape()[1] > 0)
                    || (handle.shape().len() == 2
                        && handle.shape()[1] == 3
                        && handle.shape()[0] > 0))
            {
                return Err(lab2rgb_error_with_message(
                    format!(
                        "lab2rgb: unsupported resident class {} or shape {:?}",
                        dtype.class_name(),
                        handle.shape()
                    ),
                    &LAB2RGB_ERROR_INVALID_INPUT,
                ));
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

async fn lab2rgb_execute(
    lab: Value,
    output_type: Option<LabOutputType>,
    resident_sources: Vec<runmat_accelerate_api::GpuTensorHandle>,
    mut selected: Option<SelectedColorPresentation>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let logical_input = match &lab {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    };
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    let tensor = match lab {
        Value::Tensor(tensor) => tensor,
        other => common::gather_tensor(NAME, other)
            .await
            .map_err(|err| lab2rgb_map_error(err, &LAB2RGB_ERROR_INVALID_INPUT))?,
    };
    resident_guard.restore();
    let layout = common::color_layout(&tensor, NAME)
        .map_err(|err| lab2rgb_map_error(err, &LAB2RGB_ERROR_INVALID_INPUT))?;
    let dtype = tensor.numeric_dtype();
    if logical_input || !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(lab2rgb_error_with_message(
            format!(
                "lab2rgb: {} input is not supported; expected single or double",
                if logical_input {
                    "logical"
                } else {
                    dtype.class_name()
                }
            ),
            &LAB2RGB_ERROR_INVALID_INPUT,
        ));
    }
    let values = common::tensor_values_f64(&tensor);
    let mut data = vec![0.0; values.len()];
    for pixel in 0..layout.pixels() {
        let l = values[layout.index(pixel, 0)];
        let a = values[layout.index(pixel, 1)];
        let b = values[layout.index(pixel, 2)];
        let (r, g, blue) = lab_to_rgb_unit(l, a, b);
        data[layout.index(pixel, 0)] = cast_float(r, dtype);
        data[layout.index(pixel, 1)] = cast_float(g, dtype);
        data[layout.index(pixel, 2)] = cast_float(blue, dtype);
    }
    let output_type = output_type.map_or(dtype, |output| match output {
        LabOutputType::Double => NumericDType::F64,
        LabOutputType::Single => NumericDType::F32,
        LabOutputType::UInt8 => NumericDType::U8,
        LabOutputType::UInt16 => NumericDType::U16,
    });
    let storage = lab2rgb_output_storage(data, output_type);
    let out = Tensor::from_numeric_storage(storage, layout.output_shape())
        .map_err(|err| lab2rgb_error_with_message(err, &LAB2RGB_ERROR_INTERNAL))?;
    let output = color_presentation::publish(&mut selected, || {
        common::restore_resident_numeric_result_for_sources(
            &resident_sources,
            common::image_value_from_tensor(out),
            NAME,
        )
    })?;
    crate::execution::RootedValue::new(output)
        .map_err(|err| lab2rgb_map_error(err.into_runtime_error(), &LAB2RGB_ERROR_INTERNAL))
}

fn parse_output_type(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<LabOutputType>> {
    if rest.len() > 2 {
        return Err(lab2rgb_error(&LAB2RGB_ERROR_TOO_MANY_INPUTS));
    }
    if rest.is_empty() {
        return Ok(None);
    }
    if rest.len() != 2
        || !tensor::value_to_string(&rest[0])
            .is_some_and(|name| name.eq_ignore_ascii_case("OutputType"))
    {
        return Err(lab2rgb_error(&LAB2RGB_ERROR_INVALID_OPTION));
    }
    let output = tensor::value_to_string(&rest[1])
        .ok_or_else(|| lab2rgb_error(&LAB2RGB_ERROR_INVALID_OPTION))?;
    LabOutputType::from_name(&output)
        .map(Some)
        .ok_or_else(|| lab2rgb_error(&LAB2RGB_ERROR_INVALID_OPTION))
}

fn lab2rgb_output_storage(values: Vec<f64>, dtype: NumericDType) -> NumericStorage {
    match dtype {
        NumericDType::F64 => NumericStorage::F64(values),
        NumericDType::F32 => {
            NumericStorage::F32(values.into_iter().map(|value| value as f32).collect())
        }
        NumericDType::U8 => NumericStorage::U8(
            values
                .into_iter()
                .map(|value| normalized_integer_output(value, u8::MAX as f64) as u8)
                .collect(),
        ),
        NumericDType::U16 => NumericStorage::U16(
            values
                .into_iter()
                .map(|value| normalized_integer_output(value, u16::MAX as f64) as u16)
                .collect(),
        ),
        _ => unreachable!("lab2rgb OutputType parser accepts only floating, uint8, or uint16"),
    }
}

fn normalized_integer_output(value: f64, maximum: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        (value.clamp(0.0, 1.0) * maximum).round()
    }
}

pub(crate) fn lab_to_rgb_unit(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let x = XN * lab_f_inv(fx);
    let y = YN * lab_f_inv(fy);
    let z = ZN * lab_f_inv(fz);
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let blue = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(blue))
}

fn lab_f_inv(value: f64) -> f64 {
    let cubed = value * value * value;
    if cubed > EPSILON {
        cubed
    } else {
        (116.0 * value - 16.0) / KAPPA
    }
}

pub(crate) fn linear_to_srgb(value: f64) -> f64 {
    if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn cast_float(value: f64, dtype: NumericDType) -> f64 {
    if matches!(dtype, NumericDType::F32) {
        (value as f32) as f64
    } else {
        value
    }
}

#[cfg(test)]
mod tests;
