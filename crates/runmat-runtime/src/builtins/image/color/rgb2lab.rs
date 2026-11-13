//! MATLAB-compatible `rgb2lab` color-space conversion.

use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinErrorDescriptor,
    BuiltinPreparationFormId, Rgb2LabOptionName, RgbColorSpace, RgbWhitePoint, RGB2LAB_DESCRIPTOR,
    RGB2LAB_ERROR_INTERNAL, RGB2LAB_ERROR_INVALID_INPUT, RGB2LAB_ERROR_INVALID_OPTION,
    RGB2LAB_ERROR_TOO_MANY_INPUTS, RGB2LAB_ERROR_TOO_MANY_OUTPUTS, RGB2LAB_EXTENSIONS,
};
#[cfg(test)]
use runmat_builtins::{BuiltinIntegerInputAvailability, RGB2LAB_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::tensor;
use crate::builtins::image::color::color_presentation::{
    self, ColorHostRoute, ColorHostUpload, SelectedColorPresentation,
};
use crate::builtins::image::color::common;
use crate::context::preparation::{
    execute_request, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rgb2lab";
const XN: f64 = 0.95047;
const YN: f64 = 1.0;
const ZN: f64 = 1.08883;
const EPSILON: f64 = 216.0 / 24389.0;
const KAPPA: f64 = 24389.0 / 27.0;

fn rgb2lab_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rgb2lab_error_with_message(error.message, error)
}

fn rgb2lab_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rgb2lab_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        build_runtime_error(err.message().to_string())
            .with_builtin(NAME)
            .with_identifier(fallback.identifier.expect("closed RGB to Lab error"))
            .with_source(err)
            .build()
    }
}

#[runtime_builtin(
    name = "rgb2lab",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::color::rgb2lab"
)]
async fn rgb2lab_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rgb: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let options = parse_rgb2lab_options(rest.view())?;
    crate::output_count::validate_fixed_outputs(
        &RGB2LAB_DESCRIPTOR,
        &RGB2LAB_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let admitted = admit_rgb(&rgb)?;
    let resident_sources = match &rgb {
        Value::GpuTensor(handle) => vec![handle.clone()],
        _ => Vec::new(),
    };
    let selected = match &rgb {
        Value::GpuTensor(handle) => {
            let output_dtype = if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                NumericDType::F32
            } else {
                NumericDType::F64
            };
            let color_space = match options.color_space {
                RgbColorSpace::Srgb => 0u8,
                RgbColorSpace::AdobeRgb1998 => 1,
                RgbColorSpace::ProphotoRgb => 2,
                RgbColorSpace::LinearRgb => 3,
            };
            color_presentation::select(
                runtime.as_ref(),
                ColorHostUpload {
                    route: ColorHostRoute::RgbToLab,
                    sources: &resident_sources,
                    output_shape: handle.shape().to_vec(),
                    output_dtype,
                    host_compute_scratch: color_presentation::checked_scratch(
                        color_presentation::checked_elements(handle.shape()),
                        16,
                    ),
                    control_facts: runmat_execution::Digest::sha256(
                        serde_json::to_vec(&(color_space, options.white_point))
                            .expect("typed RGB to Lab facts serialize"),
                    ),
                },
            )?
        }
        _ => None,
    };
    if !admitted {
        return rgb2lab_execute(rgb, options, resident_sources, selected).await;
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
        vec![rgb],
    )?;
    let execute = |mut operands: Vec<Value>| async move {
        rgb2lab_execute(
            operands.pop().expect("admitted RGB operand"),
            options,
            resident_sources,
            selected,
        )
        .await
    };
    execute_request(runtime.as_ref(), request, execute).await
}

fn admit_rgb(rgb: &Value) -> BuiltinResult<bool> {
    match rgb {
        Value::Tensor(tensor) => {
            common::stacked_color_layout(tensor, NAME)
                .map_err(|err| rgb2lab_map_error(err, &RGB2LAB_ERROR_INVALID_INPUT))?;
            if !matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
            ) {
                return Err(rgb2lab_error_with_message(
                    format!("rgb2lab: {} input is not supported; expected single, double, uint8, or uint16", tensor.numeric_dtype().class_name()),
                    &RGB2LAB_ERROR_INVALID_INPUT,
                ));
            }
            Ok(true)
        }
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_explicit(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &RGB2LAB_EXTENSIONS[0],
                    NAME,
                )?;
            }
            validate_resident_rgb(handle)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

async fn rgb2lab_execute(
    rgb: Value,
    options: Rgb2LabOptions,
    resident_sources: Vec<runmat_accelerate_api::GpuTensorHandle>,
    mut selected: Option<SelectedColorPresentation>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    let tensor = match rgb {
        Value::Tensor(tensor) => tensor,
        other => common::gather_tensor(NAME, other)
            .await
            .map_err(|err| rgb2lab_map_error(err, &RGB2LAB_ERROR_INVALID_INPUT))?,
    };
    resident_guard.restore();
    let layout = common::stacked_color_layout(&tensor, NAME)
        .map_err(|err| rgb2lab_map_error(err, &RGB2LAB_ERROR_INVALID_INPUT))?;
    let input_dtype = tensor.numeric_dtype();
    let supported = matches!(
        input_dtype,
        NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
    );
    if !supported {
        return Err(rgb2lab_error_with_message(
            format!(
                "rgb2lab: {} input is not supported; expected single, double, uint8, or uint16",
                input_dtype.class_name()
            ),
            &RGB2LAB_ERROR_INVALID_INPUT,
        ));
    }
    let dtype = common::image_output_dtype(input_dtype);
    let values = common::tensor_values_f64(&tensor);
    let mut data = vec![0.0; values.len()];
    for pixel in 0..layout.pixels() {
        let r = common::clamp01(common::unit_value(
            values[layout.index(pixel, 0)],
            input_dtype,
        ));
        let g = common::clamp01(common::unit_value(
            values[layout.index(pixel, 1)],
            input_dtype,
        ));
        let b = common::clamp01(common::unit_value(
            values[layout.index(pixel, 2)],
            input_dtype,
        ));
        let (l, a, bstar) = rgb_to_lab(r, g, b, options);
        data[layout.index(pixel, 0)] = cast_float(l, dtype);
        data[layout.index(pixel, 1)] = cast_float(a, dtype);
        data[layout.index(pixel, 2)] = cast_float(bstar, dtype);
    }
    let out = common::tensor_with_dtype(data, layout.output_shape(), dtype, NAME)
        .map_err(|err| rgb2lab_map_error(err, &RGB2LAB_ERROR_INTERNAL))?;
    let output = color_presentation::publish(&mut selected, || {
        common::restore_resident_numeric_result_for_sources(
            &resident_sources,
            common::image_value_from_tensor(out),
            NAME,
        )
    })?;
    crate::execution::RootedValue::new(output)
        .map_err(|err| rgb2lab_map_error(err.into_runtime_error(), &RGB2LAB_ERROR_INTERNAL))
}

fn validate_resident_rgb(handle: &runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<()> {
    let dtype = common::resident_numeric_dtype(handle, NAME)
        .map_err(|err| rgb2lab_map_error(err, &RGB2LAB_ERROR_INVALID_INPUT))?;
    let image_shape = handle.shape().len() >= 3
        && handle.shape()[0] > 0
        && handle.shape()[1] > 0
        && handle.shape().get(2) == Some(&3);
    let colormap_shape =
        handle.shape().len() == 2 && handle.shape()[0] > 0 && handle.shape().get(1) == Some(&3);
    let supported = (image_shape || colormap_shape)
        && matches!(
            dtype,
            NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
        );
    if !supported {
        return Err(rgb2lab_error_with_message(
            format!(
                "rgb2lab: unsupported resident class {} or shape {:?}",
                dtype.class_name(),
                handle.shape()
            ),
            &RGB2LAB_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Rgb2LabOptions {
    color_space: RgbColorSpace,
    white_point: [f64; 3],
}

impl Default for Rgb2LabOptions {
    fn default() -> Self {
        Self {
            color_space: RgbColorSpace::Srgb,
            white_point: [XN, YN, ZN],
        }
    }
}

fn parse_rgb2lab_options(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Rgb2LabOptions> {
    if rest.len() > 4 {
        return Err(rgb2lab_error(&RGB2LAB_ERROR_TOO_MANY_INPUTS));
    }
    if !rest.len().is_multiple_of(2) {
        return Err(rgb2lab_error_with_message(
            "rgb2lab: name-value arguments must appear in pairs",
            &RGB2LAB_ERROR_INVALID_OPTION,
        ));
    }
    let mut options = Rgb2LabOptions::default();
    for pair in rest.chunks_exact(2) {
        let name = tensor::value_to_string(&pair[0])
            .ok_or_else(|| rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION))?;
        match Rgb2LabOptionName::from_name(&name) {
            Some(Rgb2LabOptionName::ColorSpace) => {
                let value = tensor::value_to_string(&pair[1])
                    .ok_or_else(|| rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION))?;
                options.color_space = RgbColorSpace::from_name(&value)
                    .ok_or_else(|| rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION))?;
            }
            Some(Rgb2LabOptionName::WhitePoint) => {
                options.white_point = parse_white_point(&pair[1])?
            }
            _ => return Err(rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION)),
        }
    }
    Ok(options)
}

fn parse_white_point(value: &Value) -> BuiltinResult<[f64; 3]> {
    if let Some(name) = tensor::value_to_string(value) {
        return RgbWhitePoint::from_name(&name)
            .map(RgbWhitePoint::coordinates)
            .ok_or_else(|| rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION));
    }
    let Value::Tensor(tensor) = value else {
        return Err(rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION));
    };
    if tensor.shape() != [1, 3]
        || !matches!(
            tensor.numeric_dtype(),
            NumericDType::F32 | NumericDType::F64
        )
    {
        return Err(rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION));
    }
    let values = tensor.materialize_f64();
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION));
    }
    Ok([values[0], values[1], values[2]])
}

fn rgb_to_lab(r: f64, g: f64, b: f64, options: Rgb2LabOptions) -> (f64, f64, f64) {
    let (linear, matrix, source_white) = match options.color_space {
        RgbColorSpace::Srgb => (
            [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)],
            SRGB_TO_XYZ,
            [XN, YN, ZN],
        ),
        RgbColorSpace::LinearRgb => ([r, g, b], SRGB_TO_XYZ, [XN, YN, ZN]),
        RgbColorSpace::AdobeRgb1998 => (
            [
                r.powf(563.0 / 256.0),
                g.powf(563.0 / 256.0),
                b.powf(563.0 / 256.0),
            ],
            ADOBE_TO_XYZ,
            [XN, YN, ZN],
        ),
        RgbColorSpace::ProphotoRgb => (
            [
                prophoto_to_linear(r),
                prophoto_to_linear(g),
                prophoto_to_linear(b),
            ],
            PROPHOTO_TO_XYZ,
            [0.9642, 1.0, 0.8251],
        ),
    };
    let xyz = multiply_matrix_vector(matrix, linear);
    let [x, y, z] = adapt_xyz(xyz, source_white, options.white_point);
    let fx = lab_f(x / options.white_point[0]);
    let fy = lab_f(y / options.white_point[1]);
    let fz = lab_f(z / options.white_point[2]);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];
const ADOBE_TO_XYZ: [[f64; 3]; 3] = [
    [0.5767309, 0.1855540, 0.1881852],
    [0.2973769, 0.6273491, 0.0752741],
    [0.0270343, 0.0706872, 0.9911085],
];
const PROPHOTO_TO_XYZ: [[f64; 3]; 3] = [
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0, 0.0, 0.8252100],
];
const BRADFORD: [[f64; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];
const BRADFORD_INVERSE: [[f64; 3]; 3] = [
    [0.9869929, -0.1470543, 0.1599627],
    [0.4323053, 0.5183603, 0.0492912],
    [-0.0085287, 0.0400428, 0.9684867],
];

fn multiply_matrix_vector(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0] * value[0] + row[1] * value[1] + row[2] * value[2])
}

fn adapt_xyz(xyz: [f64; 3], source_white: [f64; 3], target_white: [f64; 3]) -> [f64; 3] {
    let source_cone = multiply_matrix_vector(BRADFORD, source_white);
    let target_cone = multiply_matrix_vector(BRADFORD, target_white);
    let mut cone = multiply_matrix_vector(BRADFORD, xyz);
    for channel in 0..3 {
        cone[channel] *= target_cone[channel] / source_cone[channel];
    }
    multiply_matrix_vector(BRADFORD_INVERSE, cone)
}

fn prophoto_to_linear(value: f64) -> f64 {
    if value <= 16.0 / 512.0 {
        value / 16.0
    } else {
        value.powf(1.8)
    }
}

pub(crate) fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn lab_f(value: f64) -> f64 {
    if value > EPSILON {
        value.cbrt()
    } else {
        (KAPPA * value + 16.0) / 116.0
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
