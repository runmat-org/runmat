//! MATLAB-compatible `rgb2lab` color-space conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rgb2lab";
const XN: f64 = 0.95047;
const YN: f64 = 1.0;
const ZN: f64 = 1.08883;
const EPSILON: f64 = 216.0 / 24389.0;
const KAPPA: f64 = 24389.0 / 27.0;

const RGB2LAB_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "LAB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "CIE L*a*b* image or colormap converted from RGB input.",
}];

const RGB2LAB_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "RGB",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RGB image, image stack, or c-by-3 colormap values.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "ColorSpace or WhitePoint name-value options.",
    },
];

const RGB2LAB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "LAB = rgb2lab(RGB, Name=Value)",
    inputs: &RGB2LAB_INPUTS,
    outputs: &RGB2LAB_OUTPUT,
}];

const RGB2LAB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2LAB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:rgb2lab:TooManyInputs"),
    when: "More than the two supported name-value pairs are supplied.",
    message: "rgb2lab: too many input arguments",
};

const RGB2LAB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2LAB.INVALID_INPUT",
    identifier: Some("RunMat:rgb2lab:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 RGB image or Nx3 RGB colormap.",
    message: "rgb2lab: invalid input",
};

const RGB2LAB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2LAB.INTERNAL",
    identifier: Some("RunMat:rgb2lab:Internal"),
    when: "L*a*b* output tensor construction fails internally.",
    message: "rgb2lab: internal conversion failure",
};

const RGB2LAB_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2LAB.INVALID_OPTION",
    identifier: Some("RunMat:rgb2lab:InvalidOption"),
    when: "A ColorSpace or WhitePoint name-value argument is malformed or unsupported.",
    message: "rgb2lab: invalid name-value option",
};

const RGB2LAB_ERRORS: [BuiltinErrorDescriptor; 4] = [
    RGB2LAB_ERROR_TOO_MANY_INPUTS,
    RGB2LAB_ERROR_INVALID_INPUT,
    RGB2LAB_ERROR_INTERNAL,
    RGB2LAB_ERROR_INVALID_OPTION,
];

pub const RGB2LAB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RGB2LAB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RGB2LAB_ERRORS,
};

const RGB2LAB_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rgb2lab-explicit-gpu",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rgb2lab with an explicitly requested gpuArray input is a RunMat extension because the compatibility target documents no interactive GPU Arrays capability",
    error_identifier: Some("RunMat:compatibility:Rgb2labExplicitGpuExtension"),
};
pub const RGB2LAB_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [RGB2LAB_EXPLICIT_GPU_EXTENSION];

const RGB2LAB_DOCUMENTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RGB",
        classes: &common::DOCUMENTED_IMAGE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "uint8 and uint16 RGB images, MxNx3xP stacks, and c-by-3 colormaps are documented and scale to double L*a*b* output.",
    }];
const RGB2LAB_REJECTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RGB",
        classes: &common::REJECTED_IMAGE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Signed integer and uint32/uint64 RGB values are outside the documented surface and reject before resident download.",
    }];
pub const RGB2LAB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "LAB = rgb2lab(integer_RGB)",
        inputs: &RGB2LAB_DOCUMENTED_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative uint8/uint16 samples are normalized by their full class range before the documented floating color-space conversion and produce double output. Automatic residency may gather to the documented host implementation; the separately gated explicit-gpuArray extension restores output to the exact owner or errors.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "rgb2lab(unsupported_integer_RGB)",
        inputs: &RGB2LAB_REJECTED_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer classes reject consistently on host and from resident dtype metadata.",
    },
];

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
        rgb2lab_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::rgb2lab")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rgb2lab"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation supports documented sRGB, Adobe RGB (1998), ProPhoto RGB, linear sRGB, and reference-white conversion paths.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::rgb2lab")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; conversion is channel-coupled and piecewise.",
};

#[runtime_builtin(
    name = "rgb2lab",
    category = "image/color",
    summary = "Convert RGB image or colormap values to CIE L*a*b*.",
    keywords = "rgb2lab,rgb,lab,cielab,color,image,colormap",
    accel = "sink",
    type_resolver(same_shape_type),
    descriptor(crate::builtins::image::color::rgb2lab::RGB2LAB_DESCRIPTOR),
    extensions(crate::builtins::image::color::rgb2lab::RGB2LAB_EXTENSIONS),
    integer_capabilities(crate::builtins::image::color::rgb2lab::RGB2LAB_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::image::color::rgb2lab"
)]
async fn rgb2lab_builtin(rgb: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let options = parse_rgb2lab_options(&rest)?;
    let mut resident_sources = Vec::new();
    if let Value::GpuTensor(handle) = &rgb {
        if runmat_accelerate_api::handle_is_explicit(handle) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &RGB2LAB_EXPLICIT_GPU_EXTENSION,
                NAME,
            )?;
        }
        validate_resident_rgb(handle)?;
        resident_sources.push(handle.clone());
    }
    let resident_guard = common::protect_resident_inputs(&resident_sources);
    let tensor = common::gather_tensor(NAME, rgb)
        .await
        .map_err(|err| rgb2lab_map_error(err, &RGB2LAB_ERROR_INVALID_INPUT))?;
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
    common::restore_resident_numeric_result_for_sources(
        &resident_sources,
        common::image_value_from_tensor(out),
        NAME,
    )
}

fn validate_resident_rgb(handle: &runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<()> {
    let dtype = common::resident_numeric_dtype(handle, NAME)
        .map_err(|err| rgb2lab_map_error(err, &RGB2LAB_ERROR_INVALID_INPUT))?;
    let image_shape = handle.shape.len() >= 3
        && handle.shape[0] > 0
        && handle.shape[1] > 0
        && handle.shape.get(2) == Some(&3);
    let colormap_shape =
        handle.shape.len() == 2 && handle.shape[0] > 0 && handle.shape.get(1) == Some(&3);
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
                handle.shape
            ),
            &RGB2LAB_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputColorSpace {
    Srgb,
    AdobeRgb1998,
    ProphotoRgb,
    LinearRgb,
}

#[derive(Clone, Copy, Debug)]
struct Rgb2LabOptions {
    color_space: InputColorSpace,
    white_point: [f64; 3],
}

impl Default for Rgb2LabOptions {
    fn default() -> Self {
        Self {
            color_space: InputColorSpace::Srgb,
            white_point: [XN, YN, ZN],
        }
    }
}

fn parse_rgb2lab_options(rest: &[Value]) -> BuiltinResult<Rgb2LabOptions> {
    if rest.len() > 4 {
        return Err(rgb2lab_error(&RGB2LAB_ERROR_TOO_MANY_INPUTS));
    }
    if rest.len() % 2 != 0 {
        return Err(rgb2lab_error_with_message(
            "rgb2lab: name-value arguments must appear in pairs",
            &RGB2LAB_ERROR_INVALID_OPTION,
        ));
    }
    let mut options = Rgb2LabOptions::default();
    for pair in rest.chunks_exact(2) {
        let name = tensor::value_to_string(&pair[0])
            .ok_or_else(|| rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION))?
            .to_ascii_lowercase();
        match name.as_str() {
            "colorspace" => {
                let value = tensor::value_to_string(&pair[1])
                    .ok_or_else(|| rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION))?
                    .to_ascii_lowercase();
                options.color_space = match value.as_str() {
                    "srgb" => InputColorSpace::Srgb,
                    "adobe-rgb-1998" => InputColorSpace::AdobeRgb1998,
                    "prophoto-rgb" => InputColorSpace::ProphotoRgb,
                    "linear-rgb" => InputColorSpace::LinearRgb,
                    _ => return Err(rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION)),
                };
            }
            "whitepoint" => options.white_point = parse_white_point(&pair[1])?,
            _ => return Err(rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION)),
        }
    }
    Ok(options)
}

fn parse_white_point(value: &Value) -> BuiltinResult<[f64; 3]> {
    if let Some(name) = tensor::value_to_string(value) {
        return match name.to_ascii_lowercase().as_str() {
            "a" => Ok([1.0985, 1.0, 0.3558]),
            "c" => Ok([0.9807, 1.0, 1.1822]),
            "e" => Ok([1.0, 1.0, 1.0]),
            "d50" => Ok([0.9642, 1.0, 0.8251]),
            "d55" => Ok([0.9568, 1.0, 0.9214]),
            "d65" => Ok([XN, YN, ZN]),
            "icc" => Ok([31595.0 / 32768.0, 1.0, 27030.0 / 32768.0]),
            _ => Err(rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION)),
        };
    }
    let Value::Tensor(tensor) = value else {
        return Err(rgb2lab_error(&RGB2LAB_ERROR_INVALID_OPTION));
    };
    if tensor.shape != [1, 3]
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
        InputColorSpace::Srgb => (
            [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)],
            SRGB_TO_XYZ,
            [XN, YN, ZN],
        ),
        InputColorSpace::LinearRgb => ([r, g, b], SRGB_TO_XYZ, [XN, YN, ZN]),
        InputColorSpace::AdobeRgb1998 => (
            [
                r.powf(563.0 / 256.0),
                g.powf(563.0 / 256.0),
                b.powf(563.0 / 256.0),
            ],
            ADOBE_TO_XYZ,
            [XN, YN, ZN],
        ),
        InputColorSpace::ProphotoRgb => (
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
mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, Tensor};

    fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
        let Value::Tensor(out) =
            block_on(rgb2lab_builtin(Value::Tensor(tensor), Vec::new())).expect("rgb2lab")
        else {
            panic!("expected tensor");
        };
        Ok(out)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn converts_white_to_lab_reference() {
        let rgb = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let out = call(rgb).unwrap();
        let values = values(&out);
        assert_close(values[0], 100.0, 1e-4);
        assert_close(values[1], 0.0, 1e-3);
        assert_close(values[2], 0.0, 1e-3);
    }

    #[test]
    fn rgb2lab_preserves_single_output_class() {
        let rgb = Tensor::from_f32(vec![1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let out = call(rgb).expect("single RGB");
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        let values = values(&out);
        assert_close(values[0], 100.0, 1e-4);
        assert_close(values[1], 0.0, 1e-3);
        assert_close(values[2], 0.0, 1e-3);
    }

    #[test]
    fn converts_rgb_colormap_to_lab_references() {
        let rgb = Tensor::new(
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![4, 3],
        )
        .unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.shape, vec![4, 3]);
        let expected = vec![
            0.0, 53.2408, 87.7347, 32.2970, 0.0, 80.0925, -86.1827, 79.1875, 0.0, 67.2032, 83.1793,
            -107.8602,
        ];
        for (actual, expected) in values(&out).iter().zip(expected) {
            assert_close(*actual, expected, 1e-3);
        }
    }

    #[test]
    fn accepts_documented_single_uint8_and_uint16_colormaps() {
        let single = Tensor::from_f32(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        assert_eq!(call(single).unwrap().numeric_dtype(), NumericDType::F32);
        for storage in [
            IntegerStorage::U8(vec![255, 255, 255]),
            IntegerStorage::U16(vec![65535, 65535, 65535]),
        ] {
            let out = call(Tensor::new_integer(storage, vec![1, 3]).unwrap()).unwrap();
            assert_eq!(out.numeric_dtype(), NumericDType::F64);
            assert_close(values(&out)[0], 100.0, 1e-4);
        }
    }

    #[test]
    fn supports_documented_color_space_and_white_point_options() {
        let rgb = Tensor::new(vec![0.2, 0.3, 0.4], vec![1, 3]).unwrap();
        let Value::Tensor(adobe) = block_on(rgb2lab_builtin(
            Value::Tensor(rgb.clone()),
            vec![
                Value::String("ColorSpace".into()),
                Value::String("adobe-rgb-1998".into()),
            ],
        ))
        .expect("Adobe RGB option") else {
            panic!("expected tensor")
        };
        for (actual, expected) in values(&adobe).iter().zip([30.1783, -5.6902, -20.8223]) {
            assert_close(*actual, expected, 2e-2);
        }

        let Value::Tensor(d50) = block_on(rgb2lab_builtin(
            Value::Tensor(rgb),
            vec![
                Value::String("WhitePoint".into()),
                Value::String("d50".into()),
            ],
        ))
        .expect("D50 option") else {
            panic!("expected tensor")
        };
        for (actual, expected) in values(&d50).iter().zip([31.3294, -4.0732, -18.1750]) {
            assert_close(*actual, expected, 2e-2);
        }
    }

    #[test]
    fn converts_m_by_n_by_three_by_p_image_stacks_without_crossing_planes() {
        let rgb = Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0], vec![1, 1, 3, 2]).unwrap();
        let out = call(rgb).expect("stacked RGB");
        assert_eq!(out.shape, vec![1, 1, 3, 2]);
        let values = values(&out);
        assert_close(values[0], 100.0, 1e-4);
        assert_close(values[1], 0.0, 1e-3);
        assert_close(values[2], 0.0, 1e-3);
        assert_close(values[3], 53.2408, 1e-3);
        assert_close(values[4], 80.0925, 1e-3);
        assert_close(values[5], 67.2032, 1e-3);
    }

    #[test]
    fn scales_uint8_rgb_before_lab_conversion() {
        let rgb =
            Tensor::new_with_dtype(vec![255.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::U8).unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        let values = values(&out);
        assert_close(values[0], 53.2408, 1e-3);
        assert_close(values[1], 80.0925, 1e-3);
        assert_close(values[2], 67.2032, 1e-3);
    }

    #[test]
    fn rgb2lab_reads_typed_integer_rgb_storage_exactly() {
        let rgb = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();

        let out = call(rgb).unwrap();

        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        let values = values(&out);
        assert_close(values[0], 53.2408, 1e-3);
        assert_close(values[1], 80.0925, 1e-3);
        assert_close(values[2], 67.2032, 1e-3);
    }

    #[test]
    fn rgb2lab_accepts_uint16_and_rejects_other_integer_classes() {
        let white =
            Tensor::new_integer(IntegerStorage::U16(vec![65535; 3]), vec![1, 1, 3]).unwrap();
        let out = call(white).expect("uint16 RGB");
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        let values = values(&out);
        assert_close(values[0], 100.0, 1e-4);
        assert_close(values[1], 0.0, 1e-3);
        assert_close(values[2], 0.0, 1e-3);

        for storage in [
            IntegerStorage::I8(vec![0; 3]),
            IntegerStorage::I16(vec![0; 3]),
            IntegerStorage::I32(vec![0; 3]),
            IntegerStorage::I64(vec![0; 3]),
            IntegerStorage::U32(vec![0; 3]),
            IntegerStorage::U64(vec![0; 3]),
        ] {
            let input = Tensor::new_integer(storage, vec![1, 1, 3]).unwrap();
            let err = block_on(rgb2lab_builtin(Value::Tensor(input), Vec::new()))
                .expect_err("unsupported integer class");
            assert_eq!(err.identifier(), RGB2LAB_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn rgb2lab_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = RGB2LAB_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["LAB = rgb2lab(RGB, Name=Value)"]);
    }

    #[test]
    fn rgb2lab_integer_capabilities_record_supported_and_rejected_classes() {
        assert_eq!(RGB2LAB_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            RGB2LAB_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            RGB2LAB_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
        assert_eq!(
            RGB2LAB_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::Double
        );
    }

    #[test]
    fn rgb2lab_explicit_gpu_form_is_extension_gated_before_download() {
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1, 3],
            device_id: u32::MAX - 13,
            buffer_id: 1,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(rgb2lab_builtin(
            Value::GpuTensor(handle.clone()),
            Vec::new(),
        ))
        .expect_err("explicit GPU form must be gated");
        runmat_accelerate_api::clear_handle_metadata(&handle);
        assert_eq!(
            err.identifier(),
            RGB2LAB_EXPLICIT_GPU_EXTENSION.error_identifier
        );
    }

    #[test]
    fn rgb2lab_enabled_explicit_gpu_form_restores_output_to_exact_owner() {
        test_support::with_test_provider(|provider| {
            let rgb = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3])
                .expect("RGB");
            let source = gpu_helpers::upload_tensor(provider, &rgb).expect("upload RGB");
            let source =
                source.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            let value = block_on(rgb2lab_builtin(
                Value::GpuTensor(source.clone()),
                Vec::new(),
            ))
            .expect("explicit resident rgb2lab");
            let Value::GpuTensor(output) = value else {
                panic!("explicit gpuArray output must remain resident");
            };
            assert_eq!(output.device_id, source.device_id);
            assert_ne!(output.buffer_id, source.buffer_id);
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            provider.free(&output).expect("free output");
            provider.free(&source).expect("free source");
            runmat_accelerate_api::clear_handle_metadata(&output);
            runmat_accelerate_api::clear_handle_metadata(&source);
        });
    }

    #[test]
    fn rgb2lab_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = RGB2LAB_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.RGB2LAB.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.RGB2LAB.INVALID_INPUT"));
        assert!(codes.contains(&"RM.RGB2LAB.INTERNAL"));
    }

    #[test]
    fn rgb2lab_malformed_name_value_uses_stable_identifier() {
        let err = block_on(rgb2lab_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), RGB2LAB_ERROR_INVALID_OPTION.identifier);
    }
}
