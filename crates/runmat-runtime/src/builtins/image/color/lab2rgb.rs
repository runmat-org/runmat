//! MATLAB-compatible default-D65 `lab2rgb` conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lab2rgb";
const XN: f64 = 0.95047;
const YN: f64 = 1.0;
const ZN: f64 = 1.08883;
const EPSILON: f64 = 216.0 / 24389.0;
const KAPPA: f64 = 24389.0 / 27.0;

const LAB2RGB_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "sRGB image or colormap converted from CIE L*a*b* input.",
}];

const LAB2RGB_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "LAB",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "CIE L*a*b* image or Nx3 colormap values.",
}];

const LAB2RGB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "RGB = lab2rgb(LAB)",
    inputs: &LAB2RGB_INPUTS,
    outputs: &LAB2RGB_OUTPUT,
}];

const LAB2RGB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:lab2rgb:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "lab2rgb: too many input arguments",
};

const LAB2RGB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.INVALID_INPUT",
    identifier: Some("RunMat:lab2rgb:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 L*a*b* image or Nx3 colormap.",
    message: "lab2rgb: invalid input",
};

const LAB2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LAB2RGB.INTERNAL",
    identifier: Some("RunMat:lab2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "lab2rgb: internal conversion failure",
};

const LAB2RGB_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LAB2RGB_ERROR_TOO_MANY_INPUTS,
    LAB2RGB_ERROR_INVALID_INPUT,
    LAB2RGB_ERROR_INTERNAL,
];

pub const LAB2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LAB2RGB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LAB2RGB_ERRORS,
};

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
        lab2rgb_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::lab2rgb")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("lab2rgb"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation uses the default D65/sRGB conversion path and clips out-of-gamut RGB values.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::lab2rgb")]
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
    name = "lab2rgb",
    category = "image/color",
    summary = "Convert CIE L*a*b* values to sRGB (D65 white point).",
    keywords = "lab2rgb,lab,cielab,rgb,color,image,colormap",
    accel = "sink",
    type_resolver(same_shape_type),
    descriptor(crate::builtins::image::color::lab2rgb::LAB2RGB_DESCRIPTOR),
    builtin_path = "crate::builtins::image::color::lab2rgb"
)]
async fn lab2rgb_builtin(lab: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(lab2rgb_error(&LAB2RGB_ERROR_TOO_MANY_INPUTS));
    }
    let logical_input = match &lab {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    };
    let tensor = common::gather_tensor(NAME, lab)
        .await
        .map_err(|err| lab2rgb_map_error(err, &LAB2RGB_ERROR_INVALID_INPUT))?;
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
    let out = common::tensor_with_dtype(data, layout.output_shape(), dtype, NAME)
        .map_err(|err| lab2rgb_map_error(err, &LAB2RGB_ERROR_INTERNAL))?;
    Ok(common::image_value_from_tensor(out))
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
    (
        common::clamp01(linear_to_srgb(r)),
        common::clamp01(linear_to_srgb(g)),
        common::clamp01(linear_to_srgb(blue)),
    )
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
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, LogicalArray, Tensor};

    fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
        let value = block_on(lab2rgb_builtin(Value::Tensor(tensor), Vec::new()))?;
        let Value::Tensor(out) = value else {
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
    fn converts_white_lab_to_rgb() {
        let lab = Tensor::new(vec![100.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let out = call(lab).unwrap();
        let values = values(&out);
        assert_close(values[0], 1.0, 1e-4);
        assert_close(values[1], 1.0, 1e-4);
        assert_close(values[2], 1.0, 1e-4);
    }

    #[test]
    fn converts_lab_colormap_references_to_rgb() {
        let lab = Tensor::new(
            vec![
                0.0, 53.2408, 87.7347, 32.2970, 0.0, 80.0925, -86.1827, 79.1875, 0.0, 67.2032,
                83.1793, -107.8602,
            ],
            vec![4, 3],
        )
        .unwrap();
        let out = call(lab).unwrap();
        assert_eq!(out.shape, vec![4, 3]);
        let expected = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for (actual, expected) in values(&out).iter().zip(expected) {
            assert_close(*actual, expected, 1e-3);
        }
    }

    #[test]
    fn preserves_single_precision_metadata() {
        let lab = Tensor::new_with_dtype(vec![100.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::F32)
            .unwrap();
        let out = call(lab).unwrap();
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        let values = values(&out);
        assert_close(values[0], 1.0, 1e-4);
        assert_close(values[1], 1.0, 1e-4);
        assert_close(values[2], 1.0, 1e-4);
    }

    #[test]
    fn lab2rgb_rejects_every_integer_class_and_logical() {
        for storage in [
            IntegerStorage::I8(vec![0; 3]),
            IntegerStorage::I16(vec![0; 3]),
            IntegerStorage::I32(vec![0; 3]),
            IntegerStorage::I64(vec![0; 3]),
            IntegerStorage::U8(vec![0; 3]),
            IntegerStorage::U16(vec![0; 3]),
            IntegerStorage::U32(vec![0; 3]),
            IntegerStorage::U64(vec![0; 3]),
        ] {
            let lab = Tensor::new_integer(storage, vec![1, 1, 3]).unwrap();
            let err = call(lab).unwrap_err();
            assert_eq!(err.identifier(), LAB2RGB_ERROR_INVALID_INPUT.identifier);
        }

        let logical = LogicalArray::new(vec![1, 0, 0], vec![1, 1, 3]).unwrap();
        let err = block_on(lab2rgb_builtin(Value::LogicalArray(logical), Vec::new())).unwrap_err();
        assert_eq!(err.identifier(), LAB2RGB_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn lab2rgb_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = LAB2RGB_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["RGB = lab2rgb(LAB)"]);
    }

    #[test]
    fn lab2rgb_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = LAB2RGB_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.LAB2RGB.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.LAB2RGB.INVALID_INPUT"));
        assert!(codes.contains(&"RM.LAB2RGB.INTERNAL"));
    }

    #[test]
    fn lab2rgb_too_many_args_uses_stable_identifier() {
        let err = block_on(lab2rgb_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), LAB2RGB_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
