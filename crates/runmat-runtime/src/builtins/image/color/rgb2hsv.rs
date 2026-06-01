//! MATLAB-compatible `rgb2hsv` conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rgb2hsv";

const RGB2HSV_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "HSV",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HSV image or colormap converted from RGB input.",
}];

const RGB2HSV_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB image or Nx3 RGB colormap values.",
}];

const RGB2HSV_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "HSV = rgb2hsv(RGB)",
    inputs: &RGB2HSV_INPUTS,
    outputs: &RGB2HSV_OUTPUT,
}];

const RGB2HSV_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.TOO_MANY_INPUTS",
    identifier: Some("RunMat:rgb2hsv:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "rgb2hsv: too many input arguments",
};

const RGB2HSV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.INVALID_INPUT",
    identifier: Some("RunMat:rgb2hsv:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 RGB image or Nx3 RGB colormap.",
    message: "rgb2hsv: invalid input",
};

const RGB2HSV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2HSV.INTERNAL",
    identifier: Some("RunMat:rgb2hsv:Internal"),
    when: "HSV output tensor construction fails internally.",
    message: "rgb2hsv: internal conversion failure",
};

const RGB2HSV_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RGB2HSV_ERROR_TOO_MANY_INPUTS,
    RGB2HSV_ERROR_INVALID_INPUT,
    RGB2HSV_ERROR_INTERNAL,
];

pub const RGB2HSV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RGB2HSV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RGB2HSV_ERRORS,
};

fn rgb2hsv_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rgb2hsv_error_with_message(error.message, error)
}

fn rgb2hsv_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rgb2hsv_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        rgb2hsv_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::rgb2hsv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rgb2hsv"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation; float RGB/HSV GPU providers are tracked separately.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::rgb2hsv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; conversion operates across RGB channels.",
};

#[runtime_builtin(
    name = "rgb2hsv",
    category = "image/color",
    summary = "Convert RGB image or colormap values to HSV color space.",
    keywords = "rgb2hsv,rgb,hsv,color,image,colormap",
    accel = "sink",
    type_resolver(same_shape_type),
    descriptor(crate::builtins::image::color::rgb2hsv::RGB2HSV_DESCRIPTOR),
    builtin_path = "crate::builtins::image::color::rgb2hsv"
)]
async fn rgb2hsv_builtin(rgb: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(rgb2hsv_error(&RGB2HSV_ERROR_TOO_MANY_INPUTS));
    }
    let tensor = common::gather_tensor(NAME, rgb)
        .await
        .map_err(|err| rgb2hsv_map_error(err, &RGB2HSV_ERROR_INVALID_INPUT))?;
    let layout = common::color_layout(&tensor, NAME)
        .map_err(|err| rgb2hsv_map_error(err, &RGB2HSV_ERROR_INVALID_INPUT))?;
    let dtype = common::image_output_dtype(tensor.dtype);
    let mut data = vec![0.0; tensor.data.len()];
    for pixel in 0..layout.pixels() {
        let r = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 0)],
            tensor.dtype,
        ));
        let g = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 1)],
            tensor.dtype,
        ));
        let b = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 2)],
            tensor.dtype,
        ));
        let (h, s, v) = rgb_to_hsv_unit(r, g, b);
        data[layout.index(pixel, 0)] = cast_float(h, dtype);
        data[layout.index(pixel, 1)] = cast_float(s, dtype);
        data[layout.index(pixel, 2)] = cast_float(v, dtype);
    }
    let out = common::tensor_with_dtype(data, layout.output_shape(), dtype, NAME)
        .map_err(|err| rgb2hsv_map_error(err, &RGB2HSV_ERROR_INTERNAL))?;
    Ok(common::image_value_from_tensor(out))
}

pub(crate) fn rgb_to_hsv_unit(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };
    let s = if max == 0.0 { 0.0 } else { delta / max };
    (h, s, max)
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
    use runmat_builtins::Tensor;

    fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
        let Value::Tensor(out) =
            block_on(rgb2hsv_builtin(Value::Tensor(tensor), Vec::new())).expect("rgb2hsv")
        else {
            panic!("expected tensor");
        };
        Ok(out)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn converts_red_to_hsv() {
        let rgb = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3]);
        assert_close(out.data[0], 0.0);
        assert_close(out.data[1], 1.0);
        assert_close(out.data[2], 1.0);
    }

    #[test]
    fn converts_colormap_secondary_and_gray_values() {
        let rgb = Tensor::new(
            vec![0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.5],
            vec![4, 3],
        )
        .unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.shape, vec![4, 3]);
        let expected = vec![
            0.5,
            5.0 / 6.0,
            1.0 / 6.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.5,
        ];
        for (actual, expected) in out.data.iter().zip(expected) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn scales_uint8_rgb_before_conversion() {
        let rgb = Tensor::new_with_dtype(vec![128.0, 64.0, 32.0], vec![1, 1, 3], NumericDType::U8)
            .unwrap();
        let out = call(rgb).unwrap();
        assert_eq!(out.dtype, NumericDType::F64);
        assert_close(out.data[0], 1.0 / 18.0);
        assert_close(out.data[1], 0.75);
        assert_close(out.data[2], 128.0 / 255.0);
    }

    #[test]
    fn rejects_grayscale_shape() {
        let gray = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = block_on(rgb2hsv_builtin(Value::Tensor(gray), Vec::new())).unwrap_err();
        assert!(err
            .message()
            .contains("expected an MxNx3 RGB image or an Nx3 colormap"));
    }

    #[test]
    fn rgb2hsv_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = RGB2HSV_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["HSV = rgb2hsv(RGB)"]);
    }

    #[test]
    fn rgb2hsv_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = RGB2HSV_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.RGB2HSV.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.RGB2HSV.INVALID_INPUT"));
        assert!(codes.contains(&"RM.RGB2HSV.INTERNAL"));
    }

    #[test]
    fn rgb2hsv_too_many_args_uses_stable_identifier() {
        let err = block_on(rgb2hsv_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), RGB2HSV_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
