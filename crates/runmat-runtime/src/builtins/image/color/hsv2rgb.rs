//! MATLAB-compatible `hsv2rgb` conversion.

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

const NAME: &str = "hsv2rgb";

const HSV2RGB_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB image or colormap converted from HSV input.",
}];

const HSV2RGB_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "HSV",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HSV image or Nx3 HSV colormap values.",
}];

const HSV2RGB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "RGB = hsv2rgb(HSV)",
    inputs: &HSV2RGB_INPUTS,
    outputs: &HSV2RGB_OUTPUT,
}];

const HSV2RGB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HSV2RGB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:hsv2rgb:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "hsv2rgb: too many input arguments",
};

const HSV2RGB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HSV2RGB.INVALID_INPUT",
    identifier: Some("RunMat:hsv2rgb:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 HSV image or Nx3 HSV colormap.",
    message: "hsv2rgb: invalid input",
};

const HSV2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HSV2RGB.INTERNAL",
    identifier: Some("RunMat:hsv2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "hsv2rgb: internal conversion failure",
};

const HSV2RGB_ERRORS: [BuiltinErrorDescriptor; 3] = [
    HSV2RGB_ERROR_TOO_MANY_INPUTS,
    HSV2RGB_ERROR_INVALID_INPUT,
    HSV2RGB_ERROR_INTERNAL,
];

pub const HSV2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HSV2RGB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HSV2RGB_ERRORS,
};

fn hsv2rgb_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    hsv2rgb_error_with_message(error.message, error)
}

fn hsv2rgb_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hsv2rgb_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        hsv2rgb_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::hsv2rgb")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("hsv2rgb"),
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::hsv2rgb")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; conversion operates across HSV channels.",
};

#[runtime_builtin(
    name = "hsv2rgb",
    category = "image/color",
    summary = "Convert HSV values to RGB.",
    keywords = "hsv2rgb,hsv,rgb,color,image,colormap",
    accel = "sink",
    type_resolver(same_shape_type),
    descriptor(crate::builtins::image::color::hsv2rgb::HSV2RGB_DESCRIPTOR),
    builtin_path = "crate::builtins::image::color::hsv2rgb"
)]
async fn hsv2rgb_builtin(hsv: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(hsv2rgb_error(&HSV2RGB_ERROR_TOO_MANY_INPUTS));
    }
    let tensor = common::gather_tensor(NAME, hsv)
        .await
        .map_err(|err| hsv2rgb_map_error(err, &HSV2RGB_ERROR_INVALID_INPUT))?;
    let layout = common::color_layout(&tensor, NAME)
        .map_err(|err| hsv2rgb_map_error(err, &HSV2RGB_ERROR_INVALID_INPUT))?;
    let dtype = common::image_output_dtype(tensor.dtype);
    let mut data = vec![0.0; tensor.data.len()];
    for pixel in 0..layout.pixels() {
        let h =
            common::unit_value(tensor.data[layout.index(pixel, 0)], tensor.dtype).rem_euclid(1.0);
        let s = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 1)],
            tensor.dtype,
        ));
        let v = common::clamp01(common::unit_value(
            tensor.data[layout.index(pixel, 2)],
            tensor.dtype,
        ));
        let (r, g, b) = hsv_to_rgb_unit(h, s, v);
        data[layout.index(pixel, 0)] = cast_float(r, dtype);
        data[layout.index(pixel, 1)] = cast_float(g, dtype);
        data[layout.index(pixel, 2)] = cast_float(b, dtype);
    }
    let out = common::tensor_with_dtype(data, layout.output_shape(), dtype, NAME)
        .map_err(|err| hsv2rgb_map_error(err, &HSV2RGB_ERROR_INTERNAL))?;
    Ok(common::image_value_from_tensor(out))
}

pub(crate) fn hsv_to_rgb_unit(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    if s == 0.0 {
        return (v, v, v);
    }
    let h6 = h * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
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
    use runmat_builtins::Tensor;

    fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
        let Value::Tensor(out) =
            block_on(hsv2rgb_builtin(Value::Tensor(tensor), Vec::new())).expect("hsv2rgb")
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
    fn converts_red_hsv_to_rgb() {
        let hsv = Tensor::new(vec![0.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let out = call(hsv).unwrap();
        assert_close(out.data[0], 1.0);
        assert_close(out.data[1], 0.0);
        assert_close(out.data[2], 0.0);
    }

    #[test]
    fn converts_colormap_with_wrapped_hue_and_gray() {
        let hsv = Tensor::new(
            vec![1.0 / 3.0, -1.0 / 6.0, 0.2, 1.0, 1.0, 0.0, 1.0, 1.0, 0.25],
            vec![3, 3],
        )
        .unwrap();
        let out = call(hsv).unwrap();
        assert_eq!(out.shape, vec![3, 3]);
        let expected = vec![0.0, 1.0, 0.25, 1.0, 0.0, 0.25, 0.0, 1.0, 0.25];
        for (actual, expected) in out.data.iter().zip(expected) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn clamps_saturation_and_value() {
        let hsv = Tensor::new(vec![0.0, 2.0, 1.5], vec![1, 1, 3]).unwrap();
        let out = call(hsv).unwrap();
        assert_close(out.data[0], 1.0);
        assert_close(out.data[1], 0.0);
        assert_close(out.data[2], 0.0);
    }

    #[test]
    fn hsv2rgb_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = HSV2RGB_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["RGB = hsv2rgb(HSV)"]);
    }

    #[test]
    fn hsv2rgb_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = HSV2RGB_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.HSV2RGB.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.HSV2RGB.INVALID_INPUT"));
        assert!(codes.contains(&"RM.HSV2RGB.INTERNAL"));
    }

    #[test]
    fn hsv2rgb_too_many_args_uses_stable_identifier() {
        let err = block_on(hsv2rgb_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), HSV2RGB_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
