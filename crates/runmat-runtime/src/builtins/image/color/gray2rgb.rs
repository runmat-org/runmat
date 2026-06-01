//! `gray2rgb` compatibility helper for replicating grayscale images into RGB.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::gray2rgb_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "gray2rgb";

const GRAY2RGB_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB image with replicated grayscale channels.",
}];

const GRAY2RGB_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grayscale input image.",
}];

const GRAY2RGB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "RGB = gray2rgb(I)",
    inputs: &GRAY2RGB_INPUTS,
    outputs: &GRAY2RGB_OUTPUT,
}];

const GRAY2RGB_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAY2RGB.TOO_MANY_INPUTS",
    identifier: Some("RunMat:gray2rgb:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "gray2rgb: too many input arguments",
};

const GRAY2RGB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAY2RGB.INVALID_INPUT",
    identifier: Some("RunMat:gray2rgb:InvalidInput"),
    when: "Input cannot be interpreted as an MxN grayscale image.",
    message: "gray2rgb: invalid input",
};

const GRAY2RGB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRAY2RGB.INTERNAL",
    identifier: Some("RunMat:gray2rgb:Internal"),
    when: "RGB output tensor construction fails internally.",
    message: "gray2rgb: internal conversion failure",
};

const GRAY2RGB_ERRORS: [BuiltinErrorDescriptor; 3] = [
    GRAY2RGB_ERROR_TOO_MANY_INPUTS,
    GRAY2RGB_ERROR_INVALID_INPUT,
    GRAY2RGB_ERROR_INTERNAL,
];

pub const GRAY2RGB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRAY2RGB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRAY2RGB_ERRORS,
};

fn gray2rgb_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    gray2rgb_error_with_message(error.message, error)
}

fn gray2rgb_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn gray2rgb_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        gray2rgb_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::gray2rgb")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("gray2rgb"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation replicates grayscale planes while preserving logical image dtype metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::gray2rgb")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; output rank changes from MxN to MxNx3.",
};

#[runtime_builtin(
    name = "gray2rgb",
    category = "image/color",
    summary = "Convert grayscale images to RGB.",
    keywords = "gray2rgb,gray,grayscale,rgb,image",
    accel = "sink",
    type_resolver(gray2rgb_type),
    descriptor(crate::builtins::image::color::gray2rgb::GRAY2RGB_DESCRIPTOR),
    builtin_path = "crate::builtins::image::color::gray2rgb"
)]
async fn gray2rgb_builtin(gray: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(gray2rgb_error(&GRAY2RGB_ERROR_TOO_MANY_INPUTS));
    }
    let tensor = common::gather_tensor(NAME, gray)
        .await
        .map_err(|err| gray2rgb_map_error(err, &GRAY2RGB_ERROR_INVALID_INPUT))?;
    let (rows, cols) = common::grayscale_shape(&tensor, NAME)
        .map_err(|err| gray2rgb_map_error(err, &GRAY2RGB_ERROR_INVALID_INPUT))?;
    let pixels = rows * cols;
    let mut data = vec![0.0; pixels * 3];
    for channel in 0..3 {
        data[channel * pixels..(channel + 1) * pixels].copy_from_slice(&tensor.data);
    }
    let out = common::tensor_with_dtype(data, vec![rows, cols, 3], tensor.dtype, NAME)
        .map_err(|err| gray2rgb_map_error(err, &GRAY2RGB_ERROR_INTERNAL))?;
    Ok(common::image_value_from_tensor(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{NumericDType, Tensor};

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(gray2rgb_builtin(value, Vec::new()))
    }

    #[test]
    fn replicates_grayscale_planes() {
        let gray =
            Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::U8).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(gray)).expect("gray2rgb") else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![2, 2, 3]);
        assert_eq!(out.dtype, NumericDType::U8);
        assert_eq!(
            out.data,
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn rejects_truecolor_input() {
        let rgb = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
        let err = call(Value::Tensor(rgb)).unwrap_err();
        assert!(err.message().contains("expected an MxN grayscale image"));
    }

    #[test]
    fn gray2rgb_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = GRAY2RGB_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["RGB = gray2rgb(I)"]);
    }

    #[test]
    fn gray2rgb_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = GRAY2RGB_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.GRAY2RGB.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.GRAY2RGB.INVALID_INPUT"));
        assert!(codes.contains(&"RM.GRAY2RGB.INTERNAL"));
    }

    #[test]
    fn gray2rgb_too_many_args_uses_stable_identifier() {
        let err = block_on(gray2rgb_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), GRAY2RGB_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
