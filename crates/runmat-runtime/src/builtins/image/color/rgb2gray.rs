//! MATLAB-compatible `rgb2gray` conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::rgb2gray_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rgb2gray";

const RGB2GRAY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grayscale image converted from RGB input.",
}];

const RGB2GRAY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "RGB",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RGB truecolor image.",
}];

const RGB2GRAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "I = rgb2gray(RGB)",
    inputs: &RGB2GRAY_INPUTS,
    outputs: &RGB2GRAY_OUTPUT,
}];

const RGB2GRAY_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.TOO_MANY_INPUTS",
    identifier: Some("RunMat:rgb2gray:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "rgb2gray: too many input arguments",
};

const RGB2GRAY_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.INVALID_INPUT",
    identifier: Some("RunMat:rgb2gray:InvalidInput"),
    when: "Input cannot be interpreted as an MxNx3 RGB image.",
    message: "rgb2gray: invalid input",
};

const RGB2GRAY_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RGB2GRAY.INTERNAL",
    identifier: Some("RunMat:rgb2gray:Internal"),
    when: "Grayscale output tensor construction fails internally.",
    message: "rgb2gray: internal conversion failure",
};

const RGB2GRAY_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RGB2GRAY_ERROR_TOO_MANY_INPUTS,
    RGB2GRAY_ERROR_INVALID_INPUT,
    RGB2GRAY_ERROR_INTERNAL,
];

pub const RGB2GRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RGB2GRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RGB2GRAY_ERRORS,
};

fn rgb2gray_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rgb2gray_error_with_message(error.message, error)
}

fn rgb2gray_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rgb2gray_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        rgb2gray_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::rgb2gray")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rgb2gray"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host implementation preserves integer image dtype semantics; a float GPU provider can be added independently.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::rgb2gray")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; channel-aware image shape handling is custom.",
};

#[runtime_builtin(
    name = "rgb2gray",
    category = "image/color",
    summary = "Convert an RGB image to grayscale using luminance weights.",
    keywords = "rgb2gray,rgb,gray,grayscale,luminance,image",
    accel = "sink",
    type_resolver(rgb2gray_type),
    descriptor(crate::builtins::image::color::rgb2gray::RGB2GRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::image::color::rgb2gray"
)]
async fn rgb2gray_builtin(rgb: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(rgb2gray_error(&RGB2GRAY_ERROR_TOO_MANY_INPUTS));
    }
    let tensor = common::gather_tensor(NAME, rgb)
        .await
        .map_err(|err| rgb2gray_map_error(err, &RGB2GRAY_ERROR_INVALID_INPUT))?;
    let out = rgb2gray_tensor(&tensor)
        .map_err(|err| rgb2gray_map_error(err, &RGB2GRAY_ERROR_INTERNAL))?;
    Ok(common::image_value_from_tensor(out))
}

fn rgb2gray_tensor(rgb: &Tensor) -> BuiltinResult<Tensor> {
    let common::ColorLayout::Truecolor { rows, cols } = common::truecolor_layout(rgb, NAME)? else {
        unreachable!();
    };
    let pixels = rows * cols;
    let mut data = vec![0.0; pixels];
    for (pixel, out) in data.iter_mut().enumerate() {
        let r = common::unit_value(rgb.data[pixel], rgb.dtype);
        let g = common::unit_value(rgb.data[pixel + pixels], rgb.dtype);
        let b = common::unit_value(rgb.data[pixel + 2 * pixels], rgb.dtype);
        let gray = 0.2989 * r + 0.5870 * g + 0.1140 * b;
        *out = common::unit_to_dtype(gray, rgb.dtype);
    }
    let dtype = match rgb.dtype {
        NumericDType::U8 | NumericDType::U16 | NumericDType::F32 => rgb.dtype,
        NumericDType::F64 => NumericDType::F64,
    };
    common::tensor_with_dtype(data, vec![rows, cols], dtype, NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn call(value: Value) -> Value {
        block_on(rgb2gray_builtin(value, Vec::new())).expect("rgb2gray")
    }

    #[test]
    fn converts_uint8_rgb_to_grayscale_uint8() {
        let rgb =
            Tensor::new_with_dtype(vec![255.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::U8).unwrap();
        let Value::Int(value) = call(Value::Tensor(rgb)) else {
            panic!("expected scalar int");
        };
        assert_eq!(value.to_i64(), 76);
    }

    #[test]
    fn preserves_2d_shape() {
        let rgb = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(rgb)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![2, 2]);
        assert!(out.data.iter().all(|v| (*v - 0.9999).abs() < 1e-12));
    }

    #[test]
    fn converts_column_major_uint16_planes() {
        let rgb = Tensor::new_with_dtype(
            vec![65535.0, 0.0, 0.0, 65535.0, 0.0, 0.0],
            vec![2, 1, 3],
            NumericDType::U16,
        )
        .unwrap();
        let Value::Tensor(out) = call(Value::Tensor(rgb)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::U16);
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.data, vec![19588.0, 38469.0]);
    }

    #[test]
    fn rejects_colormap_shape() {
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let err = block_on(rgb2gray_builtin(Value::Tensor(map), Vec::new())).unwrap_err();
        assert!(err.message().contains("expected an MxNx3 RGB image"));
    }

    #[test]
    fn rgb2gray_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = RGB2GRAY_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["I = rgb2gray(RGB)"]);
    }

    #[test]
    fn rgb2gray_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = RGB2GRAY_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.RGB2GRAY.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.RGB2GRAY.INVALID_INPUT"));
        assert!(codes.contains(&"RM.RGB2GRAY.INTERNAL"));
    }

    #[test]
    fn rgb2gray_too_many_args_uses_stable_identifier() {
        let err = block_on(rgb2gray_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), RGB2GRAY_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[test]
    fn rgb2gray_is_registered_with_dispatcher() {
        let rgb = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let result = block_on(crate::call_builtin_async(NAME, &[Value::Tensor(rgb)]))
            .expect("rgb2gray registered");
        assert!(matches!(result, Value::Num(value) if (value - 0.9999).abs() < 1e-12));
    }
}
