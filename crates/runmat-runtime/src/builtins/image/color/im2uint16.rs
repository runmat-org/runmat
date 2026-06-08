//! MATLAB-compatible `im2uint16` image class conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::builtins::introspection::class::class_name_for_value;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "im2uint16";

const IM2UINT16_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "J",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Converted image data in uint16 class.",
}];

const IM2UINT16_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Image data to convert.",
}];

const IM2UINT16_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "J = im2uint16(I)",
    inputs: &IM2UINT16_INPUTS,
    outputs: &IM2UINT16_OUTPUT,
}];

const IM2UINT16_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT16.TOO_MANY_INPUTS",
    identifier: Some("RunMat:im2uint16:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "im2uint16: too many input arguments",
};

const IM2UINT16_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT16.INVALID_INPUT",
    identifier: Some("RunMat:im2uint16:InvalidInput"),
    when: "Input cannot be gathered or interpreted for image conversion.",
    message: "im2uint16: invalid input",
};

const IM2UINT16_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT16.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:im2uint16:UnsupportedInputType"),
    when: "Input type is outside supported numeric/logical image classes.",
    message: "im2uint16: unsupported input type",
};

const IM2UINT16_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT16.INTERNAL",
    identifier: Some("RunMat:im2uint16:Internal"),
    when: "Internal conversion step fails while building output tensor.",
    message: "im2uint16: internal conversion failure",
};

const IM2UINT16_ERRORS: [BuiltinErrorDescriptor; 4] = [
    IM2UINT16_ERROR_TOO_MANY_INPUTS,
    IM2UINT16_ERROR_INVALID_INPUT,
    IM2UINT16_ERROR_UNSUPPORTED_INPUT_TYPE,
    IM2UINT16_ERROR_INTERNAL,
];

pub const IM2UINT16_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IM2UINT16_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IM2UINT16_ERRORS,
};

fn im2uint16_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    im2uint16_error_with_message(error.message, error)
}

fn im2uint16_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn im2uint16_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    im2uint16_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn im2uint16_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        im2uint16_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::im2uint16")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host conversion preserves MATLAB image scaling semantics for uint16 outputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::im2uint16")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; uint16 image dtype metadata is host-side.",
};

#[runtime_builtin(
    name = "im2uint16",
    category = "image/color",
    summary = "Convert images to uint16.",
    keywords = "im2uint16,image,convert,uint16,double,uint8",
    accel = "sink",
    type_resolver(same_shape_type),
    descriptor(crate::builtins::image::color::im2uint16::IM2UINT16_DESCRIPTOR),
    builtin_path = "crate::builtins::image::color::im2uint16"
)]
async fn im2uint16_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(im2uint16_error(&IM2UINT16_ERROR_TOO_MANY_INPUTS));
    }
    let value = common::gather_value(NAME, &value)
        .await
        .map_err(|err| im2uint16_map_error(err, &IM2UINT16_ERROR_INVALID_INPUT))?;
    match value {
        Value::Tensor(tensor) => Ok(common::image_value_from_tensor(
            im2uint16_tensor(tensor)
                .map_err(|err| im2uint16_map_error(err, &IM2UINT16_ERROR_INTERNAL))?,
        )),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)
                .map_err(|err| im2uint16_error_with_detail(&IM2UINT16_ERROR_INTERNAL, err))?;
            Ok(common::image_value_from_tensor(
                im2uint16_tensor(tensor)
                    .map_err(|err| im2uint16_map_error(err, &IM2UINT16_ERROR_INTERNAL))?,
            ))
        }
        Value::Int(IntValue::U16(v)) => Ok(Value::Int(IntValue::U16(v))),
        Value::Int(IntValue::U8(v)) => Ok(Value::Int(IntValue::U16((v as u16) * 257))),
        Value::Int(v) => Ok(Value::Int(IntValue::U16(
            common::clamp_round(v.to_f64(), 65535.0) as u16,
        ))),
        Value::Num(v) => Ok(Value::Int(IntValue::U16(common::unit_to_dtype(
            common::clamp01(v),
            NumericDType::U16,
        ) as u16))),
        Value::Bool(v) => Ok(Value::Int(IntValue::U16(if v { 65535 } else { 0 }))),
        other => Err(im2uint16_error_with_detail(
            &IM2UINT16_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!("type {}", class_name_for_value(&other)),
        )),
    }
}

fn im2uint16_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = match tensor.dtype {
        NumericDType::U16 => tensor.data,
        NumericDType::U8 => tensor.data.iter().map(|&value| value * 257.0).collect(),
        NumericDType::F32 | NumericDType::F64 => tensor
            .data
            .iter()
            .map(|&value| common::unit_to_dtype(common::clamp01(value), NumericDType::U16))
            .collect(),
    };
    common::tensor_with_dtype(data, tensor.shape, NumericDType::U16, NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::LogicalArray;

    fn call(value: Value) -> Value {
        block_on(im2uint16_builtin(value, Vec::new())).expect("im2uint16")
    }

    #[test]
    fn scales_double_to_uint16_image_range() {
        assert_eq!(call(Value::Num(1.0)), Value::Int(IntValue::U16(65535)));
    }

    #[test]
    fn expands_uint8_tensor_to_uint16_range() {
        let input = Tensor::new_with_dtype(vec![0.0, 255.0], vec![1, 2], NumericDType::U8).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::U16);
        assert_eq!(out.data, vec![0.0, 65535.0]);
    }

    #[test]
    fn clamps_and_rounds_float_tensor_to_uint16() {
        let input = Tensor::new(vec![-0.1, 0.0, 0.5, 1.0, 1.2, f64::NAN], vec![2, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::U16);
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, vec![0.0, 0.0, 32768.0, 65535.0, 65535.0, 0.0]);
    }

    #[test]
    fn converts_logical_array_to_uint16_extrema() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
        let Value::Tensor(out) = call(Value::LogicalArray(logical)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::U16);
        assert_eq!(out.data, vec![0.0, 65535.0, 65535.0, 0.0]);
    }

    #[test]
    fn im2uint16_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = IM2UINT16_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["J = im2uint16(I)"]);
    }

    #[test]
    fn im2uint16_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = IM2UINT16_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.IM2UINT16.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.IM2UINT16.INVALID_INPUT"));
        assert!(codes.contains(&"RM.IM2UINT16.UNSUPPORTED_INPUT_TYPE"));
        assert!(codes.contains(&"RM.IM2UINT16.INTERNAL"));
    }

    #[test]
    fn im2uint16_too_many_args_uses_stable_identifier() {
        let err = block_on(im2uint16_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), IM2UINT16_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
