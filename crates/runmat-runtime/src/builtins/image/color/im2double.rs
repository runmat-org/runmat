//! MATLAB-compatible `im2double` image class conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, NumericStorage, Tensor, Value,
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

const NAME: &str = "im2double";

const IM2DOUBLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "J",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Converted image data in double precision.",
}];

const IM2DOUBLE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Image data to convert.",
}];

const IM2DOUBLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "J = im2double(I)",
    inputs: &IM2DOUBLE_INPUTS,
    outputs: &IM2DOUBLE_OUTPUT,
}];

const IM2DOUBLE_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2DOUBLE.TOO_MANY_INPUTS",
    identifier: Some("RunMat:im2double:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "im2double: too many input arguments",
};

const IM2DOUBLE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2DOUBLE.INVALID_INPUT",
    identifier: Some("RunMat:im2double:InvalidInput"),
    when: "Input cannot be gathered or interpreted for image conversion.",
    message: "im2double: invalid input",
};

const IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2DOUBLE.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:im2double:UnsupportedInputType"),
    when: "Input type is outside supported numeric/logical image classes.",
    message: "im2double: unsupported input type",
};

const IM2DOUBLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2DOUBLE.INTERNAL",
    identifier: Some("RunMat:im2double:Internal"),
    when: "Internal conversion step fails while building output tensor.",
    message: "im2double: internal conversion failure",
};

const IM2DOUBLE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    IM2DOUBLE_ERROR_TOO_MANY_INPUTS,
    IM2DOUBLE_ERROR_INVALID_INPUT,
    IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE,
    IM2DOUBLE_ERROR_INTERNAL,
];

pub const IM2DOUBLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IM2DOUBLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IM2DOUBLE_ERRORS,
};

fn im2double_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    im2double_error_with_message(error.message, error)
}

fn im2double_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn im2double_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    im2double_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn im2double_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        im2double_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::im2double")]
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
    notes:
        "Image class scaling runs on the host today so uint8/uint16/uint32 image semantics are preserved.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::im2double")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; integer image dtype metadata is host-side.",
};

#[runtime_builtin(
    name = "im2double",
    category = "image/color",
    summary = "Convert images to double precision.",
    keywords = "im2double,image,convert,double,uint8,uint16",
    accel = "sink",
    type_resolver(same_shape_type),
    descriptor(crate::builtins::image::color::im2double::IM2DOUBLE_DESCRIPTOR),
    builtin_path = "crate::builtins::image::color::im2double"
)]
async fn im2double_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(im2double_error(&IM2DOUBLE_ERROR_TOO_MANY_INPUTS));
    }
    let value = common::gather_value(NAME, &value)
        .await
        .map_err(|err| im2double_map_error(err, &IM2DOUBLE_ERROR_INVALID_INPUT))?;
    match value {
        Value::Tensor(tensor) => Ok(common::image_value_from_tensor(
            im2double_tensor(tensor)
                .map_err(|err| im2double_map_error(err, &IM2DOUBLE_ERROR_INTERNAL))?,
        )),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)
                .map_err(|err| im2double_error_with_detail(&IM2DOUBLE_ERROR_INTERNAL, err))?;
            Ok(common::image_value_from_tensor(tensor))
        }
        Value::Int(IntValue::U8(v)) => Ok(Value::Num(v as f64 / 255.0)),
        Value::Int(IntValue::U16(v)) => Ok(Value::Num(v as f64 / 65535.0)),
        Value::Int(IntValue::I16(v)) => Ok(Value::Num(
            f64::from(i32::from(v) - i32::from(i16::MIN)) / f64::from(u16::MAX),
        )),
        Value::Int(v) => Err(im2double_error_with_detail(
            &IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!("class {}", v.class_name()),
        )),
        Value::Num(v) => Ok(Value::Num(v)),
        Value::Bool(v) => Ok(Value::Num(if v { 1.0 } else { 0.0 })),
        other => Err(im2double_error_with_detail(
            &IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!("type {}", class_name_for_value(&other)),
        )),
    }
}

fn im2double_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| im2double_error_with_detail(&IM2DOUBLE_ERROR_INTERNAL, err))?;
    let data = match storage {
        NumericStorage::F64(values) => values,
        NumericStorage::F32(values) => values.into_iter().map(f64::from).collect(),
        NumericStorage::I16(values) => values
            .into_iter()
            .map(|value| f64::from(i32::from(value) - i32::from(i16::MIN)) / f64::from(u16::MAX))
            .collect(),
        NumericStorage::U8(values) => values
            .into_iter()
            .map(|value| f64::from(value) / f64::from(u8::MAX))
            .collect(),
        NumericStorage::U16(values) => values
            .into_iter()
            .map(|value| f64::from(value) / f64::from(u16::MAX))
            .collect(),
        unsupported => {
            return Err(im2double_error_with_detail(
                &IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE,
                format!("unsupported image class {}", unsupported.class_name()),
            ));
        }
    };
    Tensor::from_numeric_storage(NumericStorage::F64(data), shape)
        .map_err(|err| im2double_error_with_detail(&IM2DOUBLE_ERROR_INTERNAL, err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, LogicalArray, NumericDType};

    fn call(value: Value) -> Value {
        block_on(im2double_builtin(value, Vec::new())).expect("im2double")
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn scales_uint8_tensor_to_unit_double() {
        let input =
            Tensor::new_with_dtype(vec![0.0, 128.0, 255.0], vec![1, 3], NumericDType::U8).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        let values = values(&out);
        assert_eq!(values[0], 0.0);
        assert!((values[1] - 128.0 / 255.0).abs() < 1e-12);
        assert_eq!(values[2], 1.0);
    }

    #[test]
    fn im2double_reads_typed_integer_tensor_storage_exactly() {
        let input = Tensor::new_integer(IntegerStorage::U8(vec![0, 128, 255]), vec![1, 3]).unwrap();

        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };

        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        let values = values(&out);
        assert_eq!(values[0], 0.0);
        assert!((values[1] - 128.0 / 255.0).abs() < 1e-12);
        assert_eq!(values[2], 1.0);
    }

    #[test]
    fn scales_uint16_scalar() {
        assert_eq!(call(Value::Int(IntValue::U16(65535))), Value::Num(1.0));
    }

    #[test]
    fn scales_int16_image_range() {
        assert_eq!(call(Value::Int(IntValue::I16(i16::MIN))), Value::Num(0.0));
        assert_eq!(call(Value::Int(IntValue::I16(i16::MAX))), Value::Num(1.0));
        let input =
            Tensor::new_integer(IntegerStorage::I16(vec![i16::MIN, 0, i16::MAX]), vec![1, 3])
                .unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        let values = values(&out);
        assert_eq!(values[0], 0.0);
        assert!((values[1] - 32768.0 / 65535.0).abs() < 1e-12);
        assert_eq!(values[2], 1.0);
    }

    #[test]
    fn preserves_float_values_and_shape() {
        let input =
            Tensor::new_with_dtype(vec![-0.25, 0.0, 0.5, 1.25], vec![2, 2], NumericDType::F32)
                .unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(values(&out), vec![-0.25, 0.0, 0.5, 1.25]);
    }

    #[test]
    fn converts_logical_array_to_double_zeros_and_ones() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let Value::Tensor(out) = call(Value::LogicalArray(logical)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(values(&out), vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn rejects_unsupported_integer_classes() {
        for (scalar, storage) in [
            (IntValue::I8(0), IntegerStorage::I8(vec![0])),
            (IntValue::I32(0), IntegerStorage::I32(vec![0])),
            (IntValue::I64(0), IntegerStorage::I64(vec![0])),
            (IntValue::U32(0), IntegerStorage::U32(vec![0])),
            (IntValue::U64(0), IntegerStorage::U64(vec![0])),
        ] {
            let scalar_err =
                block_on(im2double_builtin(Value::Int(scalar), Vec::new())).unwrap_err();
            assert_eq!(
                scalar_err.identifier(),
                IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
            );
            let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let tensor_err =
                block_on(im2double_builtin(Value::Tensor(tensor), Vec::new())).unwrap_err();
            assert_eq!(
                tensor_err.identifier(),
                IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
            );
        }
    }

    #[test]
    fn im2double_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = IM2DOUBLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["J = im2double(I)"]);
    }

    #[test]
    fn im2double_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = IM2DOUBLE_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.IM2DOUBLE.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.IM2DOUBLE.INVALID_INPUT"));
        assert!(codes.contains(&"RM.IM2DOUBLE.UNSUPPORTED_INPUT_TYPE"));
        assert!(codes.contains(&"RM.IM2DOUBLE.INTERNAL"));
    }

    #[test]
    fn im2double_too_many_args_uses_stable_identifier() {
        let err = block_on(im2double_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("expected argument error");
        assert_eq!(err.identifier(), IM2DOUBLE_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
