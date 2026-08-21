//! MATLAB-compatible `im2uint8` image class conversion.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerClass, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::image::color::common;
use crate::builtins::image::color::type_resolvers::same_shape_type;
use crate::builtins::introspection::class::class_name_for_value;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "im2uint8";

const IM2UINT8_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "J",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Converted image data in uint8 class.",
}];

const IM2UINT8_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Image data to convert.",
}];

const IM2UINT8_INDEXED_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image data to convert.",
    },
    BuiltinParamDescriptor {
        name: "indexed",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "The literal \"indexed\", selecting indexed-image offset semantics.",
    },
];

const IM2UINT8_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "J = im2uint8(I)",
        inputs: &IM2UINT8_INPUTS,
        outputs: &IM2UINT8_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "J = im2uint8(I, \"indexed\")",
        inputs: &IM2UINT8_INDEXED_INPUTS,
        outputs: &IM2UINT8_OUTPUT,
    },
];

const IM2UINT8_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT8.TOO_MANY_INPUTS",
    identifier: Some("RunMat:im2uint8:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "im2uint8: too many input arguments",
};

const IM2UINT8_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT8.INVALID_INPUT",
    identifier: Some("RunMat:im2uint8:InvalidInput"),
    when: "Input cannot be gathered or interpreted for image conversion.",
    message: "im2uint8: invalid input",
};

const IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT8.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:im2uint8:UnsupportedInputType"),
    when: "Input type is outside supported numeric/logical image classes.",
    message: "im2uint8: unsupported input type",
};

const IM2UINT8_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT8.INTERNAL",
    identifier: Some("RunMat:im2uint8:Internal"),
    when: "Internal conversion step fails while building output tensor.",
    message: "im2uint8: internal conversion failure",
};

const IM2UINT8_ERRORS: [BuiltinErrorDescriptor; 4] = [
    IM2UINT8_ERROR_TOO_MANY_INPUTS,
    IM2UINT8_ERROR_INVALID_INPUT,
    IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
    IM2UINT8_ERROR_INTERNAL,
];

pub const IM2UINT8_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IM2UINT8_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IM2UINT8_ERRORS,
};

const IM2UINT8_DOCUMENTED_INTEGER_CLASSES: [BuiltinIntegerClass; 3] = [
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
];
const IM2UINT8_REJECTED_INTEGER_CLASSES: [BuiltinIntegerClass; 5] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];
const IM2UINT8_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &IM2UINT8_DOCUMENTED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Intensity and truecolor uint8, uint16, and int16 inputs quantize into the uint8 image range.",
    }];
const IM2UINT8_INDEXED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &[BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16],
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Indexed uint8 values are retained; indexed uint16 values must not exceed 255 and narrow without intensity scaling.",
    }];
const IM2UINT8_REJECTED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &IM2UINT8_REJECTED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "These integer image classes are outside the documented im2uint8 input surface and reject before conversion.",
    }];
pub const IM2UINT8_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "J = im2uint8(integer_I)",
        inputs: &IM2UINT8_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "uint16 and offset int16 inputs quantize at the documented half-byte boundaries; output is exact uint8 storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "J = im2uint8(integer_I, \"indexed\")",
        inputs: &IM2UINT8_INDEXED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Indexed conversion preserves zero-based integer indices and errors when a uint16 index cannot fit in uint8.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "im2uint8(unsupported_integer_I, ...)",
        inputs: &IM2UINT8_REJECTED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer dtype metadata is rejected consistently for host and resident inputs.",
    },
];

fn im2uint8_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    im2uint8_error_with_message(error.message, error)
}

fn im2uint8_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn im2uint8_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    im2uint8_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn im2uint8_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        im2uint8_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::color::im2uint8")]
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
    notes: "Host conversion preserves MATLAB image scaling semantics for uint8 outputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::color::im2uint8")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fused yet; uint8 image dtype metadata is host-side.",
};

#[runtime_builtin(
    name = "im2uint8",
    category = "image/color",
    summary = "Convert images to uint8.",
    keywords = "im2uint8,image,convert,uint8,double,uint16",
    accel = "sink",
    type_resolver(same_shape_type),
    descriptor(crate::builtins::image::color::im2uint8::IM2UINT8_DESCRIPTOR),
    integer_capabilities(crate::builtins::image::color::im2uint8::IM2UINT8_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::image::color::im2uint8"
)]
async fn im2uint8_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(im2uint8_error(&IM2UINT8_ERROR_TOO_MANY_INPUTS));
    }
    let indexed = parse_indexed_mode(&rest)?;
    ensure_resident_integer_class_supported(&value, indexed)?;
    let resident_source = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let value = common::gather_value(NAME, &value)
        .await
        .map_err(|err| im2uint8_map_error(err, &IM2UINT8_ERROR_INVALID_INPUT))?;
    let result = match value {
        Value::Tensor(tensor) => Ok(common::image_value_from_tensor(
            im2uint8_tensor(tensor, indexed)
                .map_err(|err| im2uint8_map_error(err, &IM2UINT8_ERROR_INTERNAL))?,
        )),
        Value::LogicalArray(array) => {
            if indexed {
                let shape = array.shape.clone();
                let tensor = Tensor::from_numeric_storage(NumericStorage::U8(array.data), shape)
                    .map_err(|err| im2uint8_error_with_detail(&IM2UINT8_ERROR_INTERNAL, err))?;
                Ok(common::image_value_from_tensor(tensor))
            } else {
                let tensor = tensor::logical_to_tensor(&array)
                    .map_err(|err| im2uint8_error_with_detail(&IM2UINT8_ERROR_INTERNAL, err))?;
                Ok(common::image_value_from_tensor(
                    im2uint8_tensor(tensor, false)
                        .map_err(|err| im2uint8_map_error(err, &IM2UINT8_ERROR_INTERNAL))?,
                ))
            }
        }
        Value::Int(IntValue::U8(v)) => Ok(Value::Int(IntValue::U8(v))),
        Value::Int(IntValue::U16(v)) if indexed && v <= u8::MAX.into() => {
            Ok(Value::Int(IntValue::U8(v as u8)))
        }
        Value::Int(IntValue::U16(_)) if indexed => Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_INVALID_INPUT,
            "indexed uint16 values must not exceed 255",
        )),
        Value::Int(IntValue::U16(v)) => Ok(Value::Int(IntValue::U8(quantize_u16_to_u8(v)))),
        Value::Int(IntValue::I16(v)) if indexed => Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!("class {} for indexed image", IntValue::I16(v).class_name()),
        )),
        Value::Int(IntValue::I16(v)) => Ok(Value::Int(IntValue::U8(quantize_u16_to_u8(
            (i32::from(v) - i32::from(i16::MIN)) as u16,
        )))),
        Value::Int(v) => Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!("class {}", v.class_name()),
        )),
        Value::Num(v) if indexed && v <= 256.0 => Ok(Value::Int(IntValue::U8(
            common::clamp_round(v - 1.0, 255.0) as u8,
        ))),
        Value::Num(_) if indexed => Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_INVALID_INPUT,
            "indexed double values must not exceed 256",
        )),
        Value::Num(v) => Ok(Value::Int(IntValue::U8(common::unit_to_dtype(
            common::clamp01(v),
            NumericDType::U8,
        ) as u8))),
        Value::Bool(v) => Ok(Value::Int(IntValue::U8(if indexed {
            u8::from(v)
        } else if v {
            255
        } else {
            0
        }))),
        other => Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!("type {}", class_name_for_value(&other)),
        )),
    }?;
    common::restore_resident_numeric_result(resident_source.as_ref(), result, NAME)
}

fn ensure_resident_integer_class_supported(value: &Value, indexed: bool) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        return Ok(());
    };
    let Some(dtype) = runmat_accelerate_api::handle_integer_type(handle) else {
        return Ok(());
    };
    let supported = if indexed {
        matches!(
            dtype,
            runmat_accelerate_api::IntegerElementType::U8
                | runmat_accelerate_api::IntegerElementType::U16
        )
    } else {
        matches!(
            dtype,
            runmat_accelerate_api::IntegerElementType::I16
                | runmat_accelerate_api::IntegerElementType::U8
                | runmat_accelerate_api::IntegerElementType::U16
        )
    };
    if supported {
        Ok(())
    } else {
        Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!(
                "unsupported{} resident integer image class {dtype:?}",
                if indexed { " indexed" } else { "" }
            ),
        ))
    }
}

fn parse_indexed_mode(rest: &[Value]) -> BuiltinResult<bool> {
    let Some(mode) = rest.first() else {
        return Ok(false);
    };
    let text = match mode {
        Value::String(text) => text.clone(),
        Value::CharArray(chars) => chars.data.iter().collect(),
        _ => {
            return Err(im2uint8_error_with_detail(
                &IM2UINT8_ERROR_INVALID_INPUT,
                "second input must be the text \"indexed\"",
            ));
        }
    };
    if text.trim().eq_ignore_ascii_case("indexed") {
        Ok(true)
    } else {
        Err(im2uint8_error_with_detail(
            &IM2UINT8_ERROR_INVALID_INPUT,
            "second input must be the text \"indexed\"",
        ))
    }
}

fn quantize_u16_to_u8(value: u16) -> u8 {
    ((u32::from(value) + 128) >> 8).min(u32::from(u8::MAX)) as u8
}

fn im2uint8_tensor(tensor: Tensor, indexed: bool) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| im2uint8_error_with_detail(&IM2UINT8_ERROR_INTERNAL, err))?;
    let data = match (storage, indexed) {
        (NumericStorage::U8(values), true) => values,
        (NumericStorage::U16(values), true) => {
            if values.iter().any(|value| *value > u16::from(u8::MAX)) {
                return Err(im2uint8_error_with_detail(
                    &IM2UINT8_ERROR_INVALID_INPUT,
                    "indexed uint16 values must not exceed 255",
                ));
            }
            values.into_iter().map(|value| value as u8).collect()
        }
        (NumericStorage::F64(values), true) => {
            if values.iter().any(|value| *value > 256.0) {
                return Err(im2uint8_error_with_detail(
                    &IM2UINT8_ERROR_INVALID_INPUT,
                    "indexed double values must not exceed 256",
                ));
            }
            values
                .into_iter()
                .map(|value| common::clamp_round(value - 1.0, 255.0) as u8)
                .collect()
        }
        (unsupported, true) => {
            return Err(im2uint8_error_with_detail(
                &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
                format!(
                    "unsupported indexed image class {}",
                    unsupported.class_name()
                ),
            ));
        }
        (NumericStorage::U8(values), false) => values,
        (NumericStorage::I16(values), false) => values
            .into_iter()
            .map(|value| quantize_u16_to_u8((i32::from(value) - i32::from(i16::MIN)) as u16))
            .collect(),
        (NumericStorage::U16(values), false) => {
            values.into_iter().map(quantize_u16_to_u8).collect()
        }
        (NumericStorage::F32(values), false) => values
            .into_iter()
            .map(|value| {
                common::unit_to_dtype(common::clamp01(f64::from(value)), NumericDType::U8) as u8
            })
            .collect(),
        (NumericStorage::F64(values), false) => values
            .into_iter()
            .map(|value| common::unit_to_dtype(common::clamp01(value), NumericDType::U8) as u8)
            .collect(),
        (unsupported, false) => {
            return Err(im2uint8_error_with_detail(
                &IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
                format!("unsupported image class {}", unsupported.class_name()),
            ));
        }
    };
    Tensor::from_numeric_storage(NumericStorage::U8(data), shape)
        .map_err(|err| im2uint8_error_with_detail(&IM2UINT8_ERROR_INTERNAL, err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataOwned, HostTensorView, IntegerElementType};
    use runmat_value::{IntegerStorage, LogicalArray};

    fn call(value: Value) -> Value {
        block_on(im2uint8_builtin(value, Vec::new())).expect("im2uint8")
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn scales_double_to_uint8_image_range() {
        assert_eq!(call(Value::Num(0.5)), Value::Int(IntValue::U8(128)));
    }

    #[test]
    fn converts_uint16_tensor_to_uint8_range() {
        let input =
            Tensor::new_with_dtype(vec![0.0, 65535.0], vec![1, 2], NumericDType::U16).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::U8);
        assert_eq!(values(&out), vec![0.0, 255.0]);
    }

    #[test]
    fn im2uint8_reads_typed_integer_tensor_storage_exactly() {
        let input =
            Tensor::new_integer(IntegerStorage::U16(vec![0, u16::MAX]), vec![1, 2]).unwrap();

        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };

        assert_eq!(out.numeric_dtype(), NumericDType::U8);
        assert_eq!(
            out.integer_storage(),
            Some(&IntegerStorage::U8(vec![0, 255]))
        );
    }

    #[test]
    fn uint16_quantization_uses_documented_byte_boundaries() {
        let input = Tensor::new_integer(
            IntegerStorage::U16(vec![0, 127, 128, 255, 256, 383, 384, u16::MAX]),
            vec![1, 8],
        )
        .unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(
            out.integer_storage(),
            Some(&IntegerStorage::U8(vec![0, 0, 1, 1, 1, 1, 2, 255]))
        );
    }

    #[test]
    fn converts_int16_tensor_to_uint8_range_with_exact_output_storage() {
        let input =
            Tensor::new_integer(IntegerStorage::I16(vec![i16::MIN, i16::MAX]), vec![1, 2]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::U8);
        assert_eq!(
            out.integer_storage(),
            Some(&IntegerStorage::U8(vec![0, 255]))
        );
    }

    #[test]
    fn clamps_and_rounds_float_tensor_to_uint8() {
        let input = Tensor::new(vec![-0.1, 0.0, 0.5, 1.0, 1.2, f64::NAN], vec![2, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::U8);
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(values(&out), vec![0.0, 0.0, 128.0, 255.0, 255.0, 0.0]);
    }

    #[test]
    fn converts_logical_array_to_uint8_extrema() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let Value::Tensor(out) = call(Value::LogicalArray(logical)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::U8);
        assert_eq!(values(&out), vec![255.0, 0.0, 0.0, 255.0]);
    }

    #[test]
    fn scales_int16_scalar_and_rejects_unsupported_integer_classes() {
        assert_eq!(
            call(Value::Int(IntValue::I16(i16::MIN))),
            Value::Int(IntValue::U8(0))
        );
        assert_eq!(
            call(Value::Int(IntValue::I16(i16::MAX))),
            Value::Int(IntValue::U8(255))
        );
        for (scalar, storage) in [
            (IntValue::I8(0), IntegerStorage::I8(vec![0])),
            (IntValue::I32(0), IntegerStorage::I32(vec![0])),
            (IntValue::I64(0), IntegerStorage::I64(vec![0])),
            (IntValue::U32(0), IntegerStorage::U32(vec![0])),
            (IntValue::U64(0), IntegerStorage::U64(vec![0])),
        ] {
            let scalar_err =
                block_on(im2uint8_builtin(Value::Int(scalar), Vec::new())).unwrap_err();
            assert_eq!(
                scalar_err.identifier(),
                IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
            );
            let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let tensor_err =
                block_on(im2uint8_builtin(Value::Tensor(tensor), Vec::new())).unwrap_err();
            assert_eq!(
                tensor_err.identifier(),
                IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
            );
        }
    }

    #[test]
    fn im2uint8_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = IM2UINT8_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec!["J = im2uint8(I)", "J = im2uint8(I, \"indexed\")"]
        );
    }

    #[test]
    fn indexed_mode_offsets_double_and_preserves_integer_indices() {
        let Value::Tensor(double_out) = block_on(im2uint8_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 256.0], vec![1, 3]).unwrap()),
            vec![Value::String("indexed".into())],
        ))
        .expect("indexed double") else {
            panic!("expected tensor");
        };
        assert_eq!(
            double_out.integer_storage(),
            Some(&IntegerStorage::U8(vec![0, 1, 255]))
        );

        let Value::Tensor(uint16_out) = block_on(im2uint8_builtin(
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 255]), vec![1, 3]).unwrap(),
            ),
            vec![Value::String("indexed".into())],
        ))
        .expect("indexed uint16") else {
            panic!("expected tensor");
        };
        assert_eq!(
            uint16_out.integer_storage(),
            Some(&IntegerStorage::U8(vec![0, 1, 255]))
        );
    }

    #[test]
    fn indexed_mode_maps_logical_to_zero_and_one_and_checks_range() {
        let Value::Tensor(logical_out) = block_on(im2uint8_builtin(
            Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap()),
            vec![Value::String("indexed".into())],
        ))
        .expect("indexed logical") else {
            panic!("expected tensor");
        };
        assert_eq!(
            logical_out.integer_storage(),
            Some(&IntegerStorage::U8(vec![0, 1]))
        );

        for input in [
            Tensor::new_integer(IntegerStorage::U16(vec![256]), vec![1, 1]).unwrap(),
            Tensor::new(vec![257.0], vec![1, 1]).unwrap(),
        ] {
            let err = block_on(im2uint8_builtin(
                Value::Tensor(input),
                vec![Value::String("indexed".into())],
            ))
            .expect_err("out-of-range indexed image");
            assert_eq!(err.identifier(), IM2UINT8_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn im2uint8_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = IM2UINT8_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.IM2UINT8.TOO_MANY_INPUTS"));
        assert!(codes.contains(&"RM.IM2UINT8.INVALID_INPUT"));
        assert!(codes.contains(&"RM.IM2UINT8.UNSUPPORTED_INPUT_TYPE"));
        assert!(codes.contains(&"RM.IM2UINT8.INTERNAL"));
    }

    #[test]
    fn im2uint8_too_many_args_uses_stable_identifier() {
        let err = block_on(im2uint8_builtin(
            Value::Num(1.0),
            vec![Value::from("indexed"), Value::Num(2.0)],
        ))
        .expect_err("expected argument error");
        assert_eq!(err.identifier(), IM2UINT8_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[test]
    fn im2uint8_resident_double_input_restores_exact_integer_output_to_owner() {
        test_support::with_test_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[0.0, 0.5, 1.0],
                    shape: &[1, 3],
                })
                .expect("upload double image");
            let Value::GpuTensor(output) =
                block_on(im2uint8_builtin(Value::GpuTensor(input), Vec::new()))
                    .expect("resident im2uint8")
            else {
                panic!("expected resident uint8 output");
            };
            assert_eq!(output.device_id, provider.device_id());
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(IntegerElementType::U8)
            );
            assert_eq!(
                block_on(provider.download_integer(&output))
                    .expect("download uint8 image")
                    .data,
                HostIntegerDataOwned::U8(vec![0, 128, 255])
            );
        });
    }
}
