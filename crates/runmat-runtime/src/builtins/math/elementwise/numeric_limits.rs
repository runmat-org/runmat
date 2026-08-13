//! MATLAB-compatible numeric limit query builtins.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
    NumericDType, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const INPUTS_CLASS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "classname",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric class name.",
}];

const INPUTS_FLOAT_CLASS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "classname",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Numeric class name.",
}];

const INPUTS_LIKE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer prototype whose class and complexity are copied.",
    },
];

const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Limit value.",
}];

const FLOAT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "value = limit()",
        inputs: &[],
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = limit(classname)",
        inputs: &INPUTS_FLOAT_CLASS,
        outputs: &OUTPUT_VALUE,
    },
];

const INTMIN_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "value = intmin()",
        inputs: &[],
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = intmin(classname)",
        inputs: &INPUTS_CLASS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = intmin(like=prototype)",
        inputs: &INPUTS_LIKE,
        outputs: &OUTPUT_VALUE,
    },
];

const INTMAX_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "value = intmax()",
        inputs: &[],
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = intmax(classname)",
        inputs: &INPUTS_CLASS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = intmax(like=prototype)",
        inputs: &INPUTS_LIKE,
        outputs: &OUTPUT_VALUE,
    },
];

const ERROR_INVALID_CLASS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUMERIC_LIMITS.INVALID_CLASS",
    identifier: Some("RunMat:numericLimits:InvalidClass"),
    when: "The requested class is not supported by the limit query.",
    message: "numeric limit: unsupported class",
};

const ERROR_INVALID_SYNTAX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUMERIC_LIMITS.INVALID_SYNTAX",
    identifier: Some("RunMat:numericLimits:InvalidSyntax"),
    when: "The arguments do not match a documented class-name or like-prototype form.",
    message: "numeric limit: invalid syntax",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_CLASS, ERROR_INVALID_SYNTAX];
const FLOAT_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_CLASS];

pub const FLOAT_LIMIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FLOAT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FLOAT_ERRORS,
};

pub const INTMIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTMIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const INTMAX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTMAX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_LIMIT_LIKE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "prototype",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The prototype selects one of the eight integer classes and may be real or structurally complex.",
    }];

pub const INTMAX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "value = intmax('like', integer_prototype)",
        inputs: &INTEGER_LIMIT_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns the exact maximum of the prototype class as a scalar and preserves prototype complexity and documented gpuArray residency where the provider representation supports it.",
    }];

pub const INTMIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "value = intmin('like', integer_prototype)",
        inputs: &INTEGER_LIMIT_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns the exact minimum of the prototype class as a scalar and preserves prototype complexity and documented gpuArray residency where the provider representation supports it.",
    }];

#[runtime_builtin(
    name = "intmax",
    category = "math/elementwise",
    summary = "Return the largest value of an integer class.",
    keywords = "intmax,integer,limits",
    descriptor(crate::builtins::math::elementwise::numeric_limits::INTMAX_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::math::elementwise::numeric_limits::INTMAX_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn intmax_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    integer_limit(rest, LimitKind::Maximum, "intmax")
}

#[runtime_builtin(
    name = "intmin",
    category = "math/elementwise",
    summary = "Return the smallest value of an integer class.",
    keywords = "intmin,integer,limits",
    descriptor(crate::builtins::math::elementwise::numeric_limits::INTMIN_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::math::elementwise::numeric_limits::INTMIN_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn intmin_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    integer_limit(rest, LimitKind::Minimum, "intmin")
}

#[runtime_builtin(
    name = "realmax",
    category = "math/elementwise",
    summary = "Return the largest finite floating-point value.",
    keywords = "realmax,float,limits,double,single",
    descriptor(crate::builtins::math::elementwise::numeric_limits::FLOAT_LIMIT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn realmax_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let class = parse_class(rest.first(), "double")?;
    match class.as_str() {
        "double" => Ok(Value::Num(f64::MAX)),
        "single" => Ok(Value::Num(f32::MAX as f64)),
        _ => Err(limit_error(
            "realmax",
            format!("unsupported float class '{class}'"),
        )),
    }
}

#[runtime_builtin(
    name = "realmin",
    category = "math/elementwise",
    summary = "Return the smallest positive normalized floating-point value.",
    keywords = "realmin,float,limits,double,single",
    descriptor(crate::builtins::math::elementwise::numeric_limits::FLOAT_LIMIT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn realmin_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let class = parse_class(rest.first(), "double")?;
    match class.as_str() {
        "double" => Ok(Value::Num(f64::MIN_POSITIVE)),
        "single" => Ok(Value::Num(f32::MIN_POSITIVE as f64)),
        _ => Err(limit_error(
            "realmin",
            format!("unsupported float class '{class}'"),
        )),
    }
}

#[runtime_builtin(
    name = "flintmax",
    category = "math/elementwise",
    summary = "Return the largest consecutive integer in a floating-point class.",
    keywords = "flintmax,float,limits,double,single",
    descriptor(crate::builtins::math::elementwise::numeric_limits::FLOAT_LIMIT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn flintmax_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let class = parse_class(rest.first(), "double")?;
    match class.as_str() {
        "double" => Ok(Value::Num(2f64.powi(53))),
        "single" => Ok(Value::Num(2f64.powi(24))),
        _ => Err(limit_error(
            "flintmax",
            format!("unsupported float class '{class}'"),
        )),
    }
}

fn parse_class(value: Option<&Value>, default: &str) -> BuiltinResult<String> {
    match value {
        None => Ok(default.to_string()),
        Some(Value::String(text)) => Ok(normalize_class(text)),
        Some(Value::CharArray(chars)) if chars.rows == 1 => {
            Ok(normalize_class(&chars.data.iter().collect::<String>()))
        }
        Some(Value::StringArray(array)) if array.data.len() == 1 => {
            Ok(normalize_class(&array.data[0]))
        }
        Some(_) => Err(limit_error(
            "numeric limit",
            "class name must be a string scalar or character vector",
        )),
    }
}

fn normalize_class(text: &str) -> String {
    text.trim().to_ascii_lowercase()
}

#[derive(Clone, Copy)]
enum LimitKind {
    Minimum,
    Maximum,
}

fn integer_limit(args: Vec<Value>, kind: LimitKind, builtin: &'static str) -> BuiltinResult<Value> {
    match args.as_slice() {
        [] => Ok(Value::Int(limit_scalar(NumericDType::I32, kind))),
        [class] if text_value(class).is_some() => {
            let class = normalize_class(&text_value(class).expect("guarded text value"));
            let dtype = integer_dtype(&class).ok_or_else(|| {
                limit_error(builtin, format!("unsupported integer class '{class}'"))
            })?;
            Ok(Value::Int(limit_scalar(dtype, kind)))
        }
        [keyword, prototype]
            if text_value(keyword).is_some_and(|text| text.eq_ignore_ascii_case("like")) =>
        {
            integer_limit_like(prototype, kind, builtin)
        }
        _ => Err(limit_syntax_error(
            builtin,
            "expected no arguments, an integer class name, or like=integerPrototype",
        )),
    }
}

fn integer_limit_like(
    prototype: &Value,
    kind: LimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match prototype {
        Value::Int(value) => Ok(Value::Int(limit_scalar(
            integer_dtype(value.class_name()).expect("IntValue has integer dtype"),
            kind,
        ))),
        Value::Tensor(tensor) => {
            let dtype = tensor.numeric_dtype();
            if matches!(dtype, NumericDType::F32 | NumericDType::F64) {
                return Err(invalid_integer_prototype(builtin));
            }
            Ok(Value::Int(limit_scalar(dtype, kind)))
        }
        Value::ComplexTensor(tensor) => {
            let Some(prototype_storage) = tensor.integer_storage() else {
                return Err(invalid_integer_prototype(builtin));
            };
            let value = limit_scalar(prototype_storage.real.numeric_dtype(), kind);
            let real = IntegerStorage::from_scalar(value);
            let imag = real.zeros_like(1);
            let storage = IntegerComplexStorage::new(real, imag)
                .map_err(|error| limit_error(builtin, error))?;
            ComplexTensor::new_integer(storage, vec![1, 1])
                .map(Value::ComplexTensor)
                .map_err(|error| limit_error(builtin, error))
        }
        Value::GpuTensor(handle) => {
            let Some(element_type) = runmat_accelerate_api::handle_integer_type(handle) else {
                return Err(invalid_integer_prototype(builtin));
            };
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
                || runmat_accelerate_api::handle_precision(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
                || !gpu_helpers::gpu_class_metadata_matches(handle, None, Some(element_type), false)
            {
                return Err(limit_error(
                    builtin,
                    "integer gpuArray prototype has contradictory class metadata",
                ));
            }
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                return Err(limit_error(
                    builtin,
                    "complex integer gpuArray prototypes are not supported by the current provider representation",
                ));
            }
            let dtype = dtype_from_integer_element_type(element_type);
            let storage = IntegerStorage::from_scalar(limit_scalar(dtype, kind));
            let shape = [1usize, 1usize];
            let view = integer_tensor_view(&storage, &shape);
            let provider = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
                limit_error(
                    builtin,
                    "integer gpuArray prototype has no registered provider",
                )
            })?;
            let provenance = runmat_accelerate_api::handle_provenance(handle)
                .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let input_metadata = gpu_helpers::snapshot_handle_metadata(handle);
            let output = provider.upload_integer(&view);
            gpu_helpers::restore_handle_metadata(handle, &input_metadata);
            let output = output.map_err(|error| {
                limit_error(builtin, format!("GPU limit creation failed: {error}"))
            })?;
            let valid = output.shape == shape
                && output.device_id == handle.device_id
                && !gpu_helpers::same_gpu_handle(handle, &output)
                && gpu_helpers::exact_provider_for_handle(&output)
                    .is_some_and(|owner| std::ptr::eq(owner, provider))
                && runmat_accelerate_api::handle_storage(&output)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                && runmat_accelerate_api::handle_integer_type(&output) == Some(element_type)
                && runmat_accelerate_api::handle_precision(&output).is_none()
                && !runmat_accelerate_api::handle_is_logical(&output)
                && gpu_helpers::gpu_class_metadata_matches(
                    &output,
                    None,
                    Some(element_type),
                    false,
                );
            if !valid {
                gpu_helpers::free_unprotected_exact_owner(&output, &[handle]);
                return Err(limit_error(
                    builtin,
                    "GPU limit creation returned an invalid provider result",
                ));
            }
            runmat_accelerate_api::set_handle_provenance(&output, provenance);
            Ok(gpu_helpers::resident_gpu_value(output))
        }
        _ => Err(invalid_integer_prototype(builtin)),
    }
}

fn text_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn integer_dtype(class: &str) -> Option<NumericDType> {
    match class {
        "int8" => Some(NumericDType::I8),
        "int16" => Some(NumericDType::I16),
        "int32" => Some(NumericDType::I32),
        "int64" => Some(NumericDType::I64),
        "uint8" => Some(NumericDType::U8),
        "uint16" => Some(NumericDType::U16),
        "uint32" => Some(NumericDType::U32),
        "uint64" => Some(NumericDType::U64),
        _ => None,
    }
}

fn limit_scalar(dtype: NumericDType, kind: LimitKind) -> IntValue {
    match (dtype, kind) {
        (NumericDType::I8, LimitKind::Minimum) => IntValue::I8(i8::MIN),
        (NumericDType::I8, LimitKind::Maximum) => IntValue::I8(i8::MAX),
        (NumericDType::I16, LimitKind::Minimum) => IntValue::I16(i16::MIN),
        (NumericDType::I16, LimitKind::Maximum) => IntValue::I16(i16::MAX),
        (NumericDType::I32, LimitKind::Minimum) => IntValue::I32(i32::MIN),
        (NumericDType::I32, LimitKind::Maximum) => IntValue::I32(i32::MAX),
        (NumericDType::I64, LimitKind::Minimum) => IntValue::I64(i64::MIN),
        (NumericDType::I64, LimitKind::Maximum) => IntValue::I64(i64::MAX),
        (NumericDType::U8, LimitKind::Minimum) => IntValue::U8(0),
        (NumericDType::U8, LimitKind::Maximum) => IntValue::U8(u8::MAX),
        (NumericDType::U16, LimitKind::Minimum) => IntValue::U16(0),
        (NumericDType::U16, LimitKind::Maximum) => IntValue::U16(u16::MAX),
        (NumericDType::U32, LimitKind::Minimum) => IntValue::U32(0),
        (NumericDType::U32, LimitKind::Maximum) => IntValue::U32(u32::MAX),
        (NumericDType::U64, LimitKind::Minimum) => IntValue::U64(0),
        (NumericDType::U64, LimitKind::Maximum) => IntValue::U64(u64::MAX),
        (NumericDType::F32 | NumericDType::F64, _) => {
            unreachable!("limit_scalar is only called for integer dtypes")
        }
    }
}

fn dtype_from_integer_element_type(
    element_type: runmat_accelerate_api::IntegerElementType,
) -> NumericDType {
    use runmat_accelerate_api::IntegerElementType;
    match element_type {
        IntegerElementType::I8 => NumericDType::I8,
        IntegerElementType::I16 => NumericDType::I16,
        IntegerElementType::I32 => NumericDType::I32,
        IntegerElementType::I64 => NumericDType::I64,
        IntegerElementType::U8 => NumericDType::U8,
        IntegerElementType::U16 => NumericDType::U16,
        IntegerElementType::U32 => NumericDType::U32,
        IntegerElementType::U64 => NumericDType::U64,
    }
}

fn integer_tensor_view<'a>(
    storage: &'a IntegerStorage,
    shape: &'a [usize],
) -> runmat_accelerate_api::HostIntegerTensorView<'a> {
    use runmat_accelerate_api::HostIntegerDataView;
    let data = match storage {
        IntegerStorage::I8(values) => HostIntegerDataView::I8(values),
        IntegerStorage::I16(values) => HostIntegerDataView::I16(values),
        IntegerStorage::I32(values) => HostIntegerDataView::I32(values),
        IntegerStorage::I64(values) => HostIntegerDataView::I64(values),
        IntegerStorage::U8(values) => HostIntegerDataView::U8(values),
        IntegerStorage::U16(values) => HostIntegerDataView::U16(values),
        IntegerStorage::U32(values) => HostIntegerDataView::U32(values),
        IntegerStorage::U64(values) => HostIntegerDataView::U64(values),
    };
    runmat_accelerate_api::HostIntegerTensorView { data, shape }
}

fn invalid_integer_prototype(builtin: &'static str) -> RuntimeError {
    limit_error(
        builtin,
        "like prototype must be an integer variable of class int8, int16, int32, int64, uint8, uint16, uint32, or uint64",
    )
}

fn limit_syntax_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = ERROR_INVALID_SYNTAX.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn limit_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = ERROR_INVALID_CLASS.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView};
    use runmat_builtins::Tensor;

    #[test]
    fn integer_limits_support_common_classes() {
        assert_eq!(
            intmax_builtin(Vec::new()).unwrap(),
            Value::Int(IntValue::I32(i32::MAX))
        );
        assert_eq!(
            intmin_builtin(vec![Value::from("uint16")]).unwrap(),
            Value::Int(IntValue::U16(0))
        );
        assert_eq!(
            intmax_builtin(vec![Value::from("uint32")]).unwrap(),
            Value::Int(IntValue::U32(u32::MAX))
        );
    }

    #[test]
    fn integer_limits_support_all_class_names_and_exact_wide_bounds() {
        let cases = [
            ("int8", IntValue::I8(i8::MIN), IntValue::I8(i8::MAX)),
            ("int16", IntValue::I16(i16::MIN), IntValue::I16(i16::MAX)),
            ("int32", IntValue::I32(i32::MIN), IntValue::I32(i32::MAX)),
            ("int64", IntValue::I64(i64::MIN), IntValue::I64(i64::MAX)),
            ("uint8", IntValue::U8(0), IntValue::U8(u8::MAX)),
            ("uint16", IntValue::U16(0), IntValue::U16(u16::MAX)),
            ("uint32", IntValue::U32(0), IntValue::U32(u32::MAX)),
            ("uint64", IntValue::U64(0), IntValue::U64(u64::MAX)),
        ];

        for (class, minimum, maximum) in cases {
            assert_eq!(
                intmin_builtin(vec![Value::from(class)]).unwrap(),
                Value::Int(minimum)
            );
            assert_eq!(
                intmax_builtin(vec![Value::from(class)]).unwrap(),
                Value::Int(maximum)
            );
        }
    }

    #[test]
    fn integer_limit_like_copies_every_integer_prototype_class() {
        let prototypes = [
            IntegerStorage::I8(vec![7]),
            IntegerStorage::I16(vec![7]),
            IntegerStorage::I32(vec![7]),
            IntegerStorage::I64(vec![9_007_199_254_740_993]),
            IntegerStorage::U8(vec![7]),
            IntegerStorage::U16(vec![7]),
            IntegerStorage::U32(vec![7]),
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
        ];

        for storage in prototypes {
            let dtype = storage.numeric_dtype();
            let prototype = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            assert_eq!(
                intmin_builtin(vec![Value::from("like"), Value::Tensor(prototype.clone())])
                    .unwrap(),
                Value::Int(limit_scalar(dtype, LimitKind::Minimum))
            );
            assert_eq!(
                intmax_builtin(vec![Value::from("like"), Value::Tensor(prototype)]).unwrap(),
                Value::Int(limit_scalar(dtype, LimitKind::Maximum))
            );
        }
    }

    #[test]
    fn integer_limit_like_copies_complexity_without_losing_uint64_max() {
        let prototype = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993]),
                IntegerStorage::U64(vec![1]),
            )
            .unwrap(),
            vec![1, 1],
        )
        .unwrap();

        let output =
            intmax_builtin(vec![Value::from("like"), Value::ComplexTensor(prototype)]).unwrap();
        let Value::ComplexTensor(output) = output else {
            panic!("expected complex integer scalar")
        };
        assert_eq!(output.shape, vec![1, 1]);
        assert_eq!(
            output.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![0]),
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn integer_limit_like_rejects_noninteger_prototypes_and_bad_syntax() {
        assert!(intmin_builtin(vec![Value::from("like"), Value::Num(0.0)]).is_err());
        assert!(intmax_builtin(vec![Value::Bool(true)]).is_err());
        assert!(intmax_builtin(vec![Value::from("uint8"), Value::from("uint16")]).is_err());
        assert!(intmin_builtin(vec![Value::from("int")]).is_err());
    }

    #[test]
    fn integer_limit_like_preserves_gpu_class_and_wide_value() {
        test_support::with_test_provider(|provider| {
            let shape = [1usize, 1usize];
            let prototype = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[9_007_199_254_740_993]),
                    shape: &shape,
                })
                .expect("integer prototype upload");

            let output = intmax_builtin(vec![
                Value::from("like"),
                Value::GpuTensor(prototype.clone()),
            ])
            .expect("gpu intmax like");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident integer output")
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let downloaded = block_on(provider.download_integer(&output)).expect("download");
            assert_eq!(downloaded.data, HostIntegerDataOwned::U64(vec![u64::MAX]));
            assert_eq!(downloaded.shape, vec![1, 1]);
            provider.free(&prototype).ok();
            provider.free(&output).ok();
        });
    }

    #[test]
    fn integer_limit_like_rejects_contradictory_resident_class_metadata() {
        test_support::with_test_provider(|provider| {
            let prototype = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[1]),
                    shape: &[1, 1],
                })
                .expect("integer prototype upload");
            runmat_accelerate_api::set_handle_class_name(&prototype, "double");
            let error = intmax_builtin(vec![
                Value::from("like"),
                Value::GpuTensor(prototype.clone()),
            ])
            .expect_err("contradictory resident prototype must reject");
            assert!(error.message().contains("contradictory class metadata"));
            provider.free(&prototype).ok();
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn integer_limit_like_preserves_wgpu_class_and_wide_value() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let shape = [1usize, 1usize];
        let prototype = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&[9_007_199_254_740_993]),
                shape: &shape,
            })
            .expect("WGPU integer prototype upload");

        let output = intmin_builtin(vec![
            Value::from("like"),
            Value::GpuTensor(prototype.clone()),
        ])
        .expect("WGPU intmin like");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident integer output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::I64)
        );
        let downloaded = block_on(provider.download_integer(&output)).expect("download");
        assert_eq!(downloaded.data, HostIntegerDataOwned::I64(vec![i64::MIN]));
        provider.free(&prototype).ok();
        provider.free(&output).ok();
    }

    #[test]
    fn floating_limits_support_single_and_double() {
        assert_eq!(realmax_builtin(Vec::new()).unwrap(), Value::Num(f64::MAX));
        assert_eq!(
            realmin_builtin(vec![Value::from("single")]).unwrap(),
            Value::Num(f32::MIN_POSITIVE as f64)
        );
        assert_eq!(
            flintmax_builtin(vec![Value::from("single")]).unwrap(),
            Value::Num(2f64.powi(24))
        );
    }
}
