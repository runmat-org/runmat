//! MATLAB-compatible `true`/`false` builtins for logical array creation.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, LogicalArray, NumericDType, NumericScalar, Value};

use crate::builtins::common::{shape::normalize_scalar_shape, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn builtin_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

const FALSE_IMPLICIT_PROTOTYPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "false-implicit-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "false(A) implicit size-prototype syntax is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FalseImplicitPrototypeExtension"),
};
const FALSE_LOGICAL_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "false-logical-class-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "false(...,'logical') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FalseLogicalOptionExtension"),
};
const FALSE_SINGLE_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "false-single-size-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision false size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FalseSingleSizeExtension"),
};
const FALSE_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "false-resident-size-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "resident false size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FalseResidentSizeExtension"),
};
pub const FALSE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FALSE_IMPLICIT_PROTOTYPE_EXTENSION,
    FALSE_LOGICAL_OPTION_EXTENSION,
    FALSE_SINGLE_SIZE_EXTENSION,
    FALSE_RESIDENT_SIZE_EXTENSION,
];

const FALSE_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz/szN",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls. Negative signed values clamp to zero.",
    }];
const FALSE_INTEGER_LIKE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer prototype selects applicable sparsity and residency; false output remains logical and the prototype never supplies shape.",
    }];
pub const FALSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "F = false(integer_n or integer_sz1,...,integer_szN)",
        inputs: &FALSE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Exact integer dimensions determine shape only; output is always logical.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "F = false(integer_sz)",
        inputs: &FALSE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented row size vector is read from authoritative integer storage without a floating mirror.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "F = false(..., like=integer_p)",
        inputs: &FALSE_INTEGER_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The logical output preserves applicable sparse or owning-provider residency but not the integer prototype class.",
    },
];

const LOGICAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical output array.",
}];

const LOGICAL_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const LOGICAL_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const LOGICAL_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const LOGICAL_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const LOGICAL_SIG_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype value when no numeric dimension arguments are provided.",
}];

const LOGICAL_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"logical\""),
        description: "Class override keyword (logical).",
    },
];

const LOGICAL_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype array used for class/device.",
    },
];

const TRUE_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "L = true()",
        inputs: &LOGICAL_SIG_EMPTY_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(n)",
        inputs: &LOGICAL_SIG_N_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(size_vector)",
        inputs: &LOGICAL_SIG_SIZE_VECTOR_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(m, n, ...)",
        inputs: &LOGICAL_SIG_DIMS_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(prototype)",
        inputs: &LOGICAL_SIG_PROTOTYPE_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(..., \"logical\")",
        inputs: &LOGICAL_SIG_CLASS_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(..., \"like\", prototype)",
        inputs: &LOGICAL_SIG_LIKE_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
];

const FALSE_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "L = false()",
        inputs: &LOGICAL_SIG_EMPTY_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(n)",
        inputs: &LOGICAL_SIG_N_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(size_vector)",
        inputs: &LOGICAL_SIG_SIZE_VECTOR_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(m, n, ...)",
        inputs: &LOGICAL_SIG_DIMS_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(prototype)",
        inputs: &LOGICAL_SIG_PROTOTYPE_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(..., \"logical\")",
        inputs: &LOGICAL_SIG_CLASS_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(..., \"like\", prototype)",
        inputs: &LOGICAL_SIG_LIKE_INPUTS,
        outputs: &LOGICAL_OUTPUT,
    },
];

const TRUE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.TRUE.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "true: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.TRUE.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "true: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.TRUE.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not supported.",
        message: "true: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.TRUE.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "true: dimension arguments must be numeric and nonnegative",
    },
];

const FALSE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.FALSE.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "false: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.FALSE.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "false: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.FALSE.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not supported.",
        message: "false: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.FALSE.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "false: dimension arguments must be numeric and nonnegative",
    },
];

pub const TRUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRUE_ERRORS,
};

pub const FALSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FALSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FALSE_ERRORS,
};

#[runtime_builtin(
    name = "true",
    category = "array/creation",
    summary = "Create logical arrays filled with `true` values.",
    keywords = "true,logical,array",
    accel = "array_construct",
    descriptor(crate::builtins::array::creation::true_false::TRUE_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::true_false"
)]
async fn true_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    logical_fill(rest, true, "true").await
}

#[runtime_builtin(
    name = "false",
    category = "array/creation",
    summary = "Create logical arrays filled with false values.",
    keywords = "false,logical,array",
    accel = "array_construct",
    descriptor(crate::builtins::array::creation::true_false::FALSE_DESCRIPTOR),
    extensions(FALSE_EXTENSIONS),
    integer_capabilities(crate::builtins::array::creation::true_false::FALSE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::true_false"
)]
async fn false_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    false_fill(rest).await
}

struct ParsedFalse {
    shape: Vec<usize>,
    prototype: Option<Value>,
}

async fn false_fill(args: Vec<Value>) -> BuiltinResult<Value> {
    let parsed = ParsedFalse::parse(args).await?;
    false_output(parsed).await
}

impl ParsedFalse {
    async fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        let mut dims = Vec::new();
        let mut saw_size_vector = false;
        let mut prototype = None;
        let mut saw_like = false;
        let mut implicit_shape = None;
        let mut idx = 0usize;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if saw_like {
                            return Err(builtin_error(
                                "false",
                                "false: multiple 'like' specifications are not supported",
                            ));
                        }
                        let Some(value) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error(
                                "false",
                                "false: expected prototype after 'like'",
                            ));
                        };
                        ensure_false_numeric_prototype(&value)?;
                        saw_like = true;
                        prototype = Some(value);
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &FALSE_LOGICAL_OPTION_EXTENSION,
                            "false",
                        )?;
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(
                            "false",
                            format!("false: unrecognised option '{other}'"),
                        ));
                    }
                }
            }

            if false_value_is_single_size(&arg) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FALSE_SINGLE_SIZE_EXTENSION,
                    "false",
                )?;
            }
            if matches!(arg, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FALSE_RESIDENT_SIZE_EXTENSION,
                    "false",
                )?;
            }
            if let Some(parsed) = extract_false_dims(&arg).await? {
                if parsed.is_vector {
                    if saw_size_vector || !dims.is_empty() {
                        return Err(builtin_error(
                            "false",
                            "false: multiple vector size inputs are not supported",
                        ));
                    }
                    saw_size_vector = true;
                } else if saw_size_vector {
                    return Err(builtin_error(
                        "false",
                        "false: a size vector cannot be combined with other dimensions",
                    ));
                }
                dims.extend(parsed.values);
                idx += 1;
                continue;
            }

            crate::compatibility::ensure_builtin_extension_enabled(
                &FALSE_IMPLICIT_PROTOTYPE_EXTENSION,
                "false",
            )?;
            if implicit_shape.is_none() {
                implicit_shape = Some(
                    shape_from_value(&arg)
                        .map_err(|error| builtin_error("false", format!("false: {error}")))?,
                );
                prototype = Some(arg);
            }
            idx += 1;
        }

        let shape = if !dims.is_empty() || saw_size_vector {
            normalize_false_shape(dims)
        } else if let Some(shape) = implicit_shape {
            normalize_false_shape(shape)
        } else {
            vec![1, 1]
        };
        shape
            .iter()
            .try_fold(1usize, |total, dim| total.checked_mul(*dim))
            .ok_or_else(|| builtin_error("false", "false: output size overflows usize"))?;
        Ok(Self { shape, prototype })
    }
}

fn ensure_false_numeric_prototype(value: &Value) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
    ) {
        Ok(())
    } else {
        Err(builtin_error(
            "false",
            "false: like prototype must be numeric or logical",
        ))
    }
}

async fn false_output(parsed: ParsedFalse) -> BuiltinResult<Value> {
    match parsed.prototype.as_ref() {
        Some(Value::SparseTensor(_)) => {
            let [rows, cols] = parsed.shape.as_slice() else {
                return Err(builtin_error(
                    "false",
                    "false: sparse like output must be two-dimensional",
                ));
            };
            Ok(Value::SparseTensor(
                runmat_value::SparseTensor::zeros_logical(*rows, *cols),
            ))
        }
        Some(Value::GpuTensor(prototype)) => false_gpu_output(prototype, &parsed.shape),
        _ => false_host_output(parsed.shape),
    }
}

fn false_host_output(shape: Vec<usize>) -> BuiltinResult<Value> {
    let len = shape.iter().product();
    if len == 1 && shape == [1, 1] {
        return Ok(Value::Bool(false));
    }
    LogicalArray::new(vec![0; len], shape)
        .map(Value::LogicalArray)
        .map_err(|error| builtin_error("false", format!("false: {error}")))
}

fn false_gpu_output(
    prototype: &runmat_accelerate_api::GpuTensorHandle,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(prototype)
        .ok_or_else(|| builtin_error("false", "false: GPU prototype has no owning provider"))?;
    let result = match provider.zeros(shape) {
        Ok(result) => result,
        Err(_) => {
            let host = runmat_value::Tensor::new(vec![0.0; shape.iter().product()], shape.to_vec())
                .map_err(|error| builtin_error("false", format!("false: {error}")))?;
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &host)
                .map_err(|error| builtin_error("false", format!("false: {error}")))?
        }
    };
    runmat_accelerate_api::set_handle_logical(&result, true);
    let valid = result.device_id == prototype.device_id
        && result.shape == shape
        && runmat_accelerate_api::provider_for_handle(&result)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&result)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(&result).is_none()
        && runmat_accelerate_api::handle_is_logical(&result);
    if !valid {
        let owner = runmat_accelerate_api::provider_for_handle(&result).unwrap_or(provider);
        let _ = owner.free(&result);
        return Err(builtin_error(
            "false",
            "false: provider returned an invalid logical result",
        ));
    }
    Ok(crate::builtins::common::gpu_helpers::logical_gpu_value(
        result,
    ))
}

struct FalseDims {
    values: Vec<usize>,
    is_vector: bool,
}

#[async_recursion::async_recursion(?Send)]
async fn extract_false_dims(value: &Value) -> BuiltinResult<Option<FalseDims>> {
    match value {
        Value::Num(value) => parse_false_float_dimension(*value).map(|value| {
            Some(FalseDims {
                values: vec![value],
                is_vector: false,
            })
        }),
        Value::Int(value) => parse_false_integer_dimension(value).map(|value| {
            Some(FalseDims {
                values: vec![value],
                is_vector: false,
            })
        }),
        Value::Tensor(value) => {
            let len = value.len();
            if len == 0 {
                return Ok(Some(FalseDims {
                    values: Vec::new(),
                    is_vector: true,
                }));
            }
            let scalar = len == 1;
            let row = value.shape.len() >= 2 && value.shape[0] == 1;
            let column = value.shape.len() >= 2 && value.shape[1] == 1;
            if column && !row && !scalar {
                return Err(builtin_error(
                    "false",
                    "false: size vector must be a row vector",
                ));
            }
            if !(scalar || row || value.shape.len() == 1) {
                return Ok(None);
            }
            let values = (0..len)
                .map(|index| {
                    value
                        .numeric_value_at(index)
                        .ok_or_else(|| builtin_error("false", "false: missing size value"))
                        .and_then(parse_false_numeric_dimension)
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(Some(FalseDims {
                values,
                is_vector: !scalar,
            }))
        }
        Value::GpuTensor(_) => {
            let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
            extract_false_dims(&gathered).await
        }
        _ => Ok(None),
    }
}

fn parse_false_numeric_dimension(value: NumericScalar) -> BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => parse_false_float_dimension(value),
        NumericScalar::F32(value) => parse_false_float_dimension(f64::from(value)),
        integer => parse_false_integer_dimension(
            &integer.into_int_value().expect("integer numeric scalar"),
        ),
    }
}

fn parse_false_float_dimension(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(builtin_error(
            "false",
            "false: dimensions must be finite integer values",
        ));
    }
    if value <= 0.0 {
        return Ok(0);
    }
    if value >= usize::MAX as f64 {
        return Err(builtin_error(
            "false",
            "false: dimension is outside the supported platform range",
        ));
    }
    Ok(value as usize)
}

fn parse_false_integer_dimension(value: &IntValue) -> BuiltinResult<usize> {
    let result = match value {
        IntValue::I8(value) => usize::try_from((*value).max(0)),
        IntValue::I16(value) => usize::try_from((*value).max(0)),
        IntValue::I32(value) => usize::try_from((*value).max(0)),
        IntValue::I64(value) => usize::try_from((*value).max(0)),
        IntValue::U8(value) => Ok(usize::from(*value)),
        IntValue::U16(value) => Ok(usize::from(*value)),
        IntValue::U32(value) => usize::try_from(*value),
        IntValue::U64(value) => usize::try_from(*value),
    };
    result.map_err(|_| {
        builtin_error(
            "false",
            "false: dimension is outside the supported platform range",
        )
    })
}

fn false_value_is_single_size(value: &Value) -> bool {
    matches!(value, Value::Tensor(value) if value.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(value) if runmat_accelerate_api::handle_integer_type(value).is_none() && runmat_accelerate_api::handle_precision(value) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn normalize_false_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        return vec![0, 0];
    }
    if shape.len() == 1 {
        return vec![shape[0], shape[0]];
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

async fn logical_fill(args: Vec<Value>, value: bool, name: &str) -> BuiltinResult<Value> {
    let parsed = ParsedLogical::parse(args, name).await?;
    let len = tensor::element_count(&parsed.shape);
    if len == 1 {
        return Ok(Value::Bool(value));
    }
    let data = vec![if value { 1u8 } else { 0u8 }; len];
    LogicalArray::new(data, parsed.shape)
        .map(Value::LogicalArray)
        .map_err(|e| builtin_error(name, format!("{name}: {e}")))
}

struct ParsedLogical {
    shape: Vec<usize>,
}

impl ParsedLogical {
    async fn parse(args: Vec<Value>, name: &str) -> BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut saw_like = false;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if saw_like {
                            return Err(builtin_error(
                                name,
                                format!("{name}: multiple 'like' specifications are not supported"),
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error(
                                name,
                                format!("{name}: expected prototype after 'like'"),
                            ));
                        };
                        saw_like = true;
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source =
                                Some(shape_from_value(&proto).map_err(|e| builtin_error(name, e))?);
                        }
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(
                            name,
                            format!("{name}: unrecognised option '{other}'"),
                        ));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg, name).await? {
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg).map_err(|e| builtin_error(name, e))?);
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = shape_source {
            shape
        } else {
            vec![1, 1]
        };

        Ok(Self { shape })
    }
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

async fn extract_dims(value: &Value, name: &str) -> BuiltinResult<Option<Vec<usize>>> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    let gpu_scalar = match value {
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape) == 1,
        _ => false,
    };
    match tensor::dims_from_value_async(value).await {
        Ok(dims) => Ok(dims),
        Err(err) => {
            if matches!(value, Value::Tensor(_))
                || (matches!(value, Value::GpuTensor(_)) && !gpu_scalar)
            {
                Ok(None)
            } else {
                Err(builtin_error(name, format!("{name}: {err}")))
            }
        }
    }
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(normalize_scalar_shape(&h.shape)),
        Value::CharArray(ca) => Ok(ca.shape.clone()),
        Value::Cell(cell) => Ok(cell.shape.clone()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::{IntegerStorage, SparseTensor, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn true_default_scalar() {
        let result = block_on(true_builtin(Vec::new())).expect("true");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn false_default_scalar() {
        let result = block_on(false_builtin(Vec::new())).expect("false");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn true_with_dims() {
        let args = vec![Value::Num(2.0), Value::Num(1.0)];
        let result = block_on(true_builtin(args)).expect("true");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 1]);
                assert!(logical.data.iter().all(|&x| x == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn false_from_size_vector() {
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = block_on(false_builtin(args)).expect("false");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 3]);
                assert!(logical.data.iter().all(|&x| x == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn false_reads_every_integer_size_class_exactly_and_clamps_negative() {
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];
        for storage in storages {
            let size = Tensor::new_integer(storage, vec![1, 1]).expect("size");
            let Value::LogicalArray(output) =
                block_on(false_builtin(vec![Value::Tensor(size)])).expect("false")
            else {
                panic!("expected logical array");
            };
            assert_eq!(output.shape, vec![2, 2]);
        }
        let Value::LogicalArray(empty) =
            block_on(false_builtin(vec![Value::Int(IntValue::I64(-7))])).expect("false")
        else {
            panic!("expected empty logical array");
        };
        assert_eq!(empty.shape, vec![0, 0]);
    }

    #[test]
    fn false_like_does_not_infer_shape_and_preserves_sparse_logical() {
        let prototype =
            Tensor::new_integer(IntegerStorage::U64(vec![9; 6]), vec![2, 3]).expect("prototype");
        assert_eq!(
            block_on(false_builtin(vec![
                Value::from("like"),
                Value::Tensor(prototype)
            ]))
            .expect("false like"),
            Value::Bool(false)
        );

        let sparse = SparseTensor::zeros(4, 5);
        let output = block_on(false_builtin(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::from("like"),
            Value::SparseTensor(sparse),
        ]))
        .expect("sparse false");
        assert!(
            matches!(output, Value::SparseTensor(ref value) if value.shape() == vec![2, 3] && value.numeric_dtype().is_none())
        );
    }

    #[test]
    fn false_strict_mode_gates_legacy_forms_before_evaluation() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let logical =
            block_on(false_builtin(vec![Value::Num(2.0), Value::from("logical")])).unwrap_err();
        assert_eq!(
            logical.identifier(),
            Some("RunMat:compatibility:FalseLogicalOptionExtension")
        );
        let matrix = Tensor::new(vec![0.0; 4], vec![2, 2]).expect("matrix");
        let implicit = block_on(false_builtin(vec![Value::Tensor(matrix)])).unwrap_err();
        assert_eq!(
            implicit.identifier(),
            Some("RunMat:compatibility:FalseImplicitPrototypeExtension")
        );
        let single = Tensor::from_f32(vec![2.0], vec![1, 1]).expect("single size");
        let single = block_on(false_builtin(vec![Value::Tensor(single)])).unwrap_err();
        assert_eq!(
            single.identifier(),
            Some("RunMat:compatibility:FalseSingleSizeExtension")
        );
    }

    #[test]
    fn false_resident_integer_size_is_structural_and_returns_host_logical() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let shape = [1usize, 1usize];
            let size = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&[2]),
                    shape: &shape,
                })
                .expect("resident size");
            let output = block_on(false_builtin(vec![Value::GpuTensor(size)])).expect("false");
            let Value::LogicalArray(output) = output else {
                panic!("expected host logical output");
            };
            assert_eq!(output.shape, vec![2, 2]);
            assert_eq!(output.data, vec![0; 4]);
        });
    }

    #[test]
    fn false_strict_mode_rejects_resident_size_before_provider_access() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let invalid = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let error = block_on(false_builtin(vec![invalid])).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FalseResidentSizeExtension")
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn false_wgpu_like_integer_prototype_preserves_owner_and_logical_storage() {
        let _guard = test_support::accel_test_lock();
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("WGPU provider");
        let provider = runmat_accelerate_api::provider().expect("registered WGPU provider");
        let size = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U16(&[2]),
                shape: &[1, 1],
            })
            .expect("resident integer size");
        let output = block_on(false_builtin(vec![
            Value::Int(IntValue::U16(2)),
            Value::from("like"),
            Value::GpuTensor(size),
        ]))
        .expect("false");
        let Value::GpuTensor(handle) = output else {
            panic!("expected resident logical output");
        };
        assert_eq!(handle.shape, vec![2, 2]);
        assert!(runmat_accelerate_api::handle_is_logical(&handle));
        assert!(runmat_accelerate_api::provider_for_handle(&handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider)));
        let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather false");
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![0.0; 4]);
    }
}
