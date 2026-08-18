//! MATLAB-compatible `true`/`false` builtins for logical array creation.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, LogicalArray, NumericDType, NumericScalar, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::shape::normalize_scalar_shape;
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
const TRUE_IMPLICIT_PROTOTYPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "true-implicit-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "true(A) implicit size-prototype syntax is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrueImplicitPrototypeExtension"),
};
const TRUE_LOGICAL_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "true-logical-class-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "true(...,'logical') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrueLogicalOptionExtension"),
};
const TRUE_SINGLE_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "true-single-size-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision true size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrueSingleSizeExtension"),
};
const TRUE_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "true-resident-size-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "resident true size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrueResidentSizeExtension"),
};
pub const TRUE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    TRUE_IMPLICIT_PROTOTYPE_EXTENSION,
    TRUE_LOGICAL_OPTION_EXTENSION,
    TRUE_SINGLE_SIZE_EXTENSION,
    TRUE_RESIDENT_SIZE_EXTENSION,
];
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
        notes: "The documented row size vector is decoded exactly from every supported integer class.",
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

const TRUE_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz/szN",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls. Negative signed values clamp to zero.",
    }];
const TRUE_INTEGER_LIKE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer prototype selects applicable sparsity and residency; true output remains logical and the prototype never supplies shape.",
    }];
pub const TRUE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "T = true(integer_n or integer_sz1,...,integer_szN)",
        inputs: &TRUE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Exact integer dimensions determine shape only; output is always logical.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = true(integer_sz)",
        inputs: &TRUE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented row size vector is decoded exactly from every supported integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = true(..., like=integer_p)",
        inputs: &TRUE_INTEGER_LIKE_INPUTS,
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
    extensions(TRUE_EXTENSIONS),
    integer_capabilities(crate::builtins::array::creation::true_false::TRUE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::true_false"
)]
async fn true_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    logical_fill(rest, true, &TRUE_FILL_CONFIG).await
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
    logical_fill(rest, false, &FALSE_FILL_CONFIG).await
}

struct LogicalFillConfig {
    name: &'static str,
    implicit_prototype: &'static BuiltinExtensionDescriptor,
    logical_option: &'static BuiltinExtensionDescriptor,
    single_size: &'static BuiltinExtensionDescriptor,
    resident_size: &'static BuiltinExtensionDescriptor,
}

const TRUE_FILL_CONFIG: LogicalFillConfig = LogicalFillConfig {
    name: "true",
    implicit_prototype: &TRUE_IMPLICIT_PROTOTYPE_EXTENSION,
    logical_option: &TRUE_LOGICAL_OPTION_EXTENSION,
    single_size: &TRUE_SINGLE_SIZE_EXTENSION,
    resident_size: &TRUE_RESIDENT_SIZE_EXTENSION,
};

const FALSE_FILL_CONFIG: LogicalFillConfig = LogicalFillConfig {
    name: "false",
    implicit_prototype: &FALSE_IMPLICIT_PROTOTYPE_EXTENSION,
    logical_option: &FALSE_LOGICAL_OPTION_EXTENSION,
    single_size: &FALSE_SINGLE_SIZE_EXTENSION,
    resident_size: &FALSE_RESIDENT_SIZE_EXTENSION,
};

struct ParsedLogicalFill {
    shape: Vec<usize>,
    prototype: Option<Value>,
}

async fn logical_fill(
    args: Vec<Value>,
    fill: bool,
    config: &LogicalFillConfig,
) -> BuiltinResult<Value> {
    let parsed = ParsedLogicalFill::parse(args, config).await?;
    logical_output(parsed, fill, config.name)
}

impl ParsedLogicalFill {
    async fn parse(args: Vec<Value>, config: &LogicalFillConfig) -> BuiltinResult<Self> {
        let name = config.name;
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
                                name,
                                format!("{name}: multiple 'like' specifications are not supported"),
                            ));
                        }
                        let Some(value) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error(
                                name,
                                format!("{name}: expected prototype after 'like'"),
                            ));
                        };
                        ensure_numeric_prototype(&value, name)?;
                        saw_like = true;
                        prototype = Some(value);
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            config.logical_option,
                            name,
                        )?;
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

            if value_is_single_size(&arg) {
                crate::compatibility::ensure_builtin_extension_enabled(config.single_size, name)?;
            }
            if matches!(arg, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(config.resident_size, name)?;
            }
            if let Some(parsed) = extract_logical_dims(&arg, name).await? {
                if parsed.is_vector {
                    if saw_size_vector || !dims.is_empty() {
                        return Err(builtin_error(
                            name,
                            format!("{name}: multiple vector size inputs are not supported"),
                        ));
                    }
                    saw_size_vector = true;
                } else if saw_size_vector {
                    return Err(builtin_error(
                        name,
                        format!("{name}: a size vector cannot be combined with other dimensions"),
                    ));
                }
                dims.extend(parsed.values);
                idx += 1;
                continue;
            }

            crate::compatibility::ensure_builtin_extension_enabled(
                config.implicit_prototype,
                name,
            )?;
            if implicit_shape.is_none() {
                implicit_shape = Some(
                    shape_from_value(&arg)
                        .map_err(|error| builtin_error(name, format!("{name}: {error}")))?,
                );
                prototype = Some(arg);
            }
            idx += 1;
        }

        let shape = if !dims.is_empty() || saw_size_vector {
            normalize_logical_shape(dims)
        } else if let Some(shape) = implicit_shape {
            normalize_logical_shape(shape)
        } else {
            vec![1, 1]
        };
        shape
            .iter()
            .try_fold(1usize, |total, dim| total.checked_mul(*dim))
            .ok_or_else(|| builtin_error(name, format!("{name}: output size overflows usize")))?;
        Ok(Self { shape, prototype })
    }
}

fn ensure_numeric_prototype(value: &Value, name: &str) -> BuiltinResult<()> {
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
            name,
            format!("{name}: like prototype must be numeric or logical"),
        ))
    }
}

fn logical_output(parsed: ParsedLogicalFill, fill: bool, name: &str) -> BuiltinResult<Value> {
    match parsed.prototype.as_ref() {
        Some(Value::SparseTensor(_)) => {
            let [rows, cols] = parsed.shape.as_slice() else {
                return Err(builtin_error(
                    name,
                    format!("{name}: sparse like output must be two-dimensional"),
                ));
            };
            logical_sparse_output(*rows, *cols, fill, name).map(Value::SparseTensor)
        }
        Some(Value::GpuTensor(prototype)) => {
            logical_gpu_output(prototype, &parsed.shape, fill, name)
        }
        _ => logical_host_output(parsed.shape, fill, name),
    }
}

fn logical_sparse_output(
    rows: usize,
    cols: usize,
    fill: bool,
    name: &str,
) -> BuiltinResult<runmat_builtins::SparseTensor> {
    if !fill {
        return Ok(runmat_builtins::SparseTensor::zeros_logical(rows, cols));
    }
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| builtin_error(name, format!("{name}: output size overflows usize")))?;
    let col_ptrs = (0..=cols)
        .map(|col| col.checked_mul(rows))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| builtin_error(name, format!("{name}: output size overflows usize")))?;
    let row_indices = (0..cols).flat_map(|_| 0..rows).collect();
    runmat_builtins::SparseTensor::new_logical(rows, cols, col_ptrs, row_indices)
        .map_err(|error| builtin_error(name, format!("{name}: {error}")))
        .and_then(|value| {
            if value.nnz() == len {
                Ok(value)
            } else {
                Err(builtin_error(
                    name,
                    format!("{name}: sparse construction failed"),
                ))
            }
        })
}

fn logical_host_output(shape: Vec<usize>, fill: bool, name: &str) -> BuiltinResult<Value> {
    let len = shape.iter().product();
    if len == 1 && shape == [1, 1] {
        return Ok(Value::Bool(fill));
    }
    LogicalArray::new(vec![u8::from(fill); len], shape)
        .map(Value::LogicalArray)
        .map_err(|error| builtin_error(name, format!("{name}: {error}")))
}

fn logical_gpu_output(
    prototype: &runmat_accelerate_api::GpuTensorHandle,
    shape: &[usize],
    fill: bool,
    name: &str,
) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(prototype).ok_or_else(|| {
        builtin_error(
            name,
            format!("{name}: GPU prototype has no owning provider"),
        )
    })?;
    let result = match if fill {
        provider.ones(shape)
    } else {
        provider.zeros(shape)
    } {
        Ok(result) => result,
        Err(_) => {
            let host = runmat_builtins::Tensor::new(
                vec![if fill { 1.0 } else { 0.0 }; shape.iter().product()],
                shape.to_vec(),
            )
            .map_err(|error| builtin_error(name, format!("{name}: {error}")))?;
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &host)
                .map_err(|error| builtin_error(name, format!("{name}: {error}")))?
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
            name,
            format!("{name}: provider returned an invalid logical result"),
        ));
    }
    Ok(crate::builtins::common::gpu_helpers::logical_gpu_value(
        result,
    ))
}

struct LogicalDims {
    values: Vec<usize>,
    is_vector: bool,
}

#[async_recursion::async_recursion(?Send)]
async fn extract_logical_dims(value: &Value, name: &str) -> BuiltinResult<Option<LogicalDims>> {
    match value {
        Value::Num(value) => parse_float_dimension(*value, name).map(|value| {
            Some(LogicalDims {
                values: vec![value],
                is_vector: false,
            })
        }),
        Value::Int(value) => parse_integer_dimension(value, name).map(|value| {
            Some(LogicalDims {
                values: vec![value],
                is_vector: false,
            })
        }),
        Value::Tensor(value) => {
            let len = value.len();
            if len == 0 {
                return Ok(Some(LogicalDims {
                    values: Vec::new(),
                    is_vector: true,
                }));
            }
            let scalar = len == 1;
            let row = value.shape.len() >= 2 && value.shape[0] == 1;
            let column = value.shape.len() >= 2 && value.shape[1] == 1;
            if column && !row && !scalar {
                return Err(builtin_error(
                    name,
                    format!("{name}: size vector must be a row vector"),
                ));
            }
            if !(scalar || row || value.shape.len() == 1) {
                return Ok(None);
            }
            let values = (0..len)
                .map(|index| {
                    value
                        .numeric_value_at(index)
                        .ok_or_else(|| builtin_error(name, format!("{name}: missing size value")))
                        .and_then(|value| parse_numeric_dimension(value, name))
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(Some(LogicalDims {
                values,
                is_vector: !scalar,
            }))
        }
        Value::GpuTensor(_) => {
            let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
            extract_logical_dims(&gathered, name).await
        }
        _ => Ok(None),
    }
}

fn parse_numeric_dimension(value: NumericScalar, name: &str) -> BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => parse_float_dimension(value, name),
        NumericScalar::F32(value) => parse_float_dimension(f64::from(value), name),
        integer => parse_integer_dimension(
            &integer.into_int_value().expect("integer numeric scalar"),
            name,
        ),
    }
}

fn parse_float_dimension(value: f64, name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(builtin_error(
            name,
            format!("{name}: dimensions must be finite integer values"),
        ));
    }
    if value <= 0.0 {
        return Ok(0);
    }
    if value >= usize::MAX as f64 {
        return Err(builtin_error(
            name,
            format!("{name}: dimension is outside the supported platform range"),
        ));
    }
    Ok(value as usize)
}

fn parse_integer_dimension(value: &IntValue, name: &str) -> BuiltinResult<usize> {
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
            name,
            format!("{name}: dimension is outside the supported platform range"),
        )
    })
}

fn value_is_single_size(value: &Value) -> bool {
    matches!(value, Value::Tensor(value) if value.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(value) if runmat_accelerate_api::handle_integer_type(value).is_none() && runmat_accelerate_api::handle_precision(value) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn normalize_logical_shape(mut shape: Vec<usize>) -> Vec<usize> {
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
    use runmat_builtins::{IntegerStorage, SparseTensor, Tensor};

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
    fn true_reads_every_integer_size_class_exactly_and_clamps_negative() {
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
                block_on(true_builtin(vec![Value::Tensor(size)])).expect("true")
            else {
                panic!("expected logical array");
            };
            assert_eq!(output.shape, vec![2, 2]);
            assert_eq!(output.data, vec![1; 4]);
        }
        let Value::LogicalArray(empty) =
            block_on(true_builtin(vec![Value::Int(IntValue::I64(-7))])).expect("true")
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
    fn true_like_does_not_infer_shape_and_preserves_sparse_logical() {
        let prototype =
            Tensor::new_integer(IntegerStorage::U64(vec![9; 6]), vec![2, 3]).expect("prototype");
        assert_eq!(
            block_on(true_builtin(vec![
                Value::from("like"),
                Value::Tensor(prototype)
            ]))
            .expect("true like"),
            Value::Bool(true)
        );

        let sparse = SparseTensor::zeros(4, 5);
        let output = block_on(true_builtin(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::from("like"),
            Value::SparseTensor(sparse),
        ]))
        .expect("sparse true");
        let Value::SparseTensor(output) = output else {
            panic!("expected sparse logical output");
        };
        assert_eq!(output.shape(), vec![2, 3]);
        assert!(output.numeric_dtype().is_none());
        assert_eq!(output.nnz(), 6);
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
    fn true_strict_mode_gates_legacy_forms_before_evaluation() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let logical =
            block_on(true_builtin(vec![Value::Num(2.0), Value::from("logical")])).unwrap_err();
        assert_eq!(
            logical.identifier(),
            Some("RunMat:compatibility:TrueLogicalOptionExtension")
        );
        let matrix = Tensor::new(vec![0.0; 4], vec![2, 2]).expect("matrix");
        let implicit = block_on(true_builtin(vec![Value::Tensor(matrix)])).unwrap_err();
        assert_eq!(
            implicit.identifier(),
            Some("RunMat:compatibility:TrueImplicitPrototypeExtension")
        );
        let single = Tensor::from_f32(vec![2.0], vec![1, 1]).expect("single size");
        let single = block_on(true_builtin(vec![Value::Tensor(single)])).unwrap_err();
        assert_eq!(
            single.identifier(),
            Some("RunMat:compatibility:TrueSingleSizeExtension")
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
            descriptor: Default::default(),
        });
        let error = block_on(false_builtin(vec![invalid])).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FalseResidentSizeExtension")
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn logical_wgpu_like_integer_prototype_preserves_owner_and_logical_storage() {
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
            Value::GpuTensor(size.clone()),
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

        let output = block_on(true_builtin(vec![
            Value::Int(IntValue::U16(2)),
            Value::from("like"),
            Value::GpuTensor(size),
        ]))
        .expect("true");
        let Value::GpuTensor(handle) = output else {
            panic!("expected resident logical output");
        };
        assert_eq!(handle.shape, vec![2, 2]);
        assert!(runmat_accelerate_api::handle_is_logical(&handle));
        assert!(runmat_accelerate_api::provider_for_handle(&handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider)));
        let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather true");
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![1.0; 4]);
    }
}
