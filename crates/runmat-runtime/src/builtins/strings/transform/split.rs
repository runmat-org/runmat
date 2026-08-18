//! MATLAB-compatible `split` and `strsplit` builtins with GPU-aware semantics for RunMat.

use std::collections::HashSet;

use regex::RegexBuilder;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CellArray, CharArray, StringArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::builtins::strings::type_resolvers::{string_array_type, unknown_type};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::split")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "split",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes on the CPU; GPU-resident inputs are gathered to host memory before splitting.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::split")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "split",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion planning and always gathers GPU inputs.",
};

const BUILTIN_NAME: &str = "split";
const STRSPLIT_BUILTIN_NAME: &str = "strsplit";
const MAX_SPLIT_DIMENSION: usize = 1024;

const SPLIT_OUTPUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "newStr",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String or cell array containing split tokens.",
    },
    BuiltinParamDescriptor {
        name: "match",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "String or cell array containing the delimiters at which splitting occurred.",
    },
];

const SPLIT_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text scalar/array/cell to split.",
}];

const SPLIT_INPUTS_DELIMITER: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell to split.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar/array/cell.",
    },
];

const SPLIT_INPUTS_DELIMITER_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell to split.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar/array/cell.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive dimension along which output substrings are oriented.",
    },
];

const SPLIT_INPUTS_DELIMITER_NAMEVALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell to split.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar/array/cell.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name (`CollapseDelimiters` or `IncludeDelimiters`).",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option values and additional Name/Value pairs.",
    },
];

const SPLIT_INPUTS_NAMEVALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell to split.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name (`CollapseDelimiters` or `IncludeDelimiters`).",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option values and additional Name/Value pairs.",
    },
];

const SPLIT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "newStr = split(str)",
        inputs: &SPLIT_INPUTS_BASE,
        outputs: &SPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "newStr = split(str, delimiter)",
        inputs: &SPLIT_INPUTS_DELIMITER,
        outputs: &SPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[newStr, match] = split(str, delimiter, dim)",
        inputs: &SPLIT_INPUTS_DELIMITER_DIM,
        outputs: &SPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "newStr = split(str, delimiter, Name, Value, ...)",
        inputs: &SPLIT_INPUTS_DELIMITER_NAMEVALUE,
        outputs: &SPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "newStr = split(str, Name, Value, ...)",
        inputs: &SPLIT_INPUTS_NAMEVALUE,
        outputs: &SPLIT_OUTPUT,
    },
];

const SPLIT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.INVALID_INPUT",
    identifier: Some("RunMat:split:InvalidInput"),
    when: "First argument is not a string scalar/array, char array, or cell array of text scalars.",
    message:
        "split: first argument must be a string scalar, string array, character array, or cell array of character vectors",
};

const SPLIT_ERROR_DELIMITER_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.DELIMITER_TYPE",
    identifier: Some("RunMat:split:DelimiterType"),
    when: "Delimiter input is not a supported text scalar/array/cell.",
    message:
        "split: delimiter input must be a string scalar, string array, character array, or cell array of character vectors",
};

const SPLIT_ERROR_NAME_VALUE_PAIR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.NAME_VALUE_PAIR",
    identifier: Some("RunMat:split:NameValuePair"),
    when: "Name-value options are not supplied in complete pairs.",
    message: "split: name-value arguments must be supplied in pairs",
};

const SPLIT_ERROR_UNKNOWN_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.UNKNOWN_NAME",
    identifier: Some("RunMat:split:UnknownName"),
    when: "An option name is not recognized.",
    message:
        "split: unrecognized name-value argument; supported names are 'CollapseDelimiters' and 'IncludeDelimiters'",
};

const SPLIT_ERROR_EMPTY_DELIMITER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.EMPTY_DELIMITER",
    identifier: Some("RunMat:split:EmptyDelimiter"),
    when: "Delimiter list is empty or contains empty delimiter entries.",
    message: "split: delimiters must contain at least one character",
};

const SPLIT_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.CELL_ELEMENT",
    identifier: Some("RunMat:split:CellElement"),
    when: "Cell arrays contain non-text elements or non-row char arrays.",
    message: "split: cell array elements must be string scalars or character vectors",
};

const SPLIT_ERROR_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.OPTION_VALUE",
    identifier: Some("RunMat:split:OptionValue"),
    when: "Option values are not logical true/false values.",
    message: "split: option values must be logical true or false",
};

const SPLIT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.INTERNAL",
    identifier: Some("RunMat:split:InternalError"),
    when: "Internal output container construction failed.",
    message: "split: internal error",
};

const SPLIT_ERROR_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.DIMENSION",
    identifier: Some("RunMat:split:Dimension"),
    when: "The dimension argument is not a positive integer scalar in RunMat's supported rank range of 1 through 1024.",
    message: "split: dimension must be a positive integer scalar no greater than 1024",
};

const SPLIT_ERRORS: [BuiltinErrorDescriptor; 9] = [
    SPLIT_ERROR_INVALID_INPUT,
    SPLIT_ERROR_DELIMITER_TYPE,
    SPLIT_ERROR_NAME_VALUE_PAIR,
    SPLIT_ERROR_UNKNOWN_NAME,
    SPLIT_ERROR_EMPTY_DELIMITER,
    SPLIT_ERROR_CELL_ELEMENT,
    SPLIT_ERROR_OPTION_VALUE,
    SPLIT_ERROR_DIMENSION,
    SPLIT_ERROR_INTERNAL,
];

pub const SPLIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPLIT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPLIT_ERRORS,
};

const SPLIT_TYPED_DIMENSION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "split-typed-dimension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "split with a typed-integer dimension",
    error_identifier: Some("RunMat:compatibility:SplitTypedDimensionExtension"),
};

const SPLIT_RESIDENT_DIMENSION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "split-resident-dimension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "split with an explicitly GPU-resident dimension is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SplitResidentDimensionExtension"),
};

const SPLIT_ADVANCED_OPTIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "split-advanced-options",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "split CollapseDelimiters and IncludeDelimiters options are RunMat extensions",
    error_identifier: Some("RunMat:compatibility:SplitAdvancedOptionsExtension"),
};

pub const SPLIT_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    SPLIT_TYPED_DIMENSION_EXTENSION,
    SPLIT_RESIDENT_DIMENSION_EXTENSION,
    SPLIT_ADVANCED_OPTIONS_EXTENSION,
];

const SPLIT_INTEGER_DIMENSION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target specifies a positive integer dimension but does not enumerate typed storage classes. RunMat accepts every exact integer class behind a compatibility gate; ordinary host double integer dimensions remain documented behavior.",
    }];

pub const SPLIT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[newStr, match] = split(str, delimiter, integer_dim)",
        inputs: &SPLIT_INTEGER_DIMENSION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The typed dimension is a gated RunMat extension whose exact one-based value controls only output orientation. Explicit resident dimensions are separately gated before gather; automatic residency may gather transparently.",
    }];

const STRSPLIT_OUTPUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "parts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Split tokens.",
    },
    BuiltinParamDescriptor {
        name: "matches",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Matched delimiters when requested as second output.",
    },
];

const STRSPLIT_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String scalar or character vector input.",
}];

const STRSPLIT_INPUTS_DELIMITER: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String scalar or character vector input.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar/array/cell.",
    },
];

const STRSPLIT_INPUTS_DELIMITER_NAMEVALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String scalar or character vector input.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar/array/cell.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name (`CollapseDelimiters` or `DelimiterType`).",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option values and additional Name/Value pairs.",
    },
];

const STRSPLIT_INPUTS_NAMEVALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String scalar or character vector input.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name (`CollapseDelimiters` or `DelimiterType`).",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option values and additional Name/Value pairs.",
    },
];

const STRSPLIT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "[parts, matches] = strsplit(str)",
        inputs: &STRSPLIT_INPUTS_BASE,
        outputs: &STRSPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[parts, matches] = strsplit(str, delimiter)",
        inputs: &STRSPLIT_INPUTS_DELIMITER,
        outputs: &STRSPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[parts, matches] = strsplit(str, delimiter, Name, Value, ...)",
        inputs: &STRSPLIT_INPUTS_DELIMITER_NAMEVALUE,
        outputs: &STRSPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[parts, matches] = strsplit(str, Name, Value, ...)",
        inputs: &STRSPLIT_INPUTS_NAMEVALUE,
        outputs: &STRSPLIT_OUTPUT,
    },
];

const STRSPLIT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.INVALID_INPUT",
    identifier: Some("RunMat:strsplit:InvalidInput"),
    when: "First argument is not a string scalar or character vector.",
    message: "strsplit: first argument must be a string scalar or character vector",
};

const STRSPLIT_ERROR_DELIMITER_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.DELIMITER_TYPE",
    identifier: Some("RunMat:strsplit:DelimiterType"),
    when: "Delimiter input is not a supported text scalar/array/cell.",
    message:
        "strsplit: delimiter must be a character vector, string scalar, string array, or cell array of character vectors",
};

const STRSPLIT_ERROR_NAME_VALUE_PAIR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.NAME_VALUE_PAIR",
    identifier: Some("RunMat:strsplit:NameValuePair"),
    when: "Name-value options are not supplied in complete pairs.",
    message: "strsplit: name-value arguments must be supplied in pairs",
};

const STRSPLIT_ERROR_UNKNOWN_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.UNKNOWN_NAME",
    identifier: Some("RunMat:strsplit:UnknownName"),
    when: "An option name is not recognized.",
    message:
        "strsplit: unrecognized name-value argument; supported names are 'CollapseDelimiters' and 'DelimiterType'",
};

const STRSPLIT_ERROR_EMPTY_DELIMITER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.EMPTY_DELIMITER",
    identifier: Some("RunMat:strsplit:EmptyDelimiter"),
    when: "Delimiter list is empty or contains empty delimiter entries.",
    message: "strsplit: delimiters must contain at least one character",
};

const STRSPLIT_ERROR_DELIMITER_MODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.DELIMITER_MODE",
    identifier: Some("RunMat:strsplit:DelimiterMode"),
    when: "DelimiterType option is not `Simple` or `RegularExpression`.",
    message: "strsplit: value for 'DelimiterType' must be 'Simple' or 'RegularExpression'",
};

const STRSPLIT_ERROR_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.OPTION_VALUE",
    identifier: Some("RunMat:strsplit:OptionValue"),
    when: "Option values are not logical true/false values.",
    message: "strsplit: option values must be logical true or false",
};

const STRSPLIT_ERROR_REGEX_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.REGEX_INVALID",
    identifier: Some("RunMat:strsplit:RegexInvalid"),
    when: "Regular expression delimiter pattern fails to compile.",
    message: "strsplit: invalid delimiter regular expression",
};

const STRSPLIT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.INTERNAL",
    identifier: Some("RunMat:strsplit:InternalError"),
    when: "Internal output container construction failed.",
    message: "strsplit: internal error",
};

const STRSPLIT_ERRORS: [BuiltinErrorDescriptor; 9] = [
    STRSPLIT_ERROR_INVALID_INPUT,
    STRSPLIT_ERROR_DELIMITER_TYPE,
    STRSPLIT_ERROR_NAME_VALUE_PAIR,
    STRSPLIT_ERROR_UNKNOWN_NAME,
    STRSPLIT_ERROR_EMPTY_DELIMITER,
    STRSPLIT_ERROR_DELIMITER_MODE,
    STRSPLIT_ERROR_OPTION_VALUE,
    STRSPLIT_ERROR_REGEX_INVALID,
    STRSPLIT_ERROR_INTERNAL,
];

pub const STRSPLIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRSPLIT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRSPLIT_ERRORS,
};

pub const STRSPLIT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "strsplit accepts scalar text, text delimiters, and textual or logical options. Integer and provider-resident numeric values have no documented role and reject before provider access without implicit character conversion.",
};

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn split_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn split_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    split_error_with_message(error.message, error)
}

fn strsplit_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(STRSPLIT_BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn strsplit_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    strsplit_error_with_message(error.message, error)
}

#[runtime_builtin(
    name = "split",
    category = "strings/transform",
    summary = "Split text inputs into substrings using delimiter rules.",
    keywords = "split,strsplit,delimiter,CollapseDelimiters,IncludeDelimiters",
    accel = "sink",
    type_resolver(string_array_type),
    descriptor(crate::builtins::strings::transform::split::SPLIT_DESCRIPTOR),
    extensions(crate::builtins::strings::transform::split::SPLIT_EXTENSIONS),
    integer_capabilities(crate::builtins::strings::transform::split::SPLIT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::strings::transform::split"
)]
async fn split_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::dispatcher::value_contains_gpu(&text) {
        return Err(split_error(&SPLIT_ERROR_INVALID_INPUT));
    }
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let (args, dimension) = prepare_split_arguments(rest).await?;

    let options = SplitOptions::parse(&args)?;
    let matrix = TextMatrix::from_value(text)?;
    matrix.into_split_result(&options, dimension)
}

async fn prepare_split_arguments(
    mut rest: Vec<Value>,
) -> BuiltinResult<(Vec<Value>, Option<usize>)> {
    let dimension_index =
        if rest.len() >= 2 && !is_name_key(&rest[0]) && is_dimension_candidate(&rest[1]) {
            Some(1usize)
        } else {
            None
        };
    if rest.last().is_some_and(is_name_key) {
        return Err(split_error(&SPLIT_ERROR_NAME_VALUE_PAIR));
    }
    if rest.iter().any(is_name_key) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SPLIT_ADVANCED_OPTIONS_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if let Some(index) = dimension_index {
        let value = &rest[index];
        if is_typed_integer_dimension(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SPLIT_TYPED_DIMENSION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SPLIT_RESIDENT_DIMENSION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    let mut host = Vec::with_capacity(rest.len());
    for value in rest.drain(..) {
        host.push(gather_if_needed_async(&value).await.map_err(map_flow)?);
    }
    let dimension = if let Some(index) = dimension_index {
        let value = host.remove(index);
        Some(parse_split_dimension(&value)?)
    } else {
        None
    };
    Ok((host, dimension))
}

fn is_dimension_candidate(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

fn is_typed_integer_dimension(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(value) => value.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn parse_split_dimension(value: &Value) -> BuiltinResult<usize> {
    let dimension = match value {
        Value::Num(value) => positive_usize(*value),
        Value::Int(value) => value.try_to_usize().filter(|value| *value > 0),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            if let Some(integer) = value
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                integer.try_to_usize().filter(|value| *value > 0)
            } else {
                positive_usize(tensor::tensor_value_f64(value, 0))
            }
        }
        _ => None,
    };
    dimension
        .filter(|dimension| *dimension <= MAX_SPLIT_DIMENSION)
        .ok_or_else(|| split_error(&SPLIT_ERROR_DIMENSION))
}

fn positive_usize(value: f64) -> Option<usize> {
    if value.is_finite()
        && value >= 1.0
        && value.fract() == 0.0
        && (value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64))
    {
        Some(value as usize)
    } else {
        None
    }
}

#[runtime_builtin(
    name = "strsplit",
    category = "strings/transform",
    summary = "Split scalar text into substrings using simple or regex delimiters.",
    keywords = "strsplit,split,delimiter,CollapseDelimiters,DelimiterType,matches",
    accel = "sink",
    type_resolver(unknown_type),
    descriptor(crate::builtins::strings::transform::split::STRSPLIT_DESCRIPTOR),
    integer_audit(crate::builtins::strings::transform::split::STRSPLIT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::split"
)]
async fn strsplit_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::strings::common::contains_numeric_or_resident_text_input(&text)
        || rest.iter().any(|value| {
            matches!(
                value,
                Value::Num(_)
                    | Value::Int(_)
                    | Value::Tensor(_)
                    | Value::SparseTensor(_)
                    | Value::Complex(_, _)
                    | Value::ComplexTensor(_)
                    | Value::Symbolic(_)
                    | Value::GpuTensor(_)
            )
        })
    {
        return Err(split_error(&SPLIT_ERROR_INVALID_INPUT));
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(|err| map_control_flow_with_builtin(err, STRSPLIT_BUILTIN_NAME))?;
    let mut args = Vec::with_capacity(rest.len());
    for arg in rest {
        args.push(
            gather_if_needed_async(&arg)
                .await
                .map_err(|err| map_control_flow_with_builtin(err, STRSPLIT_BUILTIN_NAME))?,
        );
    }

    let (input_kind, subject) = extract_strsplit_subject(text)?;
    let options = StrsplitOptions::parse(&args)?;
    let (parts, matches) = strsplit_text(&subject, &options)?;
    let parts_value = make_strsplit_output(parts, input_kind)?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let matches_value = make_strsplit_output(matches, input_kind)?;
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![parts_value, matches_value],
        ));
    }

    Ok(parts_value)
}

#[derive(Clone)]
enum DelimiterSpec {
    Whitespace,
    Patterns(Vec<String>),
}

#[derive(Clone)]
struct SplitOptions {
    delimiters: DelimiterSpec,
    collapse_delimiters: bool,
    include_delimiters: bool,
}

impl SplitOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut index = 0usize;
        let mut delimiters = DelimiterSpec::Whitespace;

        if index < args.len() && !is_name_key(&args[index]) {
            let list = extract_delimiters(&args[index])?;
            if list.is_empty() {
                return Err(split_error(&SPLIT_ERROR_EMPTY_DELIMITER));
            }
            let mut seen = HashSet::new();
            let mut patterns: Vec<String> = Vec::new();
            for pattern in list {
                if pattern.is_empty() {
                    return Err(split_error(&SPLIT_ERROR_EMPTY_DELIMITER));
                }
                if seen.insert(pattern.clone()) {
                    patterns.push(pattern);
                }
            }
            patterns.sort_by_key(|pat| std::cmp::Reverse(pat.len()));
            delimiters = DelimiterSpec::Patterns(patterns);
            index += 1;
        }

        let mut collapse = match delimiters {
            DelimiterSpec::Whitespace => true,
            DelimiterSpec::Patterns(_) => false,
        };
        let mut include = false;

        while index < args.len() {
            let name = match name_key(&args[index]) {
                Some(NameKey::CollapseDelimiters) => NameKey::CollapseDelimiters,
                Some(NameKey::IncludeDelimiters) => NameKey::IncludeDelimiters,
                None => return Err(split_error(&SPLIT_ERROR_UNKNOWN_NAME)),
            };
            index += 1;
            if index >= args.len() {
                return Err(split_error(&SPLIT_ERROR_NAME_VALUE_PAIR));
            }
            let value = &args[index];
            index += 1;

            match name {
                NameKey::CollapseDelimiters => {
                    collapse = parse_bool(value, "CollapseDelimiters")?;
                }
                NameKey::IncludeDelimiters => {
                    include = parse_bool(value, "IncludeDelimiters")?;
                }
            }
        }

        Ok(Self {
            delimiters,
            collapse_delimiters: collapse,
            include_delimiters: include,
        })
    }
}

struct TextMatrix {
    data: Vec<String>,
    rows: usize,
    cols: usize,
}

impl TextMatrix {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Ok(Self {
                data: vec![text],
                rows: 1,
                cols: 1,
            }),
            Value::StringArray(array) => Ok(Self {
                data: array.data,
                rows: array.rows,
                cols: array.cols,
            }),
            Value::CharArray(array) => Self::from_char_array(array),
            Value::Cell(cell) => Self::from_cell_array(cell),
            _ => Err(split_error(&SPLIT_ERROR_INVALID_INPUT)),
        }
    }

    fn from_char_array(array: CharArray) -> BuiltinResult<Self> {
        let CharArray {
            data, rows, cols, ..
        } = array;
        if rows == 0 {
            return Ok(Self {
                data: Vec::new(),
                rows: 0,
                cols: 1,
            });
        }
        let mut strings = Vec::with_capacity(rows);
        for row in 0..rows {
            strings.push(char_row_to_string_slice(&data, cols, row));
        }
        Ok(Self {
            data: strings,
            rows,
            cols: 1,
        })
    }

    fn from_cell_array(cell: CellArray) -> BuiltinResult<Self> {
        let CellArray {
            data, rows, cols, ..
        } = cell;
        let mut strings = Vec::with_capacity(data.len());
        for col in 0..cols {
            for row in 0..rows {
                let idx = row * cols + col;
                let value_ref: &Value = &data[idx];
                strings.push(
                    cell_element_to_string(value_ref)
                        .ok_or_else(|| split_error(&SPLIT_ERROR_CELL_ELEMENT))?,
                );
            }
        }
        Ok(Self {
            data: strings,
            rows,
            cols,
        })
    }

    fn into_split_result(
        self,
        options: &SplitOptions,
        dimension: Option<usize>,
    ) -> BuiltinResult<Value> {
        let TextMatrix { data, rows, cols } = self;

        if data.is_empty() {
            let shape = vec![rows, cols];
            let parts = StringArray::new(Vec::new(), shape.clone()).map_err(|e| {
                split_error_with_message(format!("{BUILTIN_NAME}: {e}"), &SPLIT_ERROR_INTERNAL)
            })?;
            let parts = Value::StringArray(parts);
            if let Some(output_count) = crate::output_count::current_output_count() {
                if output_count == 0 {
                    return Ok(Value::OutputList(Vec::new()));
                }
                let matches = StringArray::new(Vec::new(), shape).map_err(|e| {
                    split_error_with_message(format!("{BUILTIN_NAME}: {e}"), &SPLIT_ERROR_INTERNAL)
                })?;
                return Ok(crate::output_count::output_list_with_padding(
                    output_count,
                    vec![parts, Value::StringArray(matches)],
                ));
            }
            return Ok(parts);
        }

        let mut per_element: Vec<Vec<String>> = Vec::with_capacity(data.len());
        let mut per_element_matches: Vec<Vec<String>> = Vec::with_capacity(data.len());
        let mut max_tokens = 0usize;
        let mut max_matches = 0usize;
        for text in &data {
            let tokens = split_text(text, options);
            let matches = split_delimiter_matches(text, options);
            max_tokens = max_tokens.max(tokens.len());
            max_matches = max_matches.max(matches.len());
            per_element.push(tokens);
            per_element_matches.push(matches);
        }
        if max_tokens == 0 {
            max_tokens = 1;
        }
        let dimension = dimension.unwrap_or_else(|| default_split_dimension(rows, cols));
        let (output, shape) = orient_split_values(&per_element, rows, cols, dimension, max_tokens)?;
        let array = StringArray::new(output, shape).map_err(|e| {
            split_error_with_message(format!("{BUILTIN_NAME}: {e}"), &SPLIT_ERROR_INTERNAL)
        })?;
        let parts = Value::StringArray(array);
        if let Some(output_count) = crate::output_count::current_output_count() {
            if output_count == 0 {
                return Ok(Value::OutputList(Vec::new()));
            }
            let (matches, match_shape) =
                orient_split_values(&per_element_matches, rows, cols, dimension, max_matches)?;
            let matches = StringArray::new(matches, match_shape).map_err(|error| {
                split_error_with_message(format!("{BUILTIN_NAME}: {error}"), &SPLIT_ERROR_INTERNAL)
            })?;
            return Ok(crate::output_count::output_list_with_padding(
                output_count,
                vec![parts, Value::StringArray(matches)],
            ));
        }
        Ok(parts)
    }
}

fn default_split_dimension(rows: usize, cols: usize) -> usize {
    if rows <= 1 && cols <= 1 {
        1
    } else if cols <= 1 {
        2
    } else {
        3
    }
}

fn orient_split_values(
    per_element: &[Vec<String>],
    rows: usize,
    cols: usize,
    dimension: usize,
    value_count: usize,
) -> BuiltinResult<(Vec<String>, Vec<usize>)> {
    let mut input_shape = vec![rows, cols.max(1)];
    while input_shape.len() < dimension {
        input_shape.push(1);
    }
    let dimension_index = dimension - 1;
    let last_non_singleton = input_shape.iter().rposition(|size| *size != 1);
    let replace_trailing_singleton = input_shape[dimension_index] == 1
        && last_non_singleton.is_none_or(|index| dimension_index > index);
    let mut output_shape = input_shape.clone();
    if replace_trailing_singleton {
        output_shape[dimension_index] = value_count;
    } else {
        output_shape.insert(dimension_index, value_count);
    }
    let total = output_shape
        .iter()
        .try_fold(1usize, |product, size| product.checked_mul(*size));
    let total = total.ok_or_else(|| split_error(&SPLIT_ERROR_INTERNAL))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(total)
        .map_err(|_| split_error(&SPLIT_ERROR_INTERNAL))?;
    output.resize(total, "<missing>".to_string());
    for (input_linear, values) in per_element.iter().enumerate() {
        let input_coords = linear_to_subscripts(input_linear, &input_shape);
        for value_index in 0..value_count {
            let mut output_coords = input_coords.clone();
            if replace_trailing_singleton {
                output_coords[dimension_index] = value_index;
            } else {
                output_coords.insert(dimension_index, value_index);
            }
            if let Some(value) = values.get(value_index) {
                let output_linear = subscripts_to_linear(&output_coords, &output_shape);
                output[output_linear] = value.clone();
            }
        }
    }
    while output_shape.len() > 2 && output_shape.last() == Some(&1) {
        output_shape.pop();
    }
    Ok((output, output_shape))
}

fn linear_to_subscripts(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut subscripts = Vec::with_capacity(shape.len());
    for size in shape {
        if *size == 0 {
            subscripts.push(0);
        } else {
            subscripts.push(linear % size);
            linear /= size;
        }
    }
    subscripts
}

fn subscripts_to_linear(subscripts: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut linear = 0usize;
    for (subscript, size) in subscripts.iter().zip(shape) {
        linear += subscript * stride;
        stride *= size;
    }
    linear
}

fn split_text(text: &str, options: &SplitOptions) -> Vec<String> {
    if is_missing_string(text) {
        return vec![text.to_string()];
    }
    match &options.delimiters {
        DelimiterSpec::Whitespace => split_whitespace(text, options),
        DelimiterSpec::Patterns(patterns) => split_by_patterns(text, patterns, options),
    }
}

fn split_delimiter_matches(text: &str, options: &SplitOptions) -> Vec<String> {
    if text.is_empty() || is_missing_string(text) {
        return Vec::new();
    }
    match &options.delimiters {
        DelimiterSpec::Whitespace => {
            let mut matches = Vec::new();
            let mut index = 0usize;
            while index < text.len() {
                let character = text[index..]
                    .chars()
                    .next()
                    .expect("valid character boundary");
                if !character.is_whitespace() {
                    index += character.len_utf8();
                    continue;
                }
                if options.collapse_delimiters {
                    let end = advance_whitespace(text, index);
                    matches.push(text[index..end].to_string());
                    index = end;
                } else {
                    let end = index + character.len_utf8();
                    matches.push(text[index..end].to_string());
                    index = end;
                }
            }
            matches
        }
        DelimiterSpec::Patterns(patterns) => {
            let mut matches = Vec::new();
            let mut index = 0usize;
            while index < text.len() {
                let Some(pattern) = patterns
                    .iter()
                    .find(|candidate| text[index..].starts_with(candidate.as_str()))
                else {
                    index += text[index..]
                        .chars()
                        .next()
                        .expect("valid character boundary")
                        .len_utf8();
                    continue;
                };
                let mut end = index + pattern.len();
                if options.collapse_delimiters {
                    while end < text.len() {
                        let Some(next) = patterns
                            .iter()
                            .find(|candidate| text[end..].starts_with(candidate.as_str()))
                        else {
                            break;
                        };
                        end += next.len();
                    }
                }
                matches.push(text[index..end].to_string());
                index = end;
            }
            matches
        }
    }
}

fn split_whitespace(text: &str, options: &SplitOptions) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut parts: Vec<String> = Vec::new();
    let mut idx = 0usize;
    let mut last = 0usize;
    let len = text.len();

    while idx < len {
        let ch = text[idx..].chars().next().unwrap();
        let width = ch.len_utf8();
        if !ch.is_whitespace() {
            idx += width;
            continue;
        }

        let token = &text[last..idx];
        if !token.is_empty() || !options.collapse_delimiters {
            parts.push(token.to_string());
        }

        let run_end = advance_whitespace(text, idx);
        if options.include_delimiters {
            if options.collapse_delimiters {
                parts.push(text[idx..run_end].to_string());
            } else {
                parts.push(text[idx..idx + width].to_string());
            }
        }

        if options.collapse_delimiters {
            idx = run_end;
            last = run_end;
        } else {
            idx += width;
            last = idx;
        }
    }

    let tail = &text[last..];
    if !tail.is_empty() || !options.collapse_delimiters {
        parts.push(tail.to_string());
    }
    if parts.is_empty() {
        parts.push(String::new());
    }
    parts
}

fn split_by_patterns(text: &str, patterns: &[String], options: &SplitOptions) -> Vec<String> {
    if patterns.is_empty() {
        return vec![text.to_string()];
    }

    let mut parts: Vec<String> = Vec::new();
    let mut idx = 0usize;
    let mut last = 0usize;
    while idx < text.len() {
        if let Some(pattern) = patterns
            .iter()
            .find(|candidate| text[idx..].starts_with(candidate.as_str()))
        {
            let token = &text[last..idx];
            if !token.is_empty() || !options.collapse_delimiters {
                parts.push(token.to_string());
            }

            let pat_len = pattern.len();
            if options.collapse_delimiters {
                let mut run_end = idx + pat_len;
                while run_end < text.len() {
                    if let Some(next) = patterns
                        .iter()
                        .find(|candidate| text[run_end..].starts_with(candidate.as_str()))
                    {
                        let len = next.len();
                        if len == 0 {
                            break;
                        }
                        run_end += len;
                    } else {
                        break;
                    }
                }
                if options.include_delimiters {
                    parts.push(text[idx..run_end].to_string());
                }
                idx = run_end;
                last = run_end;
            } else {
                if options.include_delimiters {
                    parts.push(text[idx..idx + pat_len].to_string());
                }
                idx += pat_len;
                last = idx;
            }

            continue;
        }
        let ch = text[idx..].chars().next().unwrap();
        idx += ch.len_utf8();
    }
    let tail = &text[last..];
    if !tail.is_empty() || !options.collapse_delimiters {
        parts.push(tail.to_string());
    }
    if parts.is_empty() {
        parts.push(String::new());
    }
    parts
}

fn advance_whitespace(text: &str, mut start: usize) -> usize {
    while start < text.len() {
        let ch = text[start..].chars().next().unwrap();
        if !ch.is_whitespace() {
            break;
        }
        start += ch.len_utf8();
    }
    start
}

fn extract_delimiters(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(array) => {
            if array.rows == 0 {
                return Ok(Vec::new());
            }
            let mut entries = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                entries.push(char_row_to_string_slice(&array.data, array.cols, row));
            }
            Ok(entries)
        }
        Value::Cell(cell) => {
            let mut entries = Vec::with_capacity(cell.data.len());
            for element in &cell.data {
                entries.push(
                    cell_element_to_string(element)
                        .ok_or_else(|| split_error(&SPLIT_ERROR_CELL_ELEMENT))?,
                );
            }
            Ok(entries)
        }
        _ => Err(split_error(&SPLIT_ERROR_DELIMITER_TYPE)),
    }
}

fn cell_element_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows <= 1 => {
            if array.rows == 0 {
                Some(String::new())
            } else {
                Some(char_row_to_string_slice(&array.data, array.cols, 0))
            }
        }
        _ => None,
    }
}

fn value_to_scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows <= 1 => {
            if array.rows == 0 {
                Some(String::new())
            } else {
                Some(char_row_to_string_slice(&array.data, array.cols, 0))
            }
        }
        Value::Cell(cell) if cell.data.len() == 1 => cell_element_to_string(&cell.data[0]),
        _ => None,
    }
}

fn parse_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    parse_bool_for_builtin(value, name, BUILTIN_NAME, &SPLIT_ERROR_OPTION_VALUE)
}

fn parse_bool_for_builtin(
    value: &Value,
    name: &str,
    builtin_name: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => Ok(*n != 0.0),
        Value::LogicalArray(array) => {
            if array.data.len() == 1 {
                Ok(array.data[0] != 0)
            } else {
                Err(builtin_error_with_descriptor(
                    builtin_name,
                    format!(
                        "{builtin_name}: value for '{}' must be logical true or false",
                        name
                    ),
                    error,
                ))
            }
        }
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                if let Some(value) = tensor
                    .integer_storage()
                    .and_then(|storage| storage.value_at(0))
                {
                    return Ok(!value.is_zero());
                }
                Ok(tensor::tensor_value_f64(tensor, 0) != 0.0)
            } else {
                Err(builtin_error_with_descriptor(
                    builtin_name,
                    format!(
                        "{builtin_name}: value for '{}' must be logical true or false",
                        name
                    ),
                    error,
                ))
            }
        }
        _ => {
            if let Some(text) = value_to_scalar_string(value) {
                let lowered = text.trim().to_ascii_lowercase();
                match lowered.as_str() {
                    "true" | "on" | "yes" => Ok(true),
                    "false" | "off" | "no" => Ok(false),
                    _ => Err(builtin_error_with_descriptor(
                        builtin_name,
                        format!(
                            "{builtin_name}: value for '{}' must be logical true or false",
                            name
                        ),
                        error,
                    )),
                }
            } else {
                Err(builtin_error_with_descriptor(
                    builtin_name,
                    format!(
                        "{builtin_name}: value for '{}' must be logical true or false",
                        name
                    ),
                    error,
                ))
            }
        }
    }
}

fn builtin_error_with_descriptor(
    builtin_name: &'static str,
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin_name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn extract_strsplit_subject(value: Value) -> BuiltinResult<(StrsplitInputKind, String)> {
    match value {
        Value::String(text) => Ok((StrsplitInputKind::String, text)),
        Value::StringArray(array) if array.data.len() == 1 => {
            Ok((StrsplitInputKind::String, array.data[0].clone()))
        }
        Value::CharArray(array) if array.rows <= 1 => {
            if array.rows == 0 {
                Ok((StrsplitInputKind::Char, String::new()))
            } else {
                Ok((
                    StrsplitInputKind::Char,
                    char_row_to_string_slice(&array.data, array.cols, 0),
                ))
            }
        }
        _ => Err(strsplit_error(&STRSPLIT_ERROR_INVALID_INPUT)),
    }
}

fn strsplit_text(
    text: &str,
    options: &StrsplitOptions,
) -> BuiltinResult<(Vec<String>, Vec<String>)> {
    let regex = compile_strsplit_regex(options)?;
    let mut parts = Vec::new();
    let mut matches = Vec::new();
    let mut last = 0usize;

    for found in regex.find_iter(text) {
        parts.push(text[last..found.start()].to_string());
        matches.push(found.as_str().to_string());
        last = found.end();
    }

    parts.push(text[last..].to_string());
    Ok((parts, matches))
}

fn compile_strsplit_regex(options: &StrsplitOptions) -> BuiltinResult<regex::Regex> {
    let pattern = match (&options.delimiters, options.delimiter_type) {
        (None, _) => {
            if options.collapse_delimiters {
                "[\\x20\\x0C\\n\\r\\t\\x0B]+".to_string()
            } else {
                "[\\x20\\x0C\\n\\r\\t\\x0B]".to_string()
            }
        }
        (Some(delimiters), StrsplitDelimiterType::Simple) => {
            let alternation = delimiters
                .iter()
                .map(|pattern| regex::escape(pattern))
                .collect::<Vec<_>>()
                .join("|");
            if options.collapse_delimiters {
                format!("(?:{alternation})+")
            } else {
                format!("(?:{alternation})")
            }
        }
        (Some(delimiters), StrsplitDelimiterType::RegularExpression) => {
            let alternation = delimiters.join("|");
            if options.collapse_delimiters {
                format!("(?:{alternation})+")
            } else {
                format!("(?:{alternation})")
            }
        }
    };

    RegexBuilder::new(&pattern).build().map_err(|err| {
        strsplit_error_with_message(format!("strsplit: {err}"), &STRSPLIT_ERROR_REGEX_INVALID)
    })
}

fn make_strsplit_output(tokens: Vec<String>, kind: StrsplitInputKind) -> BuiltinResult<Value> {
    match kind {
        StrsplitInputKind::String => {
            let len = tokens.len();
            let array = StringArray::new(tokens, vec![1, len]).map_err(|err| {
                strsplit_error_with_message(format!("strsplit: {err}"), &STRSPLIT_ERROR_INTERNAL)
            })?;
            Ok(Value::StringArray(array))
        }
        StrsplitInputKind::Char => {
            let values: Vec<Value> = tokens.into_iter().map(Value::String).collect();
            let len = values.len();
            make_cell(values, 1, len).map_err(|err| {
                strsplit_error_with_message(format!("strsplit: {err}"), &STRSPLIT_ERROR_INTERNAL)
            })
        }
    }
}

#[derive(PartialEq, Eq)]
enum NameKey {
    CollapseDelimiters,
    IncludeDelimiters,
}

#[derive(Clone, Copy)]
enum StrsplitInputKind {
    Char,
    String,
}

#[derive(Clone, Copy)]
enum StrsplitDelimiterType {
    Simple,
    RegularExpression,
}

#[derive(Clone)]
struct StrsplitOptions {
    delimiters: Option<Vec<String>>,
    collapse_delimiters: bool,
    delimiter_type: StrsplitDelimiterType,
}

impl StrsplitOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut index = 0usize;
        let mut delimiters = None;

        if index < args.len() && !is_strsplit_name_key(&args[index]) {
            let list = extract_delimiters(&args[index])
                .map_err(|_| strsplit_error(&STRSPLIT_ERROR_DELIMITER_TYPE))?;
            delimiters = Some(list);
            index += 1;
        }

        let mut collapse_delimiters = true;
        let mut delimiter_type = StrsplitDelimiterType::Simple;

        while index < args.len() {
            let name = match strsplit_name_key(&args[index]) {
                Some(name) => name,
                None => return Err(strsplit_error(&STRSPLIT_ERROR_UNKNOWN_NAME)),
            };
            index += 1;
            if index >= args.len() {
                return Err(strsplit_error(&STRSPLIT_ERROR_NAME_VALUE_PAIR));
            }
            let value = &args[index];
            index += 1;

            match name {
                StrsplitNameKey::CollapseDelimiters => {
                    collapse_delimiters = parse_bool_for_builtin(
                        value,
                        "CollapseDelimiters",
                        STRSPLIT_BUILTIN_NAME,
                        &STRSPLIT_ERROR_OPTION_VALUE,
                    )?;
                }
                StrsplitNameKey::DelimiterType => {
                    let text = value_to_scalar_string(value)
                        .ok_or_else(|| strsplit_error(&STRSPLIT_ERROR_DELIMITER_MODE))?;
                    delimiter_type = match text.trim().to_ascii_lowercase().as_str() {
                        "simple" => StrsplitDelimiterType::Simple,
                        "regularexpression" => StrsplitDelimiterType::RegularExpression,
                        _ => return Err(strsplit_error(&STRSPLIT_ERROR_DELIMITER_MODE)),
                    };
                }
            }
        }

        if let Some(patterns) = &delimiters {
            if patterns.is_empty() {
                return Err(strsplit_error(&STRSPLIT_ERROR_EMPTY_DELIMITER));
            }
            if matches!(delimiter_type, StrsplitDelimiterType::Simple)
                && patterns.iter().any(|pattern| pattern.is_empty())
            {
                return Err(strsplit_error(&STRSPLIT_ERROR_EMPTY_DELIMITER));
            }
        }

        Ok(Self {
            delimiters,
            collapse_delimiters,
            delimiter_type,
        })
    }
}

#[derive(PartialEq, Eq)]
enum StrsplitNameKey {
    CollapseDelimiters,
    DelimiterType,
}

fn is_name_key(value: &Value) -> bool {
    name_key(value).is_some()
}

fn is_strsplit_name_key(value: &Value) -> bool {
    strsplit_name_key(value).is_some()
}

fn name_key(value: &Value) -> Option<NameKey> {
    value_to_scalar_string(value).and_then(|text| {
        let lowered = text.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "collapsedelimiters" => Some(NameKey::CollapseDelimiters),
            "includedelimiters" => Some(NameKey::IncludeDelimiters),
            _ => None,
        }
    })
}

fn strsplit_name_key(value: &Value) -> Option<StrsplitNameKey> {
    value_to_scalar_string(value).and_then(|text| {
        let lowered = text.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "collapsedelimiters" => Some(StrsplitNameKey::CollapseDelimiters),
            "delimitertype" => Some(StrsplitNameKey::DelimiterType),
            _ => None,
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, IntValue, IntegerStorage, LogicalArray, ResolveContext, Tensor, Type,
    };

    fn split_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::split_builtin(text, rest))
    }

    fn strsplit_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::strsplit_builtin(text, rest))
    }

    #[test]
    fn split_bool_options_read_wide_uint64_truth_exactly() {
        assert!(parse_bool(&Value::Int(IntValue::U64(u64::MAX)), "IncludeDelimiters").unwrap());

        for storage in [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![u64::MAX]),
        ] {
            let enabled = Tensor::new_integer(storage, vec![1, 1]).expect("enabled");
            assert!(parse_bool(&Value::Tensor(enabled), "IncludeDelimiters").unwrap());
        }

        let disabled =
            Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).expect("disabled");
        assert!(!parse_bool(&Value::Tensor(disabled), "IncludeDelimiters").unwrap());
    }

    #[test]
    fn split_gathers_automatic_double_dimension_but_gates_explicit_dimension() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .expect("resident dimension");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let result = split_builtin(
                Value::String("Mary Butler".into()),
                vec![Value::String(" ".into()), Value::GpuTensor(handle.clone())],
            )
            .expect("automatic residency is transparent");
            let Value::StringArray(result) = result else {
                panic!("expected string array")
            };
            assert_eq!(result.shape, vec![2, 1]);

            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let error = split_builtin(
                Value::String("Mary Butler".into()),
                vec![Value::String(" ".into()), Value::GpuTensor(handle.clone())],
            )
            .expect_err("explicit resident dimension is gated");
            assert_eq!(
                error.identifier(),
                SPLIT_RESIDENT_DIMENSION_EXTENSION.error_identifier
            );
            provider.free(&handle).expect("free dimension");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_string_whitespace_default() {
        let input = Value::String("RunMat Accelerate Planner".to_string());
        let result = split_builtin(input, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![3, 1]);
                assert_eq!(
                    array.data,
                    vec![
                        "RunMat".to_string(),
                        "Accelerate".to_string(),
                        "Planner".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_string_custom_delimiter() {
        let input = Value::String("alpha,beta,gamma".to_string());
        let args = vec![Value::String(",".to_string())];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![3, 1]);
                assert_eq!(
                    array.data,
                    vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_include_delimiters_true() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Value::String("A+B-C".to_string());
        let args = vec![
            Value::StringArray(
                StringArray::new(vec!["+".to_string(), "-".to_string()], vec![1, 2]).unwrap(),
            ),
            Value::String("IncludeDelimiters".to_string()),
            Value::Bool(true),
        ];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![5, 1]);
                assert_eq!(
                    array.data,
                    vec![
                        "A".to_string(),
                        "+".to_string(),
                        "B".to_string(),
                        "-".to_string(),
                        "C".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_include_delimiters_whitespace_collapse_default() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Value::String("A  B".to_string());
        let args = vec![
            Value::String("IncludeDelimiters".to_string()),
            Value::Bool(true),
        ];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![3, 1]);
                assert_eq!(
                    array.data,
                    vec!["A".to_string(), "  ".to_string(), "B".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_patterns_include_delimiters_collapse_true() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Value::String("a,,b".to_string());
        let args = vec![
            Value::String(",".to_string()),
            Value::String("IncludeDelimiters".to_string()),
            Value::Bool(true),
            Value::String("CollapseDelimiters".to_string()),
            Value::Bool(true),
        ];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![3, 1]);
                assert_eq!(
                    array.data,
                    vec!["a".to_string(), ",,".to_string(), "b".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_collapse_false_preserves_empty_segments() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Value::String("one,,three,".to_string());
        let args = vec![
            Value::String(",".to_string()),
            Value::String("CollapseDelimiters".to_string()),
            Value::Bool(false),
        ];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![4, 1]);
                assert_eq!(
                    array.data,
                    vec![
                        "one".to_string(),
                        "".to_string(),
                        "three".to_string(),
                        "".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_character_array_rows() {
        let mut row1: Vec<char> = "GPU Accelerate".chars().collect();
        let mut row2: Vec<char> = "VM Engine".chars().collect();
        let width = row1.len().max(row2.len());
        row1.resize(width, ' ');
        row2.resize(width, ' ');
        let mut data = row1;
        data.extend(row2);
        let char_array = CharArray::new(data, 2, width).unwrap();
        let input = Value::CharArray(char_array);
        let result = split_builtin(input, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(
                    array.data,
                    vec![
                        "GPU".to_string(),
                        "VM".to_string(),
                        "Accelerate".to_string(),
                        "Engine".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_string_array_multiple_columns() {
        let data = vec![
            "RunMat Core".to_string(),
            "VM Interpreter".to_string(),
            "Accelerate Engine".to_string(),
            "<missing>".to_string(),
        ];
        let array = StringArray::new(data, vec![2, 2]).unwrap();
        let input = Value::StringArray(array);
        let result = split_builtin(input, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 2, 2]);
                assert_eq!(
                    array.data,
                    vec![
                        "RunMat".to_string(),
                        "VM".to_string(),
                        "Accelerate".to_string(),
                        "<missing>".to_string(),
                        "Core".to_string(),
                        "Interpreter".to_string(),
                        "Engine".to_string(),
                        "<missing>".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_cell_array_outputs_string_array() {
        let values = vec![
            Value::String("RunMat Snapshot".to_string()),
            Value::String("Fusion Planner".to_string()),
        ];
        let cell = crate::make_cell(values, 2, 1).expect("cell");
        let result = split_builtin(cell, vec![Value::String(" ".to_string())]).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(
                    array.data,
                    vec![
                        "RunMat".to_string(),
                        "Fusion".to_string(),
                        "Snapshot".to_string(),
                        "Planner".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_cell_array_multiple_columns() {
        let values = vec![
            Value::String("alpha beta".to_string()),
            Value::String("gamma".to_string()),
            Value::String("delta epsilon".to_string()),
            Value::String("<missing>".to_string()),
        ];
        let cell = crate::make_cell(values, 2, 2).expect("cell");
        let result = split_builtin(cell, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 2, 2]);
                assert_eq!(
                    array.data,
                    vec![
                        "alpha".to_string(),
                        "delta".to_string(),
                        "gamma".to_string(),
                        "<missing>".to_string(),
                        "beta".to_string(),
                        "epsilon".to_string(),
                        "<missing>".to_string(),
                        "<missing>".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_missing_string_propagates() {
        let input = Value::String("<missing>".to_string());
        let result = split_builtin(input, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 1]);
                assert_eq!(array.data, vec!["<missing>".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_invalid_name_value_pair_errors() {
        let input = Value::String("abc".to_string());
        let args = vec![Value::String("CollapseDelimiters".to_string())];
        let err = split_builtin(input, args).unwrap_err();
        assert!(err.to_string().contains("name-value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_invalid_text_argument_errors() {
        let err = split_builtin(Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("first argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_invalid_delimiter_type_errors() {
        let err =
            split_builtin(Value::String("abc".to_string()), vec![Value::Num(1.0)]).unwrap_err();
        assert!(err.to_string().contains("delimiter input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_empty_delimiter_errors() {
        let err = split_builtin(
            Value::String("abc".to_string()),
            vec![Value::String(String::new())],
        )
        .unwrap_err();
        assert!(err.to_string().contains("at least one character"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_unknown_name_argument_errors() {
        let err = split_builtin(
            Value::String("abc".to_string()),
            vec![
                Value::String("UnknownOption".to_string()),
                Value::Bool(true),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("unrecognized"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_collapse_delimiters_accepts_logical_array() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1u8], vec![1]).unwrap();
        let args = vec![
            Value::String(",".to_string()),
            Value::String("CollapseDelimiters".to_string()),
            Value::LogicalArray(logical),
        ];
        let result = split_builtin(Value::String("a,,b".to_string()), args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 1]);
                assert_eq!(array.data, vec!["a".to_string(), "b".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_include_delimiters_accepts_tensor_scalar() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let args = vec![
            Value::String(",".to_string()),
            Value::String("IncludeDelimiters".to_string()),
            Value::Tensor(tensor),
        ];
        let result = split_builtin(Value::String("a,b".to_string()), args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![3, 1]);
                assert_eq!(
                    array.data,
                    vec!["a".to_string(), ",".to_string(), "b".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_cell_array_mixed_inputs() {
        let values = vec![
            Value::String("alpha beta".to_string()),
            Value::CharArray(CharArray::new("gamma".chars().collect(), 1, 5).unwrap()),
        ];
        let cell = Value::Cell(CellArray::new(values, 1, 2).expect("cell array construction"));
        let result = split_builtin(cell, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 2, 2]);
                assert_eq!(
                    array.data,
                    vec![
                        "alpha".to_string(),
                        "gamma".to_string(),
                        "beta".to_string(),
                        "<missing>".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_typed_integer_dimensions_cover_all_classes_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ] {
            let dimension = Tensor::new_integer(storage, vec![1, 1]).expect("dimension");
            let input = StringArray::new(
                vec![
                    "Mary Butler".into(),
                    "Diana Lee".into(),
                    "James King".into(),
                ],
                vec![3, 1],
            )
            .expect("input");
            let result = split_builtin(
                Value::StringArray(input),
                vec![Value::String(" ".into()), Value::Tensor(dimension)],
            )
            .expect("typed dimension");
            let Value::StringArray(result) = result else {
                panic!("expected string array")
            };
            assert_eq!(result.shape, vec![2, 3]);
            assert_eq!(
                result.data,
                vec!["Mary", "Butler", "Diana", "Lee", "James", "King"]
            );
        }
    }

    #[test]
    fn split_typed_dimension_is_gated_but_documented_double_dimension_remains_available() {
        let input = || Value::String("alpha beta".into());
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_dimension = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1])
            .expect("integer dimension");
        let error = split_builtin(
            input(),
            vec![Value::String(" ".into()), Value::Tensor(integer_dimension)],
        )
        .expect_err("typed dimension extension");
        assert_eq!(
            error.identifier(),
            SPLIT_TYPED_DIMENSION_EXTENSION.error_identifier
        );

        let Value::StringArray(result) =
            split_builtin(input(), vec![Value::String(" ".into()), Value::Num(1.0)])
                .expect("documented double dimension")
        else {
            panic!("expected string array")
        };
        assert_eq!(result.shape, vec![2, 1]);
    }

    #[test]
    fn split_rejects_dimensions_above_the_supported_rank_without_allocating() {
        let error = split_builtin(
            Value::String("alpha beta".into()),
            vec![
                Value::String(" ".into()),
                Value::Num((MAX_SPLIT_DIMENSION + 1) as f64),
            ],
        )
        .expect_err("oversized double dimension");
        assert_eq!(error.identifier(), SPLIT_ERROR_DIMENSION.identifier);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let dimension = Tensor::new_integer(
            IntegerStorage::U64(vec![(MAX_SPLIT_DIMENSION + 1) as u64]),
            vec![1, 1],
        )
        .expect("typed dimension");
        let error = split_builtin(
            Value::String("alpha beta".into()),
            vec![Value::String(" ".into()), Value::Tensor(dimension)],
        )
        .expect_err("oversized typed dimension");
        assert_eq!(error.identifier(), SPLIT_ERROR_DIMENSION.identifier);
    }

    #[test]
    fn split_empty_input_honors_requested_output_count() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let input = StringArray::new(Vec::new(), vec![0, 1]).expect("empty input");
        let result = split_builtin(Value::StringArray(input), Vec::new()).expect("empty split");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list")
        };
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            let Value::StringArray(array) = output else {
                panic!("expected empty string array")
            };
            assert!(array.data.is_empty());
            assert_eq!(array.shape, vec![0, 1]);
        }
    }

    #[test]
    fn split_second_output_contains_matched_delimiters_in_requested_orientation() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let result = split_builtin(
            Value::String("a,b,c".into()),
            vec![Value::String(",".into()), Value::Num(2.0)],
        )
        .expect("two-output split");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list")
        };
        let Value::StringArray(parts) = &outputs[0] else {
            panic!("expected string parts")
        };
        let Value::StringArray(matches) = &outputs[1] else {
            panic!("expected string matches")
        };
        assert_eq!(parts.shape, vec![1, 3]);
        assert_eq!(parts.data, vec!["a", "b", "c"]);
        assert_eq!(matches.shape, vec![1, 2]);
        assert_eq!(matches.data, vec![",", ","]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strsplit_string_scalar_returns_string_array() {
        let result =
            strsplit_builtin(Value::String("one two  three".into()), Vec::new()).expect("strsplit");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(
                    array.data,
                    vec!["one".to_string(), "two".to_string(), "three".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strsplit_char_vector_returns_cell() {
        let input = Value::CharArray(CharArray::new("a,b".chars().collect(), 1, 3).unwrap());
        let result = strsplit_builtin(input, vec![Value::String(",".into())]).expect("strsplit");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 2);
                assert_eq!(&cell.data[0], &Value::String("a".into()));
                assert_eq!(&cell.data[1], &Value::String("b".into()));
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strsplit_multi_output_returns_matches() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = strsplit_builtin(
            Value::String("a,,b,".into()),
            vec![Value::String(",".into())],
        )
        .expect("strsplit");
        match result {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::StringArray(array) => {
                        assert_eq!(
                            array.data,
                            vec!["a".to_string(), "b".to_string(), "".to_string()]
                        );
                    }
                    other => panic!("expected first output string array, got {other:?}"),
                }
                match &values[1] {
                    Value::StringArray(array) => {
                        assert_eq!(array.data, vec![",,".to_string(), ",".to_string()]);
                    }
                    other => panic!("expected second output string array, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strsplit_regular_expression_mode() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = strsplit_builtin(
            Value::String("1.21m/s 1.985 m/s".into()),
            vec![
                Value::String("\\s*m/s\\s*".into()),
                Value::String("DelimiterType".into()),
                Value::String("RegularExpression".into()),
            ],
        )
        .expect("strsplit");
        match result {
            Value::OutputList(values) => {
                match &values[0] {
                    Value::StringArray(array) => {
                        assert_eq!(
                            array.data,
                            vec!["1.21".to_string(), "1.985".to_string(), "".to_string()]
                        );
                    }
                    other => panic!("expected split output string array, got {other:?}"),
                }
                match &values[1] {
                    Value::StringArray(array) => {
                        assert_eq!(array.data, vec!["m/s ".to_string(), " m/s".to_string()]);
                    }
                    other => panic!("expected matches output string array, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strsplit_collapse_false_preserves_empty_segments() {
        let result = strsplit_builtin(
            Value::String("a,,b".into()),
            vec![
                Value::String(",".into()),
                Value::String("CollapseDelimiters".into()),
                Value::Bool(false),
            ],
        )
        .expect("strsplit");
        match result {
            Value::StringArray(array) => {
                assert_eq!(
                    array.data,
                    vec!["a".to_string(), "".to_string(), "b".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strsplit_rejects_nonscalar_text_inputs() {
        let input = Value::StringArray(
            StringArray::new(vec!["a b".into(), "c d".into()], vec![2, 1]).unwrap(),
        );
        let err = strsplit_builtin(input, Vec::new()).unwrap_err();
        assert!(err.to_string().contains("first argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strsplit_invalid_delimiter_type_option_errors() {
        let err = strsplit_builtin(
            Value::String("a,b".into()),
            vec![
                Value::String(",".into()),
                Value::String("DelimiterType".into()),
                Value::String("BadMode".into()),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("DelimiterType"));
    }

    #[test]
    fn split_type_is_string_array() {
        assert_eq!(
            string_array_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::cell_of(Type::String)
        );
    }
}
