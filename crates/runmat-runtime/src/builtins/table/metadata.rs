use runmat_builtins::TABLE_ERRORS;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
pub use runmat_builtins::{PARQUETINFO_DESCRIPTOR, PARQUETREAD_DESCRIPTOR};
pub(super) use runmat_builtins::{
    TABLE_ERROR_INVALID_ARGUMENT, TABLE_ERROR_INVALID_INDEX, TABLE_ERROR_INVALID_VARIABLE,
    TABLE_ERROR_IO, TABLE_ERROR_UNSUPPORTED_FILE,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};

use super::TABLE_CLASS;

const ANY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];
const READTABLE_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text or spreadsheet file path.",
}];
const READTABLE_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value import options.",
    },
];
const SPREADSHEET_IMPORT_OPTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "opts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed spreadsheet import-options object.",
}];
const SPREADSHEET_IMPORT_OPTIONS_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs.",
    }];
const DETECT_IMPORT_OPTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "opts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed text or spreadsheet import-options object for tabular readers.",
}];
const DETECT_IMPORT_OPTIONS_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or spreadsheet file path to inspect.",
    }];
const DETECT_IMPORT_OPTIONS_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or spreadsheet file path to inspect.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Detection overrides such as Delimiter, Range, Sheet, Encoding, or TextType.",
    },
];
const TABLE_INPUTS_VALUES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "variables",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Variables to assemble as table columns.",
}];
const TABLE_INPUTS_PREALLOCATE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Size",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element table size vector [rows variables].",
    },
    BuiltinParamDescriptor {
        name: "VariableTypes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String array or cellstr naming each preallocated variable type.",
    },
    BuiltinParamDescriptor {
        name: "VariableNames",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional names for the preallocated variables.",
    },
    BuiltinParamDescriptor {
        name: "RowNames",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional names for the preallocated rows.",
    },
];
const VALUE_AND_ARGS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options or conversion arguments.",
    },
];
const ARRAY_DATASTORE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arrds",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "ArrayDatastore object retaining the in-memory data and read properties.",
}];
const ARRAY_DATASTORE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "In-memory array managed by the datastore.",
}];
const ARRAY_DATASTORE_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "In-memory array managed by the datastore.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "ReadSize, IterationDimension, or OutputType options.",
    },
];
const VARIADIC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Input values and name-value options.",
}];
const WRITE_NO_OUTPUT: [BuiltinParamDescriptor; 0] = [];

const READTABLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = readtable(filename)",
        inputs: &READTABLE_INPUTS_FILENAME,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = readtable(filename, nameValuePairs...)",
        inputs: &READTABLE_INPUTS_NAME_VALUE,
        outputs: &ANY_OUTPUT,
    },
];
const SPREADSHEET_IMPORT_OPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "opts = spreadsheetImportOptions()",
        inputs: &[],
        outputs: &SPREADSHEET_IMPORT_OPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "opts = spreadsheetImportOptions(nameValuePairs...)",
        inputs: &SPREADSHEET_IMPORT_OPTIONS_INPUTS_NAME_VALUE,
        outputs: &SPREADSHEET_IMPORT_OPTIONS_OUTPUT,
    },
];
const DETECT_IMPORT_OPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "opts = detectImportOptions(filename)",
        inputs: &DETECT_IMPORT_OPTIONS_INPUTS_FILENAME,
        outputs: &DETECT_IMPORT_OPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "opts = detectImportOptions(filename, nameValuePairs...)",
        inputs: &DETECT_IMPORT_OPTIONS_INPUTS_NAME_VALUE,
        outputs: &DETECT_IMPORT_OPTIONS_OUTPUT,
    },
];
const DICTIONARY_KEYS_VALUES_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "keys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Uniform or cell-wrapped dictionary keys.",
    },
    BuiltinParamDescriptor {
        name: "values",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Uniform, scalar-expanded, or cell-wrapped dictionary values.",
    },
];
const DICTIONARY_PAIR_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "keyValuePair1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First key/value argument pair.",
    },
    BuiltinParamDescriptor {
        name: "keyValuePairN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional key/value argument pairs.",
    },
];
const DICTIONARY_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "d = dictionary()",
        inputs: &[],
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "d = dictionary(keys, values)",
        inputs: &DICTIONARY_KEYS_VALUES_INPUTS,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "d = dictionary(k1, v1, ..., kN, vN)",
        inputs: &DICTIONARY_PAIR_INPUTS,
        outputs: &ANY_OUTPUT,
    },
];
const TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = table(variables...)",
        inputs: &TABLE_INPUTS_VALUES,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = table(Size=sz, VariableTypes=varTypes, Name=Value...)",
        inputs: &TABLE_INPUTS_PREALLOCATE,
        outputs: &ANY_OUTPUT,
    },
];
const ARRAY_DATASTORE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "arrds = arrayDatastore(A)",
        inputs: &ARRAY_DATASTORE_INPUT,
        outputs: &ARRAY_DATASTORE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "arrds = arrayDatastore(A, nameValuePairs...)",
        inputs: &ARRAY_DATASTORE_INPUTS_NAME_VALUE,
        outputs: &ARRAY_DATASTORE_OUTPUT,
    },
];
const COMPAT_VALUE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = tabularBuiltin(A, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const COMPAT_VARIADIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = tabularBuiltin(args...)",
    inputs: &VARIADIC_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const WRITE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "writeTabular(T, filename, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &WRITE_NO_OUTPUT,
}];

pub const READTABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &READTABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPREADSHEET_IMPORT_OPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const DETECT_IMPORT_OPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DETECT_IMPORT_OPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const DICTIONARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DICTIONARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const ARRAY_DATASTORE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAY_DATASTORE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_COMPAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPAT_VALUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_VARIADIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPAT_VARIADIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_WRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::table")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "table",
    op_kind: GpuOpKind::Custom("table"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Tables are host containers. GPU variables are gathered when tabular algorithms need row-wise access.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::table")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "table",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Tables are structured host containers and are not fusion operands.",
};

pub(super) fn table_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(TABLE_CLASS);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn table_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(TABLE_CLASS)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_ARGUMENT, message)
}

pub(super) fn invalid_index(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_INDEX, message)
}

pub(super) fn invalid_variable(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_VARIABLE, message)
}

pub(super) fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(ToString::to_string);
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(TABLE_CLASS)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
