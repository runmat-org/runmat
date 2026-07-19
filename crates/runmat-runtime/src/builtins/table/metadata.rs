use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
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
const NUM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Count.",
}];
const TABLE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table input.",
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
    description: "Spreadsheet import options struct.",
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
    description: "Detected import options struct accepted by readtable/readmatrix.",
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
const PARQUETREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parquet file path.",
}];
const PARQUETREAD_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parquet file path.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Options such as SelectedVariableNames, RowGroups, or OutputType.",
    },
];
const PARQUETINFO_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parquet file path to inspect.",
}];
const TABLE_INPUTS_VALUES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "variables",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Variables to assemble as table columns.",
}];
const GROUPSUMMARY_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input table.",
    },
    BuiltinParamDescriptor {
        name: "groupvars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping variable name or names.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary method name or names.",
    },
    BuiltinParamDescriptor {
        name: "datavars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Data variable name or names.",
    },
];
const GRPSTATS_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix or table.",
    },
    BuiltinParamDescriptor {
        name: "group",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping variables, variable selectors, or empty grouping.",
    },
    BuiltinParamDescriptor {
        name: "whichstats",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"mean\""),
        description: "Summary statistic name or names.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Alpha, DataVars, and VarNames options.",
    },
];
const OBJECT_INDEX_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table object receiver.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index payload.",
    },
];
const OBJECT_ASSIGN_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table object receiver.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index payload.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];
const VALUE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value.",
}];
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
const VARIADIC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Input values and name-value options.",
}];
const PREDICATE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TF",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predicate result.",
}];
const WRITE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytesWritten",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bytes written.",
}];

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
const PARQUETREAD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = parquetread(filename)",
        inputs: &PARQUETREAD_INPUTS_FILENAME,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = parquetread(filename, nameValuePairs...)",
        inputs: &PARQUETREAD_INPUTS_NAME_VALUE,
        outputs: &ANY_OUTPUT,
    },
];
const PARQUETINFO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = parquetinfo(filename)",
    inputs: &PARQUETINFO_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "T = table(variables...)",
    inputs: &TABLE_INPUTS_VALUES,
    outputs: &ANY_OUTPUT,
}];
const GROUPSUMMARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "G = groupsummary(T, groupvars, method, datavars)",
    inputs: &GROUPSUMMARY_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const GRPSTATS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tblstats = grpstats(tbl, groupvars)",
        inputs: &GRPSTATS_INPUTS,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "stats = grpstats(X, group)",
        inputs: &GRPSTATS_INPUTS,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[stats1, ..., statsN] = grpstats(X, group, whichstats)",
        inputs: &GRPSTATS_INPUTS,
        outputs: &ANY_OUTPUT,
    },
];
const HEIGHT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "n = height(T)",
    inputs: &TABLE_INPUT,
    outputs: &NUM_OUTPUT,
}];
const WIDTH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "n = width(T)",
    inputs: &TABLE_INPUT,
    outputs: &NUM_OUTPUT,
}];
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
const PREDICATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "TF = tabularPredicate(A)",
    inputs: &VALUE_INPUT,
    outputs: &PREDICATE_OUTPUT,
}];
const WRITE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "bytesWritten = writeTabular(T, filename, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &WRITE_OUTPUT,
}];
const OBJECT_SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = table.subsref(obj, kind, payload)",
    inputs: &OBJECT_INDEX_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const OBJECT_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = table.subsasgn(obj, kind, payload, rhs)",
    inputs: &OBJECT_ASSIGN_INPUTS,
    outputs: &ANY_OUTPUT,
}];

pub(super) const TABLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:table:InvalidArgument"),
    when: "Arguments or table metadata are invalid.",
    message: "table: invalid argument",
};
pub(super) const TABLE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_INDEX",
    identifier: Some("RunMat:table:InvalidIndex"),
    when: "Table indexing is invalid.",
    message: "table: invalid index",
};
pub(super) const TABLE_ERROR_INVALID_VARIABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_VARIABLE",
    identifier: Some("RunMat:table:InvalidVariable"),
    when: "A table variable name or value is invalid.",
    message: "table: invalid variable",
};
pub(super) const TABLE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READTABLE.IO",
    identifier: Some("RunMat:readtable:IOError"),
    when: "readtable cannot open or read the requested file.",
    message: "readtable: file read failed",
};
pub(super) const TABLE_ERROR_UNSUPPORTED_FILE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READTABLE.UNSUPPORTED_FILE",
    identifier: Some("RunMat:readtable:UnsupportedFileType"),
    when: "readtable receives a file type outside the text or spreadsheet import backends.",
    message: "readtable: unsupported file type",
};
const TABLE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_INDEX,
    TABLE_ERROR_INVALID_VARIABLE,
    TABLE_ERROR_IO,
    TABLE_ERROR_UNSUPPORTED_FILE,
];

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
pub const PARQUETREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARQUETREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const PARQUETINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARQUETINFO_SIGNATURES,
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
pub const GROUPSUMMARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GROUPSUMMARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const GRPSTATS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRPSTATS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const HEIGHT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HEIGHT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const WIDTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WIDTH_SIGNATURES,
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
pub const TABLE_PREDICATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PREDICATE_SIGNATURES,
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
pub const TABLE_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OBJECT_SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &TABLE_ERRORS,
};
pub const TABLE_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OBJECT_SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
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
    E: std::error::Error + Send + Sync + 'static,
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
