//! MATLAB table, timetable, categorical, and tabular workflow builtins.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use calamine::{open_workbook_auto_from_rs, Data as SpreadsheetData, Reader as SpreadsheetReader};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use encoding_rs::{Encoding, UTF_8};
use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ClassDef, ComplexTensor, LogicalArray, MethodDef, NumericDType,
    ObjectInstance, PropertyDef, StringArray, StructValue, Tensor, Value,
};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError, OBJECT_INDEX_BRACE,
    OBJECT_INDEX_MEMBER, OBJECT_INDEX_PAREN, OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD,
};

mod containers;
mod display;
mod import;
mod object;

use containers::*;
use display::{categorical_label_at, format_key_number};
pub use display::{table_display_text, table_summary_text};
use import::*;
use object::*;
pub use object::{
    is_table_value, is_tabular_object, sortrows_table, table_from_columns, table_height,
    table_variable_names_from_object, table_variables, table_width,
};
pub(crate) use object::{
    select_rows, selected_row_names, table_from_columns_like, value_row_count,
};
pub const TABLE_CLASS: &str = "table";
pub const TIMETABLE_CLASS: &str = "timetable";
const CATEGORICAL_CLASS: &str = "categorical";
const DICTIONARY_CLASS: &str = "dictionary";
const TIMERANGE_CLASS: &str = "timerange";
const VARTYPE_CLASS: &str = "vartype";
const ROWFILTER_CLASS: &str = "rowfilter";
const ARRAY_DATASTORE_CLASS: &str = "arrayDatastore";
const PARQUET_DATASTORE_CLASS: &str = "parquetDatastore";
const UITABLE_CLASS: &str = "uitable";
const TABLE_VARIABLES_FIELD: &str = "__table_variables";
const TABLE_PROPERTIES_FIELD: &str = "__table_properties";
const PROPERTIES_MEMBER: &str = "Properties";
const ROW_TIMES: &str = "RowTimes";
const VARIABLE_NAMES: &str = "VariableNames";
const ROW_NAMES: &str = "RowNames";
const DIMENSION_NAMES: &str = "DimensionNames";
const VARIABLE_UNITS: &str = "VariableUnits";
const VARIABLE_DESCRIPTIONS: &str = "VariableDescriptions";
const DESCRIPTION: &str = "Description";
const USER_DATA: &str = "UserData";
const DEFAULT_ROW_DIM_NAME: &str = "Rows";
const DEFAULT_VARIABLE_DIM_NAME: &str = "Variables";

thread_local! {
    static TABLE_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

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

const TABLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:table:InvalidArgument"),
    when: "Arguments or table metadata are invalid.",
    message: "table: invalid argument",
};
const TABLE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_INDEX",
    identifier: Some("RunMat:table:InvalidIndex"),
    when: "Table indexing is invalid.",
    message: "table: invalid index",
};
const TABLE_ERROR_INVALID_VARIABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_VARIABLE",
    identifier: Some("RunMat:table:InvalidVariable"),
    when: "A table variable name or value is invalid.",
    message: "table: invalid variable",
};
const TABLE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READTABLE.IO",
    identifier: Some("RunMat:readtable:IOError"),
    when: "readtable cannot open or read the requested file.",
    message: "readtable: file read failed",
};
const TABLE_ERROR_UNSUPPORTED_FILE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
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

fn table_error(error: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(TABLE_CLASS);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn table_error_with_source<E>(
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

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_ARGUMENT, message)
}

fn invalid_index(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_INDEX, message)
}

fn invalid_variable(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_VARIABLE, message)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
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

pub fn ensure_table_class_registered() {
    TABLE_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        register_tabular_class(TABLE_CLASS);
        register_tabular_class(TIMETABLE_CLASS);
        register_plain_object_class(CATEGORICAL_CLASS, &["Codes", "Categories", "Ordinal"]);
        register_dictionary_class();
        register_plain_object_class(TIMERANGE_CLASS, &["Start", "End", "Inclusivity"]);
        register_plain_object_class(VARTYPE_CLASS, &["Type"]);
        register_plain_object_class(ROWFILTER_CLASS, &["Variables", "Predicate"]);
        register_plain_object_class(ARRAY_DATASTORE_CLASS, &["Data", "ReadSize"]);
        register_plain_object_class(PARQUET_DATASTORE_CLASS, &["Files"]);
        register_plain_object_class(UITABLE_CLASS, &["Data", "ColumnName", "RowName"]);
        registered.set(true);
    });
}

fn register_tabular_class(name: &str) {
    let mut properties = HashMap::new();
    properties.insert(
        PROPERTIES_MEMBER.to_string(),
        PropertyDef {
            name: PROPERTIES_MEMBER.to_string(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Struct(default_properties_for_class(
                name,
                Vec::new(),
                None,
            ))),
        },
    );

    let mut methods = HashMap::new();
    for method_name in [OBJECT_SUBSREF_METHOD, OBJECT_SUBSASGN_METHOD] {
        methods.insert(
            method_name.to_string(),
            MethodDef {
                name: method_name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: format!("{TABLE_CLASS}.{method_name}"),
                implicit_class_argument: None,
            },
        );
    }

    runmat_builtins::register_class(ClassDef {
        name: name.to_string(),
        parent: None,
        properties,
        methods,
    });
}

fn register_plain_object_class(name: &str, property_names: &[&str]) {
    let mut properties = HashMap::new();
    for property_name in property_names {
        properties.insert(
            (*property_name).to_string(),
            PropertyDef {
                name: (*property_name).to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
    }
    runmat_builtins::register_class(ClassDef {
        name: name.to_string(),
        parent: None,
        properties,
        methods: HashMap::new(),
    });
}

fn register_dictionary_class() {
    let mut properties = HashMap::new();
    for property_name in ["Keys", "Values"] {
        properties.insert(
            property_name.to_string(),
            PropertyDef {
                name: property_name.to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
    }
    let mut methods = HashMap::new();
    for method_name in [OBJECT_SUBSREF_METHOD, OBJECT_SUBSASGN_METHOD] {
        methods.insert(
            method_name.to_string(),
            MethodDef {
                name: method_name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: format!("{DICTIONARY_CLASS}.{method_name}"),
                implicit_class_argument: None,
            },
        );
    }
    runmat_builtins::register_class(ClassDef {
        name: DICTIONARY_CLASS.to_string(),
        parent: None,
        properties,
        methods,
    });
}

#[runtime_builtin(
    name = "table",
    category = "table",
    summary = "Create a table from named column variables.",
    keywords = "table,VariableNames,RowNames,Properties",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::TABLE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let gathered = gather_values(&args).await?;
    let (variables, options) = split_table_constructor_args(gathered)?;
    let names = if let Some(names) = options.variable_names {
        names
    } else {
        generated_variable_names(variables.len())
    };
    table_from_columns_with_properties(names, variables, options.row_names)
}

#[runtime_builtin(
    name = "readtable",
    category = "io/tabular",
    summary = "Import tabular text or spreadsheet data into a table.",
    keywords = "readtable,table,csv,tsv,xlsx,xls,ods,spreadsheet,VariableNames,RowNames,Sheet,Range",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::READTABLE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn readtable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let args = gather_values(&rest).await?;
    let options = ReadTableOptions::parse(&args)?;
    let resolved = resolve_path(&path_value)?;
    read_table_from_file(&resolved, &options).await
}

#[runtime_builtin(
    name = "spreadsheetImportOptions",
    category = "io/tabular",
    summary = "Create spreadsheet import options for readtable.",
    keywords = "spreadsheetImportOptions,readtable,spreadsheet,xlsx,xls,DataRange,VariableTypes,VariableNames,NumVariables",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn spreadsheet_import_options_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_values(&args).await?;
    spreadsheet_import_options(gathered)
}

#[runtime_builtin(
    name = "detectImportOptions",
    category = "io/tabular",
    summary = "Inspect a text or spreadsheet file and create import options.",
    keywords = "detectImportOptions,readtable,readmatrix,csv,tsv,xlsx,Delimiter,VariableTypes,VariableNames",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::DETECT_IMPORT_OPTIONS_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn detect_import_options_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let args = gather_values(&rest).await?;
    let options = ReadTableOptions::parse(&args)?;
    let resolved = resolve_path(&path_value)?;
    detect_import_options_from_file(&resolved, &options).await
}

#[runtime_builtin(
    name = "height",
    category = "table",
    summary = "Return the number of rows in a table.",
    keywords = "height,table,rows",
    descriptor(crate::builtins::table::HEIGHT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn height_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    if let Some(object) = table_object(&host) {
        return Ok(Value::Num(table_height(object)? as f64));
    }
    value_row_count(&host).map(|n| Value::Num(n as f64))
}

#[runtime_builtin(
    name = "width",
    category = "table",
    summary = "Return the number of variables in a table.",
    keywords = "width,table,variables",
    descriptor(crate::builtins::table::WIDTH_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn width_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    if let Some(object) = table_object(&host) {
        return Ok(Value::Num(table_width(object)? as f64));
    }
    match host {
        Value::Tensor(t) => Ok(Value::Num(t.cols() as f64)),
        Value::ComplexTensor(t) => Ok(Value::Num(t.cols as f64)),
        Value::StringArray(sa) => Ok(Value::Num(sa.cols() as f64)),
        Value::LogicalArray(la) => Ok(Value::Num(la.shape.get(1).copied().unwrap_or(1) as f64)),
        Value::Cell(ca) => Ok(Value::Num(ca.cols as f64)),
        Value::CharArray(ca) => Ok(Value::Num(ca.cols as f64)),
        _ => Ok(Value::Num(1.0)),
    }
}

#[runtime_builtin(
    name = "istable",
    category = "table",
    summary = "Return true for table arrays.",
    keywords = "istable,table,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn istable_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    Ok(Value::Bool(matches!(
        host,
        Value::Object(ref object) if object.is_class(TABLE_CLASS)
    )))
}

#[runtime_builtin(
    name = "istimetable",
    category = "table",
    summary = "Return true for timetable arrays.",
    keywords = "istimetable,timetable,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn istimetable_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    Ok(Value::Bool(matches!(
        host,
        Value::Object(ref object) if object.is_class(TIMETABLE_CLASS)
    )))
}

#[runtime_builtin(
    name = "iscategorical",
    category = "table",
    summary = "Return true for categorical arrays.",
    keywords = "iscategorical,categorical,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn iscategorical_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    Ok(Value::Bool(matches!(
        host,
        Value::Object(ref object) if object.is_class(CATEGORICAL_CLASS)
    )))
}

#[runtime_builtin(
    name = "array2table",
    category = "table",
    summary = "Convert an array into a table.",
    keywords = "array2table,table,VariableNames,RowNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn array2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_table_options(&rest, "array2table")?;
    let columns = split_value_columns(value)?;
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    table_from_columns_with_properties(names, columns, options.row_names)
}

#[runtime_builtin(
    name = "cell2table",
    category = "table",
    summary = "Convert a cell array into a table.",
    keywords = "cell2table,table,cell,VariableNames,RowNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn cell2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_table_options(&rest, "cell2table")?;
    let Value::Cell(cell) = value else {
        return Err(invalid_argument("cell2table: expected cell array input"));
    };
    let mut columns = Vec::with_capacity(cell.cols);
    for col in 0..cell.cols {
        let mut data = Vec::with_capacity(cell.rows);
        for row in 0..cell.rows {
            data.push(cell.get(row, col).map_err(invalid_index)?);
        }
        columns
            .push(Value::Cell(CellArray::new(data, cell.rows, 1).map_err(
                |err| invalid_variable(format!("cell2table: {err}")),
            )?));
    }
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    table_from_columns_with_properties(names, columns, options.row_names)
}

#[runtime_builtin(
    name = "struct2table",
    category = "table",
    summary = "Convert a scalar struct into a table.",
    keywords = "struct2table,table,struct,AsArray,RowNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn struct2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_struct2table_options(&rest)?;
    match value {
        Value::Struct(st) => {
            let mut names = Vec::with_capacity(st.fields.len());
            let mut columns = Vec::with_capacity(st.fields.len());
            for (name, value) in st.fields {
                names.push(name);
                if options.as_array && value_row_count(&value)? != 1 {
                    columns.push(Value::Cell(
                        CellArray::new(vec![value], 1, 1).map_err(invalid_variable)?,
                    ));
                } else {
                    columns.push(value);
                }
            }
            let names = options.table.variable_names.unwrap_or(names);
            table_from_columns_with_properties(names, columns, options.table.row_names)
        }
        Value::Cell(cell)
            if cell
                .data
                .iter()
                .all(|value| matches!(value, Value::Struct(_))) =>
        {
            let rows = cell.data.len();
            let first = cell.data.iter().find_map(|value| match value {
                Value::Struct(st) => Some(st),
                _ => None,
            });
            let field_names = first
                .map(|st| st.fields.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            let mut columns = Vec::with_capacity(field_names.len());
            for name in &field_names {
                let mut values = Vec::with_capacity(rows);
                for value in &cell.data {
                    let Value::Struct(st) = value else {
                        unreachable!("checked above")
                    };
                    values.push(st.fields.get(name).cloned().unwrap_or(Value::Num(f64::NAN)));
                }
                columns.push(Value::Cell(
                    CellArray::new(values, rows, 1).map_err(invalid_variable)?,
                ));
            }
            let names = options.table.variable_names.unwrap_or(field_names);
            table_from_columns_with_properties(names, columns, options.table.row_names)
        }
        other => Err(invalid_argument(format!(
            "struct2table: expected struct or struct array, got {other:?}"
        ))),
    }
}

#[runtime_builtin(
    name = "table2struct",
    category = "table",
    summary = "Convert a table into row structs or a scalar struct of variables.",
    keywords = "table2struct,table,struct,ToScalar",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table2struct_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let to_scalar = parse_table2struct_to_scalar(&rest)?;
    let object = into_table_object(host, "table2struct")?;
    if to_scalar {
        return Ok(Value::Struct(table_variables(&object)?));
    }
    let height = table_height(&object)?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let mut rows = Vec::with_capacity(height);
    for row in 0..height {
        let mut st = StructValue::new();
        for name in &names {
            let value = variables.fields.get(name).ok_or_else(|| {
                invalid_variable(format!("table2struct: missing variable '{name}'"))
            })?;
            st.insert(name.clone(), row_value(value, row)?);
        }
        rows.push(Value::Struct(st));
    }
    CellArray::new(rows, height, 1)
        .map(Value::Cell)
        .map_err(invalid_variable)
}

#[runtime_builtin(
    name = "table2array",
    category = "table",
    summary = "Convert table variables into a homogeneous array when possible.",
    keywords = "table2array,table,array",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table2array_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let object = into_table_object(host, "table2array")?;
    table_brace_get(&object, &colon_colon_payload())
}

#[runtime_builtin(
    name = "table2cell",
    category = "table",
    summary = "Convert table variables into a cell array.",
    keywords = "table2cell,table,cell",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table2cell_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let object = into_table_object(host, "table2cell")?;
    table_to_cell_array(&object)
}

#[runtime_builtin(
    name = "head",
    category = "table",
    summary = "Return the first rows of a table, timetable, or array.",
    keywords = "head,table,timetable,preview,rows",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn head_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let n = rest
        .first()
        .map(|value| nonnegative_usize(value, "head row count"))
        .transpose()?
        .unwrap_or(8);
    let rows = value_row_count(&value)?;
    let selected = (0..rows.min(n)).collect::<Vec<_>>();
    if let Some(object) = table_object(&value) {
        let names = table_variable_names_from_object(object)?;
        let variables = table_variables(object)?;
        let mut columns = Vec::with_capacity(names.len());
        for name in &names {
            columns.push(select_rows(
                variables
                    .fields
                    .get(name)
                    .ok_or_else(|| invalid_variable(format!("head: missing variable '{name}'")))?,
                &selected,
            )?);
        }
        return subset_tabular_object(object, names, columns, &selected);
    }
    select_rows(&value, &selected)
}

#[runtime_builtin(
    name = "timetable",
    category = "table",
    summary = "Create a timetable from row times and variables.",
    keywords = "timetable,table,RowTimes,TimeStep,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn timetable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let (row_times, variables, options) = split_timetable_constructor_args(args)?;
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(variables.len()));
    let mut value =
        table_from_columns_with_class(TIMETABLE_CLASS, names, variables, options.row_names)?;
    if let Value::Object(object) = &mut value {
        set_timetable_row_times(object, row_times)?;
    }
    Ok(value)
}

#[runtime_builtin(
    name = "array2timetable",
    category = "table",
    summary = "Convert an array into a timetable.",
    keywords = "array2timetable,timetable,RowTimes,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn array2timetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let (row_times, options) = parse_timetable_options(&rest, "array2timetable")?;
    let columns = split_value_columns(value)?;
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    let mut out =
        table_from_columns_with_class(TIMETABLE_CLASS, names, columns, options.row_names)?;
    if let Value::Object(object) = &mut out {
        set_timetable_row_times(object, row_times)?;
    }
    Ok(out)
}

#[runtime_builtin(
    name = "table2timetable",
    category = "table",
    summary = "Convert a table into a timetable.",
    keywords = "table2timetable,timetable,RowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table2timetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let (row_times, _options) = parse_timetable_options(&rest, "table2timetable")?;
    let object = into_table_object(host, "table2timetable")?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let row_names = selected_row_names(&object, &(0..table_height(&object)?).collect::<Vec<_>>())?;
    let (times, out_names) = if let Some(row_times) = row_times {
        (Some(row_times), names)
    } else if let Some(first) = names.first() {
        let first_value = variables.fields.get(first).cloned();
        if first_value
            .as_ref()
            .map(is_time_like_value)
            .unwrap_or(false)
        {
            (first_value, names[1..].to_vec())
        } else {
            (None, names)
        }
    } else {
        (None, names)
    };
    let mut out_columns = Vec::with_capacity(out_names.len());
    for name in &out_names {
        out_columns.push(variables.fields.get(name).cloned().ok_or_else(|| {
            invalid_variable(format!("table2timetable: missing variable '{name}'"))
        })?);
    }
    let mut out =
        table_from_columns_with_class(TIMETABLE_CLASS, out_names, out_columns, row_names)?;
    if let Value::Object(object) = &mut out {
        set_timetable_row_times(object, times)?;
    }
    Ok(out)
}

#[runtime_builtin(
    name = "timetable2table",
    category = "table",
    summary = "Convert a timetable into a table.",
    keywords = "timetable2table,timetable,table,ConvertRowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn timetable2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let convert_row_times = parse_bool_option(&rest, "ConvertRowTimes", false, "timetable2table")?;
    let object = into_timetable_object(host, "timetable2table")?;
    let mut names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let mut columns = Vec::with_capacity(names.len() + usize::from(convert_row_times));
    if convert_row_times {
        if let Some(row_times) = timetable_row_times(&object)? {
            columns.push(row_times);
            names.insert(0, "Time".to_string());
        }
    }
    for name in table_variable_names_from_object(&object)? {
        columns.push(variables.fields.get(&name).cloned().ok_or_else(|| {
            invalid_variable(format!("timetable2table: missing variable '{name}'"))
        })?);
    }
    let row_names = selected_row_names(&object, &(0..table_height(&object)?).collect::<Vec<_>>())?;
    table_from_columns_with_properties(names, columns, row_names)
}

#[runtime_builtin(
    name = "readtimetable",
    category = "io/tabular",
    summary = "Read tabular data into a timetable.",
    keywords = "readtimetable,timetable,readtable,RowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn readtimetable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let (readtable_args, timetable_args) = split_readtimetable_options(&rest)?;
    let table = readtable_builtin(path, readtable_args).await?;
    table2timetable_builtin(table, timetable_args).await
}

#[runtime_builtin(
    name = "writetable",
    category = "io/tabular",
    summary = "Write a table to a delimited text file.",
    keywords = "writetable,table,csv,delimited text,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_WRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn writetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    write_tabular_file(value, rest, false).await
}

#[runtime_builtin(
    name = "writetimetable",
    category = "io/tabular",
    summary = "Write a timetable to a delimited text file.",
    keywords = "writetimetable,timetable,csv,delimited text,RowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_WRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn writetimetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    write_tabular_file(value, rest, true).await
}

#[runtime_builtin(
    name = "readcell",
    category = "io/tabular",
    summary = "Read text or spreadsheet data into a cell array.",
    keywords = "readcell,cell,readtable,csv,spreadsheet",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn readcell_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let path = resolve_path(&path)?;
    let options = ReadTableOptions::parse(&rest)?;
    read_cell_from_file(&path, &options).await
}

#[runtime_builtin(
    name = "categorical",
    category = "table",
    summary = "Create a categorical array.",
    keywords = "categorical,categories,ordinal,table",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn categorical_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    categorical_from_args(args)
}

#[runtime_builtin(
    name = "dictionary",
    category = "table",
    summary = "Create a key-value dictionary object.",
    keywords = "dictionary,containers.Map,key,value,map",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn dictionary_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    dictionary_from_args(args)
}

#[runtime_builtin(
    name = "timerange",
    category = "table",
    summary = "Create a timetable row-time range selector.",
    keywords = "timerange,timetable,row times",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn timerange_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args.len() > 3 {
        return Err(invalid_argument(
            "timerange: expected start, end, and optional inclusivity",
        ));
    }
    let gathered = gather_values(&args).await?;
    let mut object = ObjectInstance::new(TIMERANGE_CLASS.to_string());
    object.properties.insert(
        "Start".to_string(),
        gathered
            .first()
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    );
    object.properties.insert(
        "End".to_string(),
        gathered
            .get(1)
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    );
    object.properties.insert(
        "Inclusivity".to_string(),
        gathered
            .get(2)
            .cloned()
            .unwrap_or_else(|| Value::from("closed")),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "vartype",
    category = "table",
    summary = "Create a table variable type selector.",
    keywords = "vartype,table,selector,variable type",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn vartype_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let mut object = ObjectInstance::new(VARTYPE_CLASS.to_string());
    object.properties.insert("Type".to_string(), value);
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "rowfilter",
    category = "table",
    summary = "Create a table row filter descriptor.",
    keywords = "rowfilter,table,rows,filter",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn rowfilter_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(ROWFILTER_CLASS.to_string());
    object.properties.insert(
        "Variables".to_string(),
        args.first()
            .cloned()
            .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap())),
    );
    object.properties.insert(
        "Predicate".to_string(),
        args.get(1)
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "pivot",
    category = "table",
    summary = "Pivot or summarize table data by grouping variables.",
    keywords = "pivot,table,reshape,groupsummary",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn pivot_builtin(
    table: Value,
    rowvars: Value,
    colvars: Value,
    datavar: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let table = gather_if_needed_async(&table)
        .await
        .map_err(map_control_flow)?;
    let rowvars = gather_if_needed_async(&rowvars)
        .await
        .map_err(map_control_flow)?;
    let colvars = gather_if_needed_async(&colvars)
        .await
        .map_err(map_control_flow)?;
    let datavar = gather_if_needed_async(&datavar)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let method = parse_named_text_option(&rest, "Method", "sum", "pivot")?;
    pivot_impl(table, rowvars, colvars, datavar, &method)
}

#[runtime_builtin(
    name = "arrayDatastore",
    category = "io/tabular",
    summary = "Create an array datastore descriptor.",
    keywords = "arrayDatastore,datastore,array,data",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn array_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(ARRAY_DATASTORE_CLASS.to_string());
    object.properties.insert(
        "Data".to_string(),
        args.first()
            .cloned()
            .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap())),
    );
    object
        .properties
        .insert("ReadSize".to_string(), Value::Num(1.0));
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "parquetDatastore",
    category = "io/tabular",
    summary = "Create a parquet datastore descriptor.",
    keywords = "parquetDatastore,datastore,parquet,table",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn parquet_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(PARQUET_DATASTORE_CLASS.to_string());
    object.properties.insert(
        "Files".to_string(),
        args.first().cloned().unwrap_or_else(|| {
            Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap())
        }),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "uitable",
    category = "table",
    summary = "Create a table UI component descriptor.",
    keywords = "uitable,ui,table,Data",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn uitable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(UITABLE_CLASS.to_string());
    let data = parse_named_option(&args, "Data")
        .cloned()
        .or_else(|| args.first().cloned())
        .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()));
    object.properties.insert("Data".to_string(), data);
    object.properties.insert(
        "ColumnName".to_string(),
        Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
    );
    object.properties.insert(
        "RowName".to_string(),
        Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "groupsummary",
    category = "table",
    summary = "Group table rows and compute summary statistics for data variables.",
    keywords = "groupsummary,group,table,mean,sum,count,median,min,max",
    accel = "cpu",
    descriptor(crate::builtins::table::GROUPSUMMARY_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn groupsummary_builtin(
    table: Value,
    groupvars: Value,
    method: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let table = gather_if_needed_async(&table)
        .await
        .map_err(map_control_flow)?;
    let groupvars = gather_if_needed_async(&groupvars)
        .await
        .map_err(map_control_flow)?;
    let method = gather_if_needed_async(&method)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    groupsummary_impl(table, groupvars, method, rest)
}

#[runtime_builtin(
    name = "table.subsref",
    descriptor(crate::builtins::table::TABLE_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table_subsref(obj: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
    let object = into_table_object(obj, "table.subsref")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => table_member_get(&object, &payload),
        OBJECT_INDEX_PAREN => table_paren_get(&object, &payload),
        OBJECT_INDEX_BRACE => table_brace_get(&object, &payload),
        other => Err(invalid_index(format!(
            "table.subsref: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runtime_builtin(
    name = "table.subsasgn",
    descriptor(crate::builtins::table::TABLE_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let mut object = into_table_object(obj, "table.subsasgn")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "table member")?;
            table_member_set(&mut object, &field, rhs)?;
            Ok(Value::Object(object))
        }
        OBJECT_INDEX_PAREN => table_paren_assign(object, &payload, rhs),
        OBJECT_INDEX_BRACE => table_brace_assign(object, &payload, rhs),
        other => Err(invalid_index(format!(
            "table.subsasgn: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runtime_builtin(
    name = "dictionary.subsref",
    descriptor(crate::builtins::table::TABLE_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn dictionary_subsref(obj: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
    let object = into_dictionary_object(obj, "dictionary.subsref")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "dictionary member")?;
            object
                .properties
                .get(&field)
                .cloned()
                .ok_or_else(|| invalid_variable(format!("dictionary: unknown property '{field}'")))
        }
        OBJECT_INDEX_PAREN | OBJECT_INDEX_BRACE => dictionary_lookup(&object, &payload),
        other => Err(invalid_index(format!(
            "dictionary.subsref: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runtime_builtin(
    name = "dictionary.subsasgn",
    descriptor(crate::builtins::table::TABLE_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn dictionary_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let mut object = into_dictionary_object(obj, "dictionary.subsasgn")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "dictionary member")?;
            if field != "Keys" && field != "Values" {
                return Err(invalid_variable(format!(
                    "dictionary: unknown property '{field}'"
                )));
            }
            object.properties.insert(field, rhs);
            Ok(Value::Object(object))
        }
        OBJECT_INDEX_PAREN | OBJECT_INDEX_BRACE => dictionary_assign(object, &payload, rhs),
        other => Err(invalid_index(format!(
            "dictionary.subsasgn: unsupported indexing kind '{other}'"
        ))),
    }
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(invalid_argument(format!(
            "table: {context} must be a string scalar or character vector"
        ))),
    }
}

fn bool_scalar(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Int(value) => Ok(value.to_i64() != 0),
        Value::Num(value) if value.is_finite() => Ok(*value != 0.0),
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            let text = scalar_text(value, context)?;
            match text.to_ascii_lowercase().as_str() {
                "true" | "on" | "yes" => Ok(true),
                "false" | "off" | "no" => Ok(false),
                _ => Err(invalid_argument(format!(
                    "table: {context} must be logical"
                ))),
            }
        }
        _ => Err(invalid_argument(format!(
            "table: {context} must be logical"
        ))),
    }
}

fn nonnegative_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(value) if value.to_i64() >= 0 => Ok(value.to_i64() as usize),
        Value::Num(value)
            if value.is_finite()
                && *value >= 0.0
                && (value.round() - value).abs() <= f64::EPSILON =>
        {
            Ok(value.round() as usize)
        }
        _ => Err(invalid_argument(format!(
            "table: {context} must be a non-negative integer"
        ))),
    }
}

fn positive_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    let value = nonnegative_usize(value, context)?;
    if value == 0 {
        return Err(invalid_argument(format!(
            "table: {context} must be a positive integer"
        )));
    }
    Ok(value)
}

fn option_value_is_empty(value: &Value) -> bool {
    match value {
        Value::String(text) => text.trim().is_empty(),
        Value::CharArray(array) => {
            array.data.is_empty()
                || (array.rows == 1 && array.data.iter().all(|ch| ch.is_whitespace()))
        }
        Value::StringArray(array) => {
            array.data.is_empty() || (array.data.len() == 1 && array.data[0].trim().is_empty())
        }
        Value::Cell(cell) => {
            cell.data.is_empty() || cell.data.iter().all(|handle| option_value_is_empty(handle))
        }
        _ => false,
    }
}

fn string_list(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(ca) if ca.rows == 1 => Ok(vec![ca.data.iter().collect()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for handle in &cell.data {
                let value = handle;
                out.extend(string_list(value)?);
            }
            Ok(out)
        }
        _ => Err(invalid_argument(
            "table: expected string, string array, character vector, or cellstr",
        )),
    }
}

fn optional_raw_variable_name_list(value: &Value) -> BuiltinResult<Option<Vec<String>>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        raw_variable_name_list(value).map(Some)
    }
}

fn raw_variable_name_list(value: &Value) -> BuiltinResult<Vec<String>> {
    let names = string_list(value)?;
    if names.is_empty() {
        return Err(invalid_variable("table: variable names must not be empty"));
    }
    Ok(names)
}

fn variable_name_list(value: &Value) -> BuiltinResult<Vec<String>> {
    raw_variable_name_list(value).map(make_unique_variable_names)
}

fn optional_variable_type_list(value: &Value) -> BuiltinResult<Option<Vec<ImportVariableType>>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        variable_type_list(value).map(Some)
    }
}

fn variable_type_list(value: &Value) -> BuiltinResult<Vec<ImportVariableType>> {
    string_list(value)?
        .iter()
        .map(|raw| ImportVariableType::parse(raw))
        .collect()
}

fn variable_type_names(value: &Value) -> BuiltinResult<Vec<String>> {
    string_list(value)?
        .iter()
        .map(|raw| ImportVariableType::canonical_label(raw))
        .collect()
}

fn optional_range_spec(value: &Value) -> BuiltinResult<Option<RangeSpec>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        RangeSpec::parse(value).map(Some)
    }
}

fn optional_sheet_selector(value: &Value) -> BuiltinResult<Option<SheetSelector>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        SheetSelector::parse(value).map(Some)
    }
}

fn generated_variable_names(count: usize) -> Vec<String> {
    (1..=count).map(|idx| format!("Var{idx}")).collect()
}

fn make_unique_variable_names(names: Vec<String>) -> Vec<String> {
    make_unique_names(
        names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| make_valid_variable_name(&name, idx + 1))
            .collect(),
    )
}

fn make_unique_names(names: Vec<String>) -> Vec<String> {
    let mut used = HashSet::new();
    let mut out = Vec::with_capacity(names.len());
    for (idx, name) in names.into_iter().enumerate() {
        let base = if name.trim().is_empty() {
            format!("Var{}", idx + 1)
        } else {
            name.trim().to_string()
        };
        let mut candidate = base.clone();
        let mut suffix = 1usize;
        while used.contains(&candidate.to_ascii_lowercase()) {
            suffix += 1;
            candidate = format!("{base}_{suffix}");
        }
        used.insert(candidate.to_ascii_lowercase());
        out.push(candidate);
    }
    out
}

fn make_valid_variable_name(raw: &str, fallback_index: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in raw.trim().chars().enumerate() {
        if (idx == 0 && (ch.is_ascii_alphabetic() || ch == '_'))
            || (idx > 0 && (ch.is_ascii_alphanumeric() || ch == '_'))
        {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() || !out.chars().next().unwrap().is_ascii_alphabetic() {
        format!("Var{fallback_index}")
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use async_trait::async_trait;
    use futures::executor::block_on;
    #[cfg(not(target_arch = "wasm32"))]
    use runmat_filesystem::{
        DirEntry, FileHandle, FsMetadata, FsProvider, NativeFsProvider, OpenFlags,
        SandboxFsProvider,
    };
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    #[cfg(not(target_arch = "wasm32"))]
    use std::io;
    use std::io::Write;

    #[cfg(not(target_arch = "wasm32"))]
    struct PrefixSandboxProvider {
        prefix: &'static str,
        sandbox: SandboxFsProvider,
        native: NativeFsProvider,
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl PrefixSandboxProvider {
        fn is_virtual(&self, path: &Path) -> bool {
            path.to_string_lossy().starts_with(self.prefix)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[async_trait(?Send)]
    impl FsProvider for PrefixSandboxProvider {
        fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
            if self.is_virtual(path) {
                self.sandbox.open(path, flags)
            } else {
                self.native.open(path, flags)
            }
        }

        async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
            if self.is_virtual(path) {
                self.sandbox.read(path).await
            } else {
                self.native.read(path).await
            }
        }

        async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
            if self.is_virtual(path) {
                self.sandbox.write(path, data).await
            } else {
                self.native.write(path, data).await
            }
        }

        async fn remove_file(&self, path: &Path) -> io::Result<()> {
            if self.is_virtual(path) {
                self.sandbox.remove_file(path).await
            } else {
                self.native.remove_file(path).await
            }
        }

        async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
            if self.is_virtual(path) {
                self.sandbox.metadata(path).await
            } else {
                self.native.metadata(path).await
            }
        }

        async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
            if self.is_virtual(path) {
                self.sandbox.symlink_metadata(path).await
            } else {
                self.native.symlink_metadata(path).await
            }
        }

        async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
            if self.is_virtual(path) {
                self.sandbox.read_dir(path).await
            } else {
                self.native.read_dir(path).await
            }
        }

        async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
            if self.is_virtual(path) {
                self.sandbox.canonicalize(path).await
            } else {
                self.native.canonicalize(path).await
            }
        }

        async fn create_dir(&self, path: &Path) -> io::Result<()> {
            if self.is_virtual(path) {
                self.sandbox.create_dir(path).await
            } else {
                self.native.create_dir(path).await
            }
        }

        async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
            if self.is_virtual(path) {
                self.sandbox.create_dir_all(path).await
            } else {
                self.native.create_dir_all(path).await
            }
        }

        async fn remove_dir(&self, path: &Path) -> io::Result<()> {
            if self.is_virtual(path) {
                self.sandbox.remove_dir(path).await
            } else {
                self.native.remove_dir(path).await
            }
        }

        async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
            if self.is_virtual(path) {
                self.sandbox.remove_dir_all(path).await
            } else {
                self.native.remove_dir_all(path).await
            }
        }

        async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
            match (self.is_virtual(from), self.is_virtual(to)) {
                (true, true) => self.sandbox.rename(from, to).await,
                (false, false) => self.native.rename(from, to).await,
                _ => Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "cross-provider rename is unsupported in test provider",
                )),
            }
        }

        async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
            if self.is_virtual(path) {
                self.sandbox.set_readonly(path, readonly).await
            } else {
                self.native.set_readonly(path, readonly).await
            }
        }
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_{prefix}_{}_{}",
            std::process::id(),
            unix_timestamp_ms()
        ));
        path
    }

    fn read_table(path: &Path, args: Vec<Value>) -> Value {
        block_on(readtable_builtin(
            Value::from(path.to_string_lossy().to_string()),
            args,
        ))
        .expect("readtable")
    }

    fn read_table_err(path: &Path, args: Vec<Value>) -> RuntimeError {
        block_on(readtable_builtin(
            Value::from(path.to_string_lossy().to_string()),
            args,
        ))
        .expect_err("expected readtable failure")
    }

    fn spreadsheet_options(args: Vec<Value>) -> StructValue {
        match block_on(spreadsheet_import_options_builtin(args)).expect("spreadsheetImportOptions")
        {
            Value::Struct(options) => options,
            other => panic!("expected struct options, got {other:?}"),
        }
    }

    fn detect_options(path: &Path, args: Vec<Value>) -> StructValue {
        match block_on(detect_import_options_builtin(
            Value::from(path.to_string_lossy().to_string()),
            args,
        ))
        .expect("detectImportOptions")
        {
            Value::Struct(options) => options,
            other => panic!("expected struct options, got {other:?}"),
        }
    }

    fn char_row(array: &CharArray, row: usize) -> String {
        let start = row * array.cols;
        array.data[start..start + array.cols].iter().collect()
    }

    fn object(value: Value) -> ObjectInstance {
        match value {
            Value::Object(object) => object,
            other => panic!("expected table object, got {other:?}"),
        }
    }

    #[test]
    fn readtable_imports_headered_numeric_and_text_columns() {
        let path = unique_path("readtable_basic");
        fs::write(&path, "Name,Score\nAda,10\nGrace,12\n").expect("write sample");
        let table = object(read_table(&path, Vec::new()));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["Name".to_string(), "Score".to_string()]
        );
        match table_member_get(&table, &Value::from("Score")).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![10.0, 12.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match table_member_get(&table, &Value::from("Name")).unwrap() {
            Value::StringArray(array) => {
                assert_eq!(array.data, vec!["Ada".to_string(), "Grace".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_auto_does_not_consume_headerless_numeric_rows() {
        let path = unique_path("readtable_headerless_numeric");
        fs::write(&path, "1,2\n3,4\n").expect("write sample");
        let table = object(read_table(&path, Vec::new()));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["Var1".to_string(), "Var2".to_string()]
        );
        match table_member_get(&table, &Value::from("Var1")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 3.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        match table_member_get(&table, &Value::from("Var2")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0, 4.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_rejects_unknown_and_invalid_options() {
        let path = unique_path("readtable_invalid_options");
        fs::write(&path, "A\n1\n").expect("write sample");
        let err = read_table_err(
            &path,
            vec![Value::from("DefinitelyNotAnOption"), Value::from(1.0)],
        );
        assert!(err.message().contains("unsupported option"));
        let err = read_table_err(
            &path,
            vec![Value::from("VariableNamingRule"), Value::from("mangle")],
        );
        assert!(err.message().contains("unsupported VariableNamingRule"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_handles_quoted_delimiters_and_newlines() {
        let path = unique_path("readtable_quoted_newlines");
        fs::write(
            &path,
            "Name,Note\nAda,\"hello, world\"\nGrace,\"line one\nline two\"\n",
        )
        .expect("write sample");
        let table = object(read_table(&path, Vec::new()));
        match table_member_get(&table, &Value::from("Note")).unwrap() {
            Value::StringArray(array) => assert_eq!(
                array.data,
                vec!["hello, world".to_string(), "line one\nline two".to_string()]
            ),
            other => panic!("expected string array, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_supports_explicit_names_and_missing_tokens() {
        let path = unique_path("readtable_options");
        fs::write(&path, "1,NA\n2,4\n").expect("write sample");
        let names =
            StringArray::new(vec!["A".to_string(), "B".to_string()], vec![1, 2]).expect("names");
        let table = object(read_table(
            &path,
            vec![
                Value::from("ReadVariableNames"),
                Value::Bool(false),
                Value::from("VariableNames"),
                Value::StringArray(names),
                Value::from("TreatAsMissing"),
                Value::from("NA"),
            ],
        ));
        match table_member_get(&table, &Value::from("B")).unwrap() {
            Value::Tensor(tensor) => {
                assert!(tensor.data[0].is_nan());
                assert_eq!(tensor.data[1], 4.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_preserves_variable_names_when_requested() {
        let path = unique_path("readtable_preserve_names");
        fs::write(&path, "daily revenue,total orders\n100,10\n").expect("write sample");
        let table = object(read_table(
            &path,
            vec![Value::from("VariableNamingRule"), Value::from("preserve")],
        ));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["daily revenue".to_string(), "total orders".to_string()]
        );
        let _ = fs::remove_file(&path);
    }

    fn write_zip_file(zip: &mut zip::ZipWriter<std::fs::File>, name: &str, contents: &str) {
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file(name, options).expect("start xlsx part");
        zip.write_all(contents.as_bytes()).expect("write xlsx part");
    }

    fn write_minimal_xlsx(path: &Path) {
        let file = std::fs::File::create(path).expect("create xlsx");
        let mut zip = zip::ZipWriter::new(file);
        write_zip_file(
            &mut zip,
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"#,
        );
        write_zip_file(
            &mut zip,
            "_rels/.rels",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#,
        );
        write_zip_file(
            &mut zip,
            "xl/workbook.xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Data" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"#,
        );
        write_zip_file(
            &mut zip,
            "xl/_rels/workbook.xml.rels",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#,
        );
        write_zip_file(
            &mut zip,
            "xl/styles.xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellXfs>
</styleSheet>"#,
        );
        write_zip_file(
            &mut zip,
            "xl/worksheets/sheet1.xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="inlineStr"><is><t>Date</t></is></c>
      <c r="B1" t="inlineStr"><is><t>Orders</t></is></c>
      <c r="C1" t="inlineStr"><is><t>Revenue</t></is></c>
    </row>
    <row r="2">
      <c r="A2" t="inlineStr"><is><t>2026-06-01</t></is></c>
      <c r="B2"><v>10</v></c>
      <c r="C2"><v>200</v></c>
    </row>
    <row r="3">
      <c r="A3" t="inlineStr"><is><t>2026-06-02</t></is></c>
      <c r="B3"><v>4</v></c>
      <c r="C3"><v>90</v></c>
    </row>
  </sheetData>
</worksheet>"#,
        );
        zip.finish().expect("finish xlsx");
    }

    #[test]
    fn readtable_imports_xlsx_sheet_and_range() {
        let path = unique_path("readtable_spreadsheet");
        let path = path.with_extension("xlsx");
        write_minimal_xlsx(&path);
        let table = object(read_table(
            &path,
            vec![
                Value::from("Sheet"),
                Value::from("Data"),
                Value::from("Range"),
                Value::from("A1:C3"),
            ],
        ));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec![
                "Date".to_string(),
                "Orders".to_string(),
                "Revenue".to_string()
            ]
        );
        match table_member_get(&table, &Value::from("Revenue")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![200.0, 90.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn spreadsheet_import_options_registers_public_descriptor() {
        assert!(runmat_builtins::builtin_function_by_name("spreadsheetImportOptions").is_some());
        let labels = SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"opts = spreadsheetImportOptions()"));
        assert!(labels.contains(&"opts = spreadsheetImportOptions(nameValuePairs...)"));
    }

    #[test]
    fn detect_import_options_registers_public_descriptor() {
        assert!(runmat_builtins::builtin_function_by_name("detectImportOptions").is_some());
        let labels = DETECT_IMPORT_OPTIONS_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"opts = detectImportOptions(filename)"));
        assert!(labels.contains(&"opts = detectImportOptions(filename, nameValuePairs...)"));
    }

    #[test]
    fn detect_import_options_infers_text_delimiter_names_and_types() {
        let path = unique_path("detect_import_options_text");
        fs::write(
            &path,
            "Name;Score;Flag;When\nAda;10;true;2026-06-01\nGrace;12;false;2026-06-02\n",
        )
        .expect("write sample");
        let options = detect_options(&path, Vec::new());
        assert_eq!(options.fields.get("FileType"), Some(&Value::from("text")));
        assert_eq!(options.fields.get("Delimiter"), Some(&Value::from(";")));
        assert_eq!(options.fields.get("NumHeaderLines"), Some(&Value::Num(1.0)));
        assert_eq!(
            options.fields.get("ReadVariableNames"),
            Some(&Value::Bool(false))
        );
        match options.fields.get("VariableNames").unwrap() {
            Value::StringArray(array) => assert_eq!(
                array.data,
                vec![
                    "Name".to_string(),
                    "Score".to_string(),
                    "Flag".to_string(),
                    "When".to_string()
                ]
            ),
            other => panic!("expected string array, got {other:?}"),
        }
        match options.fields.get("VariableTypes").unwrap() {
            Value::StringArray(array) => assert_eq!(
                array.data,
                vec![
                    "string".to_string(),
                    "double".to_string(),
                    "logical".to_string(),
                    "datetime".to_string()
                ]
            ),
            other => panic!("expected string array, got {other:?}"),
        }
        let table = object(read_table(&path, vec![Value::Struct(options)]));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec![
                "Name".to_string(),
                "Score".to_string(),
                "Flag".to_string(),
                "When".to_string()
            ]
        );
        match table_member_get(&table, &Value::from("Score")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![10.0, 12.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn detect_import_options_struct_can_drive_readmatrix() {
        let path = unique_path("detect_import_options_readmatrix");
        fs::write(&path, "A,B\n1,2\n3,4\n").expect("write sample");
        let options = detect_options(&path, Vec::new());
        let matrix = block_on(
            crate::builtins::io::tabular::readmatrix::readmatrix_builtin(
                Value::from(path.to_string_lossy().to_string()),
                vec![Value::Struct(options)],
            ),
        )
        .expect("readmatrix");
        match matrix {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert_eq!(tensor.data, vec![1.0, 3.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn detect_import_options_strips_bom_from_detected_names() {
        let path = unique_path("detect_import_options_bom");
        fs::write(&path, "\u{FEFF}A,B\n1,2\n3,4\n").expect("write sample");
        let options = detect_options(&path, Vec::new());
        match options.fields.get("VariableNames").unwrap() {
            Value::StringArray(array) => {
                assert_eq!(array.data, vec!["A".to_string(), "B".to_string()])
            }
            other => panic!("expected string array, got {other:?}"),
        }
        let table = object(read_table(&path, vec![Value::Struct(options)]));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["A".to_string(), "B".to_string()]
        );
        match table_member_get(&table, &Value::from("A")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 3.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn detect_import_options_preserves_partial_ranges_for_replay() {
        let path = unique_path("detect_import_options_partial_range");
        fs::write(&path, "ID,A,B,C\nr1,1,2,3\nr2,4,5,6\nr3,7,8,9\n").expect("write sample");

        let column_options = detect_options(&path, vec![Value::from("Range"), Value::from("C:D")]);
        assert_eq!(
            column_options.fields.get("Range"),
            Some(&Value::from("C2:D"))
        );
        let table = object(read_table(&path, vec![Value::Struct(column_options)]));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["B".to_string(), "C".to_string()]
        );
        match table_member_get(&table, &Value::from("B")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0, 5.0, 8.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        fs::write(&path, "11,12\n21,22\n31,32\n41,42\n").expect("write numeric sample");
        let row_options = detect_options(&path, vec![Value::from("Range"), Value::from("2:3")]);
        assert_eq!(row_options.fields.get("Range"), Some(&Value::from("2:3")));
        let table = object(read_table(&path, vec![Value::Struct(row_options)]));
        match table_member_get(&table, &Value::from("Var2")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![22.0, 32.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn detect_import_options_read_row_names_replays_through_readtable() {
        let path = unique_path("detect_import_options_row_names");
        fs::write(&path, "Row,Name,Score\nr1,Ada,10\nr2,Grace,12\n").expect("write sample");
        let options = detect_options(&path, vec![Value::from("ReadRowNames"), Value::Bool(true)]);
        assert_eq!(options.fields.get("NumVariables"), Some(&Value::Num(2.0)));
        match options.fields.get("VariableNames").unwrap() {
            Value::StringArray(array) => assert_eq!(
                array.data,
                vec!["Row".to_string(), "Name".to_string(), "Score".to_string()]
            ),
            other => panic!("expected string array, got {other:?}"),
        }
        let table = object(read_table(&path, vec![Value::Struct(options)]));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["Name".to_string(), "Score".to_string()]
        );
        let props = table_public_properties(&table).unwrap();
        match props.fields.get(ROW_NAMES).unwrap() {
            Value::StringArray(array) => {
                assert_eq!(array.data, vec!["r1".to_string(), "r2".to_string()])
            }
            other => panic!("expected row names, got {other:?}"),
        }
        match table_member_get(&table, &Value::from("Score")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![10.0, 12.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn detect_import_options_encoding_replays_through_readmatrix() {
        let path = unique_path("detect_import_options_encoding_readmatrix");
        fs::write(&path, b"Caf\xe9,Score\n1,2\n3,4\n").expect("write sample");
        let options = detect_options(
            &path,
            vec![Value::from("Encoding"), Value::from("windows-1252")],
        );
        let matrix = block_on(
            crate::builtins::io::tabular::readmatrix::readmatrix_builtin(
                Value::from(path.to_string_lossy().to_string()),
                vec![Value::Struct(options)],
            ),
        )
        .expect("readmatrix");
        match matrix {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert_eq!(tensor.data, vec![1.0, 3.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn detect_import_options_replays_through_filesystem_provider() {
        let root = unique_path("detect_import_options_provider_root");
        {
            let _provider_lock = runmat_filesystem::provider_override_lock();
            let provider = PrefixSandboxProvider {
                prefix: "/provider",
                sandbox: SandboxFsProvider::new(root.clone()).expect("sandbox provider"),
                native: NativeFsProvider,
            };
            let _provider_guard =
                runmat_filesystem::replace_provider(std::sync::Arc::new(provider));
            block_on(runmat_filesystem::write_async(
                "/provider.csv",
                b"Name,Score\nAda,10\nGrace,12\n",
            ))
            .expect("write provider sample");

            let virtual_path = Path::new("/provider.csv");
            let options = detect_options(virtual_path, Vec::new());
            let table = object(read_table(
                virtual_path,
                vec![Value::Struct(options.clone())],
            ));
            assert_eq!(
                table_variable_names_from_object(&table).unwrap(),
                vec!["Name".to_string(), "Score".to_string()]
            );
            match table_member_get(&table, &Value::from("Score")).unwrap() {
                Value::Tensor(tensor) => assert_eq!(tensor.data, vec![10.0, 12.0]),
                other => panic!("expected tensor, got {other:?}"),
            }

            block_on(runmat_filesystem::write_async(
                "/provider_numeric.csv",
                b"A,B\n1,2\n3,4\n",
            ))
            .expect("write provider numeric sample");
            let matrix_options = detect_options(Path::new("/provider_numeric.csv"), Vec::new());
            let matrix = block_on(
                crate::builtins::io::tabular::readmatrix::readmatrix_builtin(
                    Value::from("/provider_numeric.csv"),
                    vec![Value::Struct(matrix_options)],
                ),
            )
            .expect("readmatrix");
            match matrix {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape, vec![2, 2]);
                    assert_eq!(tensor.data, vec![1.0, 3.0, 2.0, 4.0]);
                }
                other => panic!("expected tensor, got {other:?}"),
            }
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn detect_import_options_honors_overrides_and_range() {
        let path = unique_path("detect_import_options_overrides");
        fs::write(&path, "ignore me\nRaw A|Raw B\n5|yes\n6|no\n").expect("write sample");
        let options = detect_options(
            &path,
            vec![
                Value::from("Delimiter"),
                Value::from("|"),
                Value::from("NumHeaderLines"),
                Value::Num(1.0),
                Value::from("VariableNamingRule"),
                Value::from("preserve"),
                Value::from("TextType"),
                Value::from("char"),
            ],
        );
        assert_eq!(options.fields.get("Delimiter"), Some(&Value::from("|")));
        assert_eq!(options.fields.get("NumHeaderLines"), Some(&Value::Num(2.0)));
        assert_eq!(
            options.fields.get("VariableNamingRule"),
            Some(&Value::from("preserve"))
        );
        match options.fields.get("VariableNames").unwrap() {
            Value::StringArray(array) => {
                assert_eq!(array.data, vec!["Raw A".to_string(), "Raw B".to_string()])
            }
            other => panic!("expected string array, got {other:?}"),
        }
        match options.fields.get("VariableTypes").unwrap() {
            Value::StringArray(array) => {
                assert_eq!(
                    array.data,
                    vec!["double".to_string(), "logical".to_string()]
                )
            }
            other => panic!("expected string array, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn spreadsheet_import_options_builds_editable_options_struct() {
        let options = spreadsheet_options(vec![
            Value::from("NumVariables"),
            Value::Num(2.0),
            Value::from("VariableTypes"),
            Value::StringArray(
                StringArray::new(vec!["double".into(), "string".into()], vec![1, 2]).unwrap(),
            ),
            Value::from("DataRange"),
            Value::from("A2:B5"),
        ]);
        assert_eq!(
            options.fields.get("FileType"),
            Some(&Value::from("spreadsheet"))
        );
        assert_eq!(options.fields.get("NumVariables"), Some(&Value::Num(2.0)));
        assert_eq!(options.fields.get("DataRange"), Some(&Value::from("A2:B5")));
        match options.fields.get("VariableNames").unwrap() {
            Value::StringArray(array) => {
                assert_eq!(array.data, vec!["Var1".to_string(), "Var2".to_string()]);
                assert_eq!(array.shape, vec![1, 2]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
        match options.fields.get("VariableTypes").unwrap() {
            Value::StringArray(array) => {
                assert_eq!(array.data, vec!["double".to_string(), "string".to_string()]);
                assert_eq!(array.shape, vec![1, 2]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn readtable_consumes_spreadsheet_import_options_struct() {
        let path = unique_path("readtable_spreadsheet_options");
        let path = path.with_extension("xlsx");
        write_minimal_xlsx(&path);
        let mut options = spreadsheet_options(vec![Value::from("NumVariables"), Value::Num(1.0)]);
        options.insert("Sheet", Value::from("Data"));
        options.insert("DataRange", Value::from("C2:C3"));
        options.insert(
            "VariableNames",
            Value::StringArray(StringArray::new(vec!["Amount".into()], vec![1, 1]).unwrap()),
        );
        options.insert(
            "VariableTypes",
            Value::StringArray(StringArray::new(vec!["double".into()], vec![1, 1]).unwrap()),
        );
        let table = object(read_table(&path, vec![Value::Struct(options)]));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["Amount".to_string()]
        );
        match table_member_get(&table, &Value::from("Amount")).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![200.0, 90.0]);
                assert_eq!(tensor.dtype, NumericDType::F64);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_default_spreadsheet_options_still_infers_headers() {
        let path = unique_path("readtable_default_spreadsheet_options");
        let path = path.with_extension("xlsx");
        write_minimal_xlsx(&path);
        let options = spreadsheet_options(Vec::new());
        let table = object(read_table(&path, vec![Value::Struct(options)]));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec![
                "Date".to_string(),
                "Orders".to_string(),
                "Revenue".to_string()
            ]
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_variable_types_coerce_imported_columns() {
        let path = unique_path("readtable_variable_types");
        fs::write(
            &path,
            "Value,Flag,When,Elapsed,Kind\n1.5,true,2026-06-01,01:30:00,A\n2.25,false,2026-06-02,02:00:00,B\n",
        )
        .expect("write sample");
        let types = StringArray::new(
            vec![
                "single".to_string(),
                "logical".to_string(),
                "datetime".to_string(),
                "duration".to_string(),
                "categorical".to_string(),
            ],
            vec![1, 5],
        )
        .unwrap();
        let table = object(read_table(
            &path,
            vec![Value::from("VariableTypes"), Value::StringArray(types)],
        ));
        match table_member_get(&table, &Value::from("Value")).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.dtype, NumericDType::F32);
                assert_eq!(tensor.data, vec![1.5, 2.25]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match table_member_get(&table, &Value::from("Flag")).unwrap() {
            Value::LogicalArray(array) => assert_eq!(array.data, vec![1, 0]),
            other => panic!("expected logical array, got {other:?}"),
        }
        match table_member_get(&table, &Value::from("When")).unwrap() {
            Value::Object(object) => assert!(object.is_class("datetime")),
            other => panic!("expected datetime object, got {other:?}"),
        }
        match table_member_get(&table, &Value::from("Elapsed")).unwrap() {
            Value::Object(object) => assert!(object.is_class("duration")),
            other => panic!("expected duration object, got {other:?}"),
        }
        match table_member_get(&table, &Value::from("Kind")).unwrap() {
            Value::Object(object) => assert!(object.is_class(CATEGORICAL_CLASS)),
            other => panic!("expected categorical object, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_preserves_explicit_import_variable_names_when_requested() {
        let path = unique_path("readtable_preserve_explicit_names");
        fs::write(&path, "100,10\n125,12\n").expect("write sample");
        let names = StringArray::new(
            vec!["daily revenue".to_string(), "total orders".to_string()],
            vec![1, 2],
        )
        .unwrap();
        let table = object(read_table(
            &path,
            vec![
                Value::from("ReadVariableNames"),
                Value::Bool(false),
                Value::from("VariableNames"),
                Value::StringArray(names),
                Value::from("VariableNamingRule"),
                Value::from("preserve"),
            ],
        ));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["daily revenue".to_string(), "total orders".to_string()]
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_text_type_char_imports_text_columns_as_char_matrix() {
        let path = unique_path("readtable_text_type_char");
        fs::write(&path, "Name\nAda\nGrace\n").expect("write sample");
        let table = object(read_table(
            &path,
            vec![Value::from("TextType"), Value::from("char")],
        ));
        match table_member_get(&table, &Value::from("Name")).unwrap() {
            Value::CharArray(array) => {
                assert_eq!(array.rows, 2);
                assert_eq!(array.cols, 5);
                assert_eq!(char_row(&array, 0), "Ada  ");
                assert_eq!(char_row(&array, 1), "Grace");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_variable_types_cellstr_imports_cell_column() {
        let path = unique_path("readtable_variable_types_cellstr");
        fs::write(&path, "Name\nAda\nGrace\n").expect("write sample");
        let types = StringArray::new(vec!["cellstr".to_string()], vec![1, 1]).unwrap();
        let table = object(read_table(
            &path,
            vec![Value::from("VariableTypes"), Value::StringArray(types)],
        ));
        match table_member_get(&table, &Value::from("Name")).unwrap() {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 1);
                assert_eq!(
                    cell.get(0, 0).unwrap(),
                    Value::CharArray(CharArray::new_row("Ada"))
                );
                assert_eq!(
                    cell.get(1, 0).unwrap(),
                    Value::CharArray(CharArray::new_row("Grace"))
                );
            }
            other => panic!("expected cell array, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readtable_rejects_unrepresented_import_variable_types() {
        let path = unique_path("readtable_unsupported_variable_types");
        fs::write(&path, "A\n1\n").expect("write sample");
        let unsupported_integer = StringArray::new(vec!["int8".to_string()], vec![1, 1]).unwrap();
        let err = read_table_err(
            &path,
            vec![
                Value::from("VariableTypes"),
                Value::StringArray(unsupported_integer),
            ],
        );
        assert!(err
            .message()
            .contains("unsupported VariableTypes entry 'int8'"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn table_properties_variable_names_rename_columns() {
        let a = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap());
        let mut table =
            object(table_from_columns(vec!["A".into(), "B".into()], vec![a, b]).unwrap());
        let mut props = table_public_properties(&table).unwrap();
        props.insert(
            VARIABLE_NAMES,
            Value::StringArray(StringArray::new(vec!["X".into(), "Y".into()], vec![1, 2]).unwrap()),
        );
        table_member_set(&mut table, PROPERTIES_MEMBER, Value::Struct(props)).unwrap();
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["X".to_string(), "Y".to_string()]
        );
    }

    #[test]
    fn table_paren_selects_rows_and_named_variables() {
        let a = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap());
        let table = object(table_from_columns(vec!["A".into(), "B".into()], vec![a, b]).unwrap());
        let selector = CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![3.0, 1.0], vec![1, 2]).unwrap()),
                Value::Cell(CellArray::new(vec![Value::from("B")], 1, 1).unwrap()),
            ],
            1,
            2,
        )
        .unwrap();
        let subset = object(table_paren_get(&table, &Value::Cell(selector)).unwrap());
        assert_eq!(
            table_variable_names_from_object(&subset).unwrap(),
            vec!["B".to_string()]
        );
        match table_member_get(&subset, &Value::from("B")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![6.0, 4.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn sortrows_preserves_row_names() {
        let values = Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap());
        let table = table_from_columns_with_properties(
            vec!["X".into()],
            vec![values],
            Some(vec!["second".into(), "first".into()]),
        )
        .unwrap();
        let (sorted, _) = sortrows_table(table, &[Value::from("X")]).unwrap();
        let sorted = object(sorted);
        let props = table_public_properties(&sorted).unwrap();
        match props.fields.get(ROW_NAMES).unwrap() {
            Value::StringArray(array) => {
                assert_eq!(array.data, vec!["first".to_string(), "second".to_string()]);
            }
            other => panic!("expected row names, got {other:?}"),
        }
    }

    #[test]
    fn groupsummary_mean_counts_groups() {
        let group = Value::StringArray(
            StringArray::new(vec!["a".into(), "b".into(), "a".into()], vec![3, 1]).unwrap(),
        );
        let value = Value::Tensor(Tensor::new(vec![2.0, 5.0, 4.0], vec![3, 1]).unwrap());
        let table = table_from_columns(vec!["G".into(), "X".into()], vec![group, value]).unwrap();
        let summary = groupsummary_impl(
            table,
            Value::from("G"),
            Value::from("mean"),
            vec![Value::from("X")],
        )
        .unwrap();
        let summary = object(summary);
        assert_eq!(
            table_variable_names_from_object(&summary).unwrap(),
            vec![
                "G".to_string(),
                "GroupCount".to_string(),
                "mean_X".to_string()
            ]
        );
        match table_member_get(&summary, &Value::from("mean_X")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![3.0, 5.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn groupsummary_orders_numeric_groups_numerically() {
        let group = Value::Tensor(Tensor::new(vec![10.0, 2.0, 10.0], vec![3, 1]).unwrap());
        let value = Value::Tensor(Tensor::new(vec![1.0, 5.0, 3.0], vec![3, 1]).unwrap());
        let table = table_from_columns(vec!["G".into(), "X".into()], vec![group, value]).unwrap();
        let summary =
            object(groupsummary_impl(table, Value::from("G"), Value::from("sum"), vec![]).unwrap());
        match table_member_get(&summary, &Value::from("G")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0, 10.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        match table_member_get(&summary, &Value::from("sum_X")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![5.0, 4.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn table_conversion_builtins_round_trip_arrays_cells_and_structs() {
        let matrix = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let table = block_on(array2table_builtin(
            matrix,
            vec![
                Value::from("VariableNames"),
                Value::Cell(
                    CellArray::new(vec![Value::from("A"), Value::from("B")], 1, 2).unwrap(),
                ),
            ],
        ))
        .unwrap();
        assert!(matches!(
            block_on(istable_builtin(table.clone())).unwrap(),
            Value::Bool(true)
        ));
        let array = block_on(table2array_builtin(table.clone())).unwrap();
        match array {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        let cells = block_on(table2cell_builtin(table.clone())).unwrap();
        match cells {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 2);
            }
            other => panic!("expected cell, got {other:?}"),
        }
        let st = block_on(table2struct_builtin(table.clone(), Vec::new())).unwrap();
        let round_trip = block_on(struct2table_builtin(st, Vec::new())).unwrap();
        assert_eq!(table_width(&object(round_trip)).unwrap(), 2);
    }

    #[test]
    fn timetable_conversion_predicates_and_head_work() {
        let values = Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap());
        let times = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let timetable = block_on(timetable_builtin(vec![
            times,
            values,
            Value::from("VariableNames"),
            Value::Cell(CellArray::new(vec![Value::from("X")], 1, 1).unwrap()),
        ]))
        .unwrap();
        assert!(matches!(
            block_on(istimetable_builtin(timetable.clone())).unwrap(),
            Value::Bool(true)
        ));
        let first_two = block_on(head_builtin(timetable.clone(), vec![Value::Num(2.0)])).unwrap();
        let first_two_object = object(first_two);
        assert_eq!(first_two_object.class_name, TIMETABLE_CLASS);
        assert_eq!(table_height(&first_two_object).unwrap(), 2);
        match timetable_row_times(&first_two_object).unwrap().unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 2.0]),
            other => panic!("expected selected row times, got {other:?}"),
        }
        let table = block_on(timetable2table_builtin(
            timetable,
            vec![Value::from("ConvertRowTimes"), Value::Bool(true)],
        ))
        .unwrap();
        assert_eq!(
            table_variable_names_from_object(&object(table.clone())).unwrap(),
            vec!["Time".to_string(), "X".to_string()]
        );
        let converted = block_on(table2timetable_builtin(table, Vec::new())).unwrap();
        let converted_object = object(converted);
        assert_eq!(converted_object.class_name, TIMETABLE_CLASS);
        assert_eq!(
            table_variable_names_from_object(&converted_object).unwrap(),
            vec!["Time".to_string(), "X".to_string()]
        );
        match timetable_row_times(&converted_object).unwrap().unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 2.0, 3.0]),
            other => panic!("expected timetable Time member, got {other:?}"),
        }
        match table_member_get(&converted_object, &Value::from("Time")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 2.0, 3.0]),
            other => panic!("expected retained Time variable, got {other:?}"),
        }
    }

    #[test]
    fn categorical_dictionary_and_selector_objects_materialize() {
        let categorical = block_on(categorical_builtin(vec![Value::StringArray(
            StringArray::new(vec!["red".into(), "blue".into(), "red".into()], vec![3, 1]).unwrap(),
        )]))
        .unwrap();
        assert!(matches!(
            block_on(iscategorical_builtin(categorical.clone())).unwrap(),
            Value::Bool(true)
        ));
        let Value::Object(cat) = categorical else {
            panic!("expected categorical object");
        };
        match cat.properties.get("Categories").unwrap() {
            Value::StringArray(array) => assert_eq!(array.data, vec!["red", "blue"]),
            other => panic!("expected categories, got {other:?}"),
        }

        let dictionary = block_on(dictionary_builtin(vec![
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        assert!(matches!(dictionary, Value::Object(obj) if obj.class_name == DICTIONARY_CLASS));
        assert!(matches!(
            block_on(timerange_builtin(vec![Value::Num(1.0), Value::Num(3.0)])).unwrap(),
            Value::Object(obj) if obj.class_name == TIMERANGE_CLASS
        ));
        assert!(matches!(
            block_on(vartype_builtin(Value::from("numeric"))).unwrap(),
            Value::Object(obj) if obj.class_name == VARTYPE_CLASS
        ));
    }

    #[test]
    fn writetable_and_readcell_cover_delimited_interop() {
        let path = unique_path("writetable_round_trip").with_extension("csv");
        let table = table_from_columns(
            vec!["A".into(), "Name".into()],
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                Value::StringArray(
                    StringArray::new(vec!["Ada".into(), "Grace".into()], vec![2, 1]).unwrap(),
                ),
            ],
        )
        .unwrap();
        let bytes = block_on(writetable_builtin(
            table,
            vec![Value::from(path.to_string_lossy().to_string())],
        ))
        .unwrap();
        assert!(matches!(bytes, Value::Num(n) if n > 0.0));
        let cells = block_on(readcell_builtin(
            Value::from(path.to_string_lossy().to_string()),
            Vec::new(),
        ))
        .unwrap();
        match cells {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 2);
                assert_eq!(cell.get(0, 0).unwrap(), Value::from("A"));
                assert_eq!(cell.get(0, 1).unwrap(), Value::from("Name"));
                assert_eq!(cell.get(1, 0).unwrap(), Value::Num(1.0));
                assert_eq!(cell.get(1, 1).unwrap(), Value::from("Ada"));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn timetable_rowtimes_option_keeps_all_variables_and_table2timetable_does_not_drop_data() {
        let a = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![30.0, 40.0], vec![2, 1]).unwrap());
        let times = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let tt = block_on(timetable_builtin(vec![
            a,
            b,
            Value::from("RowTimes"),
            times,
            Value::from("VariableNames"),
            Value::Cell(CellArray::new(vec![Value::from("A"), Value::from("B")], 1, 2).unwrap()),
        ]))
        .unwrap();
        let tt_obj = object(tt);
        assert_eq!(
            table_variable_names_from_object(&tt_obj).unwrap(),
            vec!["A".to_string(), "B".to_string()]
        );

        let t = table_from_columns(
            vec!["A".into(), "B".into()],
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
            ],
        )
        .unwrap();
        let converted = block_on(table2timetable_builtin(t, Vec::new())).unwrap();
        let converted_obj = object(converted);
        assert_eq!(
            table_variable_names_from_object(&converted_obj).unwrap(),
            vec!["A".to_string(), "B".to_string()]
        );
        match timetable_row_times(&converted_obj).unwrap().unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 2.0]),
            other => panic!("expected generated row times, got {other:?}"),
        }

        let path = unique_path("readtimetable_rowtimes").with_extension("csv");
        fs::write(&path, "A\n10\n20\n").unwrap();
        let read = block_on(readtimetable_builtin(
            Value::from(path.to_string_lossy().to_string()),
            vec![
                Value::from("RowTimes"),
                Value::Tensor(Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap()),
            ],
        ))
        .unwrap();
        let read_obj = object(read);
        match timetable_row_times(&read_obj).unwrap().unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![5.0, 6.0]),
            other => panic!("expected explicit readtimetable row times, got {other:?}"),
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn table_selector_objects_filter_rows_and_variables() {
        let t = table_from_columns(
            vec!["A".into(), "Name".into()],
            vec![
                Value::Tensor(Tensor::new(vec![-1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
                Value::StringArray(
                    StringArray::new(vec!["x".into(), "y".into(), "z".into()], vec![3, 1]).unwrap(),
                ),
            ],
        )
        .unwrap();
        let t_obj = object(t.clone());
        let numeric = block_on(vartype_builtin(Value::from("numeric"))).unwrap();
        let subset = object(
            table_paren_get(
                &t_obj,
                &Value::Cell(CellArray::new(vec![Value::from(":"), numeric], 1, 2).unwrap()),
            )
            .unwrap(),
        );
        assert_eq!(
            table_variable_names_from_object(&subset).unwrap(),
            vec!["A".to_string()]
        );

        let filter = block_on(rowfilter_builtin(vec![
            Value::Cell(CellArray::new(vec![Value::from("A")], 1, 1).unwrap()),
            Value::from("@gt0"),
        ]))
        .unwrap();
        let filtered = object(
            table_paren_get(
                &t_obj,
                &Value::Cell(CellArray::new(vec![filter, Value::from(":")], 1, 2).unwrap()),
            )
            .unwrap(),
        );
        assert_eq!(table_height(&filtered).unwrap(), 2);

        let tt = block_on(timetable_builtin(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap()),
            Value::from("VariableNames"),
            Value::Cell(CellArray::new(vec![Value::from("X")], 1, 1).unwrap()),
        ]))
        .unwrap();
        let tt_obj = object(tt);
        let range = block_on(timerange_builtin(vec![Value::Num(2.0), Value::Num(3.0)])).unwrap();
        let ranged = object(
            table_paren_get(
                &tt_obj,
                &Value::Cell(CellArray::new(vec![range, Value::from(":")], 1, 2).unwrap()),
            )
            .unwrap(),
        );
        assert_eq!(ranged.class_name, TIMETABLE_CLASS);
        assert_eq!(table_height(&ranged).unwrap(), 2);

        let open_left = block_on(timerange_builtin(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::from("openleft"),
        ]))
        .unwrap();
        let ranged = object(
            table_paren_get(
                &tt_obj,
                &Value::Cell(CellArray::new(vec![open_left, Value::from(":")], 1, 2).unwrap()),
            )
            .unwrap(),
        );
        match table_member_get(&ranged, &Value::from("X")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![30.0]),
            other => panic!("expected openleft range result, got {other:?}"),
        }

        let open_right = block_on(timerange_builtin(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::from("openright"),
        ]))
        .unwrap();
        let ranged = object(
            table_paren_get(
                &tt_obj,
                &Value::Cell(CellArray::new(vec![open_right, Value::from(":")], 1, 2).unwrap()),
            )
            .unwrap(),
        );
        match table_member_get(&ranged, &Value::from("X")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![20.0]),
            other => panic!("expected openright range result, got {other:?}"),
        }
    }

    #[test]
    fn pivot_builds_wide_summary_table() {
        let t = table_from_columns(
            vec!["Group".into(), "Kind".into(), "Value".into()],
            vec![
                Value::StringArray(
                    StringArray::new(vec!["A".into(), "A".into(), "B".into()], vec![3, 1]).unwrap(),
                ),
                Value::StringArray(
                    StringArray::new(vec!["x".into(), "y".into(), "x".into()], vec![3, 1]).unwrap(),
                ),
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            ],
        )
        .unwrap();
        let out = block_on(pivot_builtin(
            t,
            Value::from("Group"),
            Value::from("Kind"),
            Value::from("Value"),
            Vec::new(),
        ))
        .unwrap();
        let out_obj = object(out);
        let names = table_variable_names_from_object(&out_obj).unwrap();
        assert!(names.contains(&"x_Value".to_string()));
        assert!(names.contains(&"y_Value".to_string()));
        assert_eq!(table_height(&out_obj).unwrap(), 2);
    }

    #[test]
    fn table2struct_defaults_to_row_structs_and_to_scalar_preserves_columns() {
        let t = table_from_columns(
            vec!["A".into(), "B".into()],
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
            ],
        )
        .unwrap();
        match block_on(table2struct_builtin(t.clone(), Vec::new())).unwrap() {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert!(matches!(cell.data[0], Value::Struct(_)));
            }
            other => panic!("expected struct array cell, got {other:?}"),
        }
        match block_on(table2struct_builtin(
            t,
            vec![Value::from("ToScalar"), Value::Bool(true)],
        ))
        .unwrap()
        {
            Value::Struct(st) => assert!(st.fields.contains_key("A")),
            other => panic!("expected scalar struct, got {other:?}"),
        }
    }

    #[test]
    fn categorical_categories_and_dictionary_lookup_have_semantics() {
        let categorical = block_on(categorical_builtin(vec![
            Value::StringArray(
                StringArray::new(vec!["b".into(), "a".into(), "c".into()], vec![3, 1]).unwrap(),
            ),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
            Value::from("Ordinal"),
            Value::Bool(true),
        ]))
        .unwrap();
        let Value::Object(cat) = categorical else {
            panic!("expected categorical");
        };
        match cat.properties.get("Codes").unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.data[0], 2.0);
                assert_eq!(tensor.data[1], 1.0);
                assert!(tensor.data[2].is_nan());
            }
            other => panic!("expected categorical codes, got {other:?}"),
        }

        let dictionary = block_on(dictionary_builtin(vec![
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
        ]))
        .unwrap();
        let value = block_on(dictionary_subsref(
            dictionary,
            OBJECT_INDEX_PAREN.to_string(),
            Value::from("b"),
        ))
        .unwrap();
        assert_eq!(value, Value::Num(20.0));
    }
}
