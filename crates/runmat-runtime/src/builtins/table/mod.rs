//! MATLAB table datatype support and tabular workflow builtins.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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

pub const TABLE_CLASS: &str = "table";
const TABLE_VARIABLES_FIELD: &str = "__table_variables";
const TABLE_PROPERTIES_FIELD: &str = "__table_properties";
const PROPERTIES_MEMBER: &str = "Properties";
const VARIABLE_NAMES: &str = "VariableNames";
const ROW_NAMES: &str = "RowNames";
const DIMENSION_NAMES: &str = "DimensionNames";
const VARIABLE_UNITS: &str = "VariableUnits";
const VARIABLE_DESCRIPTIONS: &str = "VariableDescriptions";
const DESCRIPTION: &str = "Description";
const USER_DATA: &str = "UserData";
const DEFAULT_ROW_DIM_NAME: &str = "Rows";
const DEFAULT_VARIABLE_DIM_NAME: &str = "Variables";

static TABLE_CLASS_REGISTERED: OnceLock<()> = OnceLock::new();

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
    TABLE_CLASS_REGISTERED.get_or_init(|| {
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
                default_value: Some(Value::Struct(default_properties(Vec::new(), None))),
            },
        );

        let mut methods = HashMap::new();
        for name in [OBJECT_SUBSREF_METHOD, OBJECT_SUBSASGN_METHOD] {
            methods.insert(
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: format!("{TABLE_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        runmat_builtins::register_class(ClassDef {
            name: TABLE_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
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

async fn gather_values(values: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
}

#[derive(Default)]
struct TableConstructorOptions {
    variable_names: Option<Vec<String>>,
    row_names: Option<Vec<String>>,
}

fn split_table_constructor_args(
    args: Vec<Value>,
) -> BuiltinResult<(Vec<Value>, TableConstructorOptions)> {
    let mut variables = Vec::new();
    let mut options = TableConstructorOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if let Ok(name) = scalar_text(&args[idx], "table option") {
            if idx + 1 < args.len() && is_table_constructor_option(&name) {
                let value = &args[idx + 1];
                if name.eq_ignore_ascii_case("VariableNames") {
                    options.variable_names = Some(variable_name_list(value)?);
                } else if name.eq_ignore_ascii_case("RowNames") {
                    options.row_names = Some(string_list(value)?);
                }
                idx += 2;
                continue;
            }
        }
        variables.push(args[idx].clone());
        idx += 1;
    }
    Ok((variables, options))
}

fn is_table_constructor_option(name: &str) -> bool {
    name.eq_ignore_ascii_case("VariableNames") || name.eq_ignore_ascii_case("RowNames")
}

#[derive(Clone)]
struct ReadTableOptions {
    file_type: ImportFileType,
    delimiter: Option<Delimiter>,
    read_variable_names: Option<bool>,
    read_row_names: bool,
    num_variables: Option<usize>,
    variable_names: Option<Vec<String>>,
    variable_types: Option<Vec<ImportVariableType>>,
    row_names: Option<Vec<String>>,
    num_header_lines: usize,
    range: Option<RangeSpec>,
    sheet: Option<SheetSelector>,
    preserve_variable_names: bool,
    treat_as_missing: HashSet<String>,
    empty_line_rule: EmptyLineRule,
    text_type: TextImportType,
    encoding: String,
    datetime_type: DatetimeImportType,
}

impl Default for ReadTableOptions {
    fn default() -> Self {
        Self {
            file_type: ImportFileType::Auto,
            delimiter: None,
            read_variable_names: None,
            read_row_names: false,
            num_variables: None,
            variable_names: None,
            variable_types: None,
            row_names: None,
            num_header_lines: 0,
            range: None,
            sheet: None,
            preserve_variable_names: false,
            treat_as_missing: HashSet::new(),
            empty_line_rule: EmptyLineRule::Skip,
            text_type: TextImportType::String,
            encoding: "utf-8".to_string(),
            datetime_type: DatetimeImportType::Datetime,
        }
    }
}

impl ReadTableOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut options = Self::default();
        let mut idx = 0usize;
        if let Some(Value::Struct(st)) = args.first() {
            for (name, value) in &st.fields {
                options.apply(name, value)?;
            }
            idx = 1;
        }
        while idx < args.len() {
            if idx + 1 >= args.len() {
                return Err(invalid_argument(
                    "readtable: name-value options must be provided in pairs",
                ));
            }
            let name = scalar_text(&args[idx], "readtable option")?;
            options.apply(&name, &args[idx + 1])?;
            idx += 2;
        }
        Ok(options)
    }

    fn apply(&mut self, name: &str, value: &Value) -> BuiltinResult<()> {
        if name.eq_ignore_ascii_case("FileType") {
            self.file_type = ImportFileType::parse(value)?;
        } else if name.eq_ignore_ascii_case("Delimiter") {
            self.delimiter = Some(Delimiter::parse(value)?);
        } else if name.eq_ignore_ascii_case("ReadVariableNames") {
            self.read_variable_names = Some(bool_scalar(value, "ReadVariableNames")?);
        } else if name.eq_ignore_ascii_case("ReadRowNames") {
            self.read_row_names = bool_scalar(value, "ReadRowNames")?;
        } else if name.eq_ignore_ascii_case("NumVariables") {
            let count = nonnegative_usize(value, "NumVariables")?;
            self.num_variables = (count > 0).then_some(count);
        } else if name.eq_ignore_ascii_case("VariableNames") {
            self.variable_names = optional_raw_variable_name_list(value)?;
        } else if name.eq_ignore_ascii_case("VariableTypes") {
            self.variable_types = optional_variable_type_list(value)?;
        } else if name.eq_ignore_ascii_case("RowNames") {
            self.row_names = Some(string_list(value)?);
        } else if name.eq_ignore_ascii_case("NumHeaderLines") {
            self.num_header_lines = nonnegative_usize(value, "NumHeaderLines")?;
        } else if name.eq_ignore_ascii_case("Range") {
            self.range = Some(RangeSpec::parse(value)?);
        } else if name.eq_ignore_ascii_case("DataRange") {
            self.range = optional_range_spec(value)?;
        } else if name.eq_ignore_ascii_case("Sheet") {
            self.sheet = optional_sheet_selector(value)?;
        } else if name.eq_ignore_ascii_case("TreatAsMissing") {
            for token in string_list(value)? {
                self.treat_as_missing
                    .insert(token.trim().to_ascii_lowercase());
            }
        } else if name.eq_ignore_ascii_case("PreserveVariableNames") {
            self.preserve_variable_names = bool_scalar(value, "PreserveVariableNames")?;
        } else if name.eq_ignore_ascii_case("VariableNamingRule") {
            let rule = scalar_text(value, "VariableNamingRule")?;
            if rule.eq_ignore_ascii_case("preserve") {
                self.preserve_variable_names = true;
            } else if rule.eq_ignore_ascii_case("modify") {
                self.preserve_variable_names = false;
            } else {
                return Err(invalid_argument(format!(
                    "readtable: unsupported VariableNamingRule '{rule}'"
                )));
            }
        } else if name.eq_ignore_ascii_case("EmptyLineRule") {
            let rule = scalar_text(value, "EmptyLineRule")?;
            self.empty_line_rule = if rule.eq_ignore_ascii_case("read") {
                EmptyLineRule::Read
            } else if rule.eq_ignore_ascii_case("skip") {
                EmptyLineRule::Skip
            } else {
                return Err(invalid_argument(format!(
                    "readtable: unsupported EmptyLineRule '{rule}'"
                )));
            };
        } else if name.eq_ignore_ascii_case("Encoding") {
            let encoding = scalar_text(value, "Encoding")?;
            validate_encoding_label(&encoding)?;
            self.encoding = encoding;
        } else if name.eq_ignore_ascii_case("TextType") {
            self.text_type = TextImportType::parse(value, "readtable")?;
        } else if name.eq_ignore_ascii_case("DatetimeType") {
            self.datetime_type = DatetimeImportType::parse(value)?;
        } else {
            return Err(invalid_argument(format!(
                "readtable: unsupported option '{name}'"
            )));
        }
        Ok(())
    }

    fn is_missing(&self, token: &str) -> bool {
        let trimmed = token.trim();
        trimmed.is_empty()
            || self
                .treat_as_missing
                .contains(&trimmed.to_ascii_lowercase())
    }
}

fn spreadsheet_import_options(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "spreadsheetImportOptions: name-value options must be provided in pairs",
        ));
    }
    let mut options = SpreadsheetImportOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "spreadsheetImportOptions option")?;
        options.apply(&name, &args[idx + 1])?;
        idx += 2;
    }
    Ok(Value::Struct(options.into_struct()?))
}

#[derive(Clone)]
struct SpreadsheetImportOptions {
    num_variables: usize,
    read_variable_names: Option<bool>,
    read_row_names: bool,
    variable_names: Vec<String>,
    variable_types: Vec<String>,
    data_range: Option<Value>,
    sheet: Option<Value>,
    treat_as_missing: Vec<String>,
    preserve_variable_names: bool,
    empty_line_rule: String,
    text_type: String,
    datetime_type: String,
}

impl Default for SpreadsheetImportOptions {
    fn default() -> Self {
        let num_variables = 0;
        Self {
            num_variables,
            read_variable_names: None,
            read_row_names: false,
            variable_names: Vec::new(),
            variable_types: Vec::new(),
            data_range: None,
            sheet: None,
            treat_as_missing: Vec::new(),
            preserve_variable_names: false,
            empty_line_rule: "skip".to_string(),
            text_type: "string".to_string(),
            datetime_type: "datetime".to_string(),
        }
    }
}

impl SpreadsheetImportOptions {
    fn apply(&mut self, name: &str, value: &Value) -> BuiltinResult<()> {
        if name.eq_ignore_ascii_case("NumVariables") {
            self.resize_variables(positive_usize(value, "NumVariables")?);
        } else if name.eq_ignore_ascii_case("VariableNames") {
            self.variable_names = raw_variable_name_list(value)?;
            self.align_variable_metadata_count(self.variable_names.len(), "VariableNames")?;
            self.ensure_variable_metadata_len();
        } else if name.eq_ignore_ascii_case("VariableTypes") {
            let types = variable_type_names(value)?;
            self.variable_types = types;
            self.align_variable_metadata_count(self.variable_types.len(), "VariableTypes")?;
            self.ensure_variable_metadata_len();
        } else if name.eq_ignore_ascii_case("DataRange") || name.eq_ignore_ascii_case("Range") {
            self.data_range = if option_value_is_empty(value) {
                None
            } else {
                RangeSpec::parse(value)?;
                Some(value.clone())
            };
        } else if name.eq_ignore_ascii_case("Sheet") {
            self.sheet = if option_value_is_empty(value) {
                None
            } else {
                SheetSelector::parse(value)?;
                Some(value.clone())
            };
        } else if name.eq_ignore_ascii_case("ReadVariableNames") {
            self.read_variable_names = Some(bool_scalar(value, "ReadVariableNames")?);
        } else if name.eq_ignore_ascii_case("ReadRowNames") {
            self.read_row_names = bool_scalar(value, "ReadRowNames")?;
        } else if name.eq_ignore_ascii_case("TreatAsMissing") {
            self.treat_as_missing = string_list(value)?;
        } else if name.eq_ignore_ascii_case("PreserveVariableNames") {
            self.preserve_variable_names = bool_scalar(value, "PreserveVariableNames")?;
        } else if name.eq_ignore_ascii_case("VariableNamingRule") {
            let rule = scalar_text(value, "VariableNamingRule")?;
            if rule.eq_ignore_ascii_case("preserve") {
                self.preserve_variable_names = true;
            } else if rule.eq_ignore_ascii_case("modify") {
                self.preserve_variable_names = false;
            } else {
                return Err(invalid_argument(format!(
                    "spreadsheetImportOptions: unsupported VariableNamingRule '{rule}'"
                )));
            }
        } else if name.eq_ignore_ascii_case("EmptyLineRule") {
            let rule = scalar_text(value, "EmptyLineRule")?;
            if !(rule.eq_ignore_ascii_case("read") || rule.eq_ignore_ascii_case("skip")) {
                return Err(invalid_argument(format!(
                    "spreadsheetImportOptions: unsupported EmptyLineRule '{rule}'"
                )));
            }
            self.empty_line_rule = rule.to_ascii_lowercase();
        } else if name.eq_ignore_ascii_case("TextType") {
            let text_type = scalar_text(value, "TextType")?;
            if !(text_type.eq_ignore_ascii_case("string") || text_type.eq_ignore_ascii_case("char"))
            {
                return Err(invalid_argument(format!(
                    "spreadsheetImportOptions: unsupported TextType '{text_type}'"
                )));
            }
            self.text_type = text_type.to_ascii_lowercase();
        } else if name.eq_ignore_ascii_case("DatetimeType") {
            let datetime_type = scalar_text(value, "DatetimeType")?;
            if !(datetime_type.eq_ignore_ascii_case("datetime")
                || datetime_type.eq_ignore_ascii_case("text")
                || datetime_type.eq_ignore_ascii_case("exceldatenum"))
            {
                return Err(invalid_argument(format!(
                    "spreadsheetImportOptions: unsupported DatetimeType '{datetime_type}'"
                )));
            }
            self.datetime_type = datetime_type.to_ascii_lowercase();
        } else {
            return Err(invalid_argument(format!(
                "spreadsheetImportOptions: unsupported option '{name}'"
            )));
        }
        Ok(())
    }

    fn resize_variables(&mut self, num_variables: usize) {
        self.num_variables = num_variables;
        if self.variable_names.len() > num_variables {
            self.variable_names.truncate(num_variables);
        }
        if self.variable_types.len() > num_variables {
            self.variable_types.truncate(num_variables);
        }
        self.ensure_variable_metadata_len();
    }

    fn align_variable_metadata_count(&mut self, len: usize, field: &str) -> BuiltinResult<()> {
        if self.num_variables == 0 {
            self.num_variables = len;
            return Ok(());
        }
        if len > self.num_variables {
            return Err(invalid_argument(format!(
                "spreadsheetImportOptions: {field} length exceeds NumVariables"
            )));
        }
        Ok(())
    }

    fn ensure_variable_metadata_len(&mut self) {
        if self.num_variables == 0 {
            return;
        }
        while self.variable_names.len() < self.num_variables {
            self.variable_names
                .push(format!("Var{}", self.variable_names.len() + 1));
        }
        self.variable_names.truncate(self.num_variables);
        while self.variable_types.len() < self.num_variables {
            self.variable_types.push("auto".to_string());
        }
        self.variable_types.truncate(self.num_variables);
    }

    fn into_struct(mut self) -> BuiltinResult<StructValue> {
        self.ensure_variable_metadata_len();
        let mut out = StructValue::new();
        out.insert("FileType", Value::String("spreadsheet".to_string()));
        out.insert("NumVariables", Value::Num(self.num_variables as f64));
        if let Some(read_variable_names) = self.read_variable_names {
            out.insert("ReadVariableNames", Value::Bool(read_variable_names));
        }
        out.insert("ReadRowNames", Value::Bool(self.read_row_names));
        out.insert(
            "VariableNames",
            Value::StringArray(
                StringArray::new(
                    self.variable_names.clone(),
                    vec![1, self.variable_names.len()],
                )
                .map_err(|err| invalid_variable(format!("spreadsheetImportOptions: {err}")))?,
            ),
        );
        out.insert(
            "VariableTypes",
            Value::StringArray(
                StringArray::new(
                    self.variable_types.clone(),
                    vec![1, self.variable_types.len()],
                )
                .map_err(|err| invalid_variable(format!("spreadsheetImportOptions: {err}")))?,
            ),
        );
        out.insert(
            "DataRange",
            self.data_range
                .unwrap_or_else(|| Value::String(String::new())),
        );
        out.insert(
            "Sheet",
            self.sheet.unwrap_or_else(|| Value::String(String::new())),
        );
        out.insert(
            "TreatAsMissing",
            Value::StringArray(
                StringArray::new(
                    self.treat_as_missing.clone(),
                    vec![1, self.treat_as_missing.len()],
                )
                .map_err(|err| invalid_variable(format!("spreadsheetImportOptions: {err}")))?,
            ),
        );
        out.insert(
            "PreserveVariableNames",
            Value::Bool(self.preserve_variable_names),
        );
        out.insert(
            "VariableNamingRule",
            Value::String(if self.preserve_variable_names {
                "preserve".to_string()
            } else {
                "modify".to_string()
            }),
        );
        out.insert("EmptyLineRule", Value::String(self.empty_line_rule));
        out.insert("TextType", Value::String(self.text_type));
        out.insert("DatetimeType", Value::String(self.datetime_type));
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImportVariableType {
    Auto,
    Numeric(NumericDType),
    Logical,
    Text(TextImportType),
    CellStr,
    Datetime,
    Duration,
}

impl ImportVariableType {
    fn parse(raw: &str) -> BuiltinResult<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "double" => Ok(Self::Numeric(NumericDType::F64)),
            "single" => Ok(Self::Numeric(NumericDType::F32)),
            "uint8" => Ok(Self::Numeric(NumericDType::U8)),
            "uint16" => Ok(Self::Numeric(NumericDType::U16)),
            "logical" | "bool" | "boolean" => Ok(Self::Logical),
            "string" => Ok(Self::Text(TextImportType::String)),
            "char" => Ok(Self::Text(TextImportType::Char)),
            "cellstr" => Ok(Self::CellStr),
            "int8" | "int16" | "int32" | "int64" | "uint32" | "uint64" => {
                Err(invalid_argument(format!(
                    "readtable: unsupported VariableTypes entry '{}'; RunMat table imports currently support double, single, uint8, and uint16 numeric arrays",
                    raw.trim()
                )))
            }
            "categorical" => Err(invalid_argument(
                "readtable: unsupported VariableTypes entry 'categorical'; categorical arrays are not implemented in RunMat yet",
            )),
            "datetime" => Ok(Self::Datetime),
            "duration" => Ok(Self::Duration),
            other => Err(invalid_argument(format!(
                "readtable: unsupported VariableTypes entry '{other}'"
            ))),
        }
    }

    fn canonical_label(raw: &str) -> BuiltinResult<String> {
        Self::parse(raw)?;
        let label = raw.trim().to_ascii_lowercase();
        Ok(if label.is_empty() {
            "auto".to_string()
        } else {
            label
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextImportType {
    String,
    Char,
}

impl TextImportType {
    fn parse(value: &Value, context: &str) -> BuiltinResult<Self> {
        let text_type = scalar_text(value, "TextType")?;
        match text_type.trim().to_ascii_lowercase().as_str() {
            "string" => Ok(Self::String),
            "char" => Ok(Self::Char),
            other => Err(invalid_argument(format!(
                "{context}: unsupported TextType '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Copy)]
enum EmptyLineRule {
    Skip,
    Read,
}

#[derive(Clone, Copy)]
enum DatetimeImportType {
    Datetime,
    Text,
    ExcelDatenum,
}

impl DatetimeImportType {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "DatetimeType")?;
        match text.trim().to_ascii_lowercase().as_str() {
            "datetime" => Ok(Self::Datetime),
            "text" => Ok(Self::Text),
            "exceldatenum" => Ok(Self::ExcelDatenum),
            other => Err(invalid_argument(format!(
                "readtable: unsupported DatetimeType '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImportFileType {
    Auto,
    Text,
    Spreadsheet,
}

impl ImportFileType {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "FileType")?;
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "text" | "delimitedtext" | "delimited" => Ok(Self::Text),
            "spreadsheet" | "excel" => Ok(Self::Spreadsheet),
            other => Err(invalid_argument(format!(
                "readtable: unsupported FileType '{other}'"
            ))),
        }
    }
}

#[derive(Clone)]
enum SheetSelector {
    Name(String),
    Index(usize),
}

impl SheetSelector {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::Int(i) if i.to_i64() >= 1 => Ok(Self::Index(i.to_i64() as usize - 1)),
            Value::Num(n)
                if n.is_finite() && *n >= 1.0 && (n.round() - n).abs() <= f64::EPSILON =>
            {
                Ok(Self::Index(n.round() as usize - 1))
            }
            _ => {
                let text = scalar_text(value, "Sheet")?;
                if text.trim().is_empty() {
                    return Err(invalid_argument("readtable: Sheet must not be empty"));
                }
                Ok(Self::Name(text))
            }
        }
    }
}

#[derive(Clone)]
enum Delimiter {
    Char(char),
    String(String),
    Whitespace,
}

impl Delimiter {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "Delimiter")?;
        if text.is_empty() {
            return Err(invalid_argument("readtable: Delimiter must not be empty"));
        }
        match text.trim().to_ascii_lowercase().as_str() {
            "tab" => Ok(Self::Char('\t')),
            "space" | "whitespace" => Ok(Self::Whitespace),
            "comma" => Ok(Self::Char(',')),
            "semicolon" => Ok(Self::Char(';')),
            "bar" | "pipe" => Ok(Self::Char('|')),
            _ if text.chars().count() == 1 => Ok(Self::Char(text.chars().next().unwrap())),
            _ => Ok(Self::String(text)),
        }
    }
}

#[derive(Clone, Copy)]
struct RangeSpec {
    start_row: usize,
    start_col: usize,
    end_row: Option<usize>,
    end_col: Option<usize>,
}

impl RangeSpec {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Self::parse_text(text),
            Value::CharArray(ca) if ca.rows == 1 => {
                let text: String = ca.data.iter().collect();
                Self::parse_text(&text)
            }
            Value::StringArray(sa) if sa.data.len() == 1 => Self::parse_text(&sa.data[0]),
            Value::Tensor(t) if t.data.len() == 2 || t.data.len() == 4 => {
                let mut indices = Vec::with_capacity(t.data.len());
                for value in &t.data {
                    indices.push(one_based_to_zero(*value, usize::MAX, "Range")?);
                }
                Ok(Self {
                    start_row: indices[0],
                    start_col: indices[1],
                    end_row: indices.get(2).copied(),
                    end_col: indices.get(3).copied(),
                })
            }
            _ => Err(invalid_argument(
                "readtable: Range must be a cell reference string or numeric vector",
            )),
        }
    }

    fn parse_text(text: &str) -> BuiltinResult<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(invalid_argument("readtable: Range must not be empty"));
        }
        let parts: Vec<&str> = trimmed.split(':').collect();
        if parts.len() > 2 {
            return Err(invalid_argument(format!(
                "readtable: invalid Range specification '{trimmed}'"
            )));
        }
        let start = parse_cell_ref(parts[0])?;
        let end = if parts.len() == 2 {
            Some(parse_cell_ref(parts[1])?)
        } else {
            None
        };
        Ok(Self {
            start_row: start.0.unwrap_or(0),
            start_col: start.1.unwrap_or(0),
            end_row: end.and_then(|item| item.0),
            end_col: end.and_then(|item| item.1),
        })
    }
}

fn parse_cell_ref(token: &str) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    let mut letters = String::new();
    let mut digits = String::new();
    for ch in token.trim().chars() {
        if ch == '$' {
            continue;
        }
        if ch.is_ascii_alphabetic() {
            letters.push(ch.to_ascii_uppercase());
        } else if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            return Err(invalid_argument(format!(
                "readtable: invalid Range component '{token}'"
            )));
        }
    }
    let col = if letters.is_empty() {
        None
    } else {
        let mut value = 0usize;
        for ch in letters.chars() {
            value = value
                .checked_mul(26)
                .and_then(|v| v.checked_add((ch as u8 - b'A' + 1) as usize))
                .ok_or_else(|| invalid_argument("readtable: Range column overflow"))?;
        }
        Some(value - 1)
    };
    let row = if digits.is_empty() {
        None
    } else {
        let parsed = digits
            .parse::<usize>()
            .map_err(|_| invalid_argument("readtable: invalid Range row"))?;
        if parsed == 0 {
            return Err(invalid_argument("readtable: Range rows are one-based"));
        }
        Some(parsed - 1)
    };
    Ok((row, col))
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let text = scalar_text(value, "filename").map_err(|_| {
        table_error(
            &TABLE_ERROR_INVALID_ARGUMENT,
            "readtable: filename must be a string scalar or character vector",
        )
    })?;
    if text.trim().is_empty() {
        return Err(invalid_argument("readtable: filename must not be empty"));
    }
    let expanded =
        expand_user_path(&text, "readtable").map_err(|msg| invalid_argument(msg.to_string()))?;
    Ok(Path::new(&expanded).to_path_buf())
}

async fn read_table_from_file(path: &Path, options: &ReadTableOptions) -> BuiltinResult<Value> {
    match options.file_type {
        ImportFileType::Spreadsheet => read_spreadsheet_table(path, options).await,
        ImportFileType::Text => read_text_table(path, options).await,
        ImportFileType::Auto if is_spreadsheet_path(path) => {
            read_spreadsheet_table(path, options).await
        }
        ImportFileType::Auto => read_text_table(path, options).await,
    }
}

async fn read_text_table(path: &Path, options: &ReadTableOptions) -> BuiltinResult<Value> {
    if options.sheet.is_some() {
        return Err(invalid_argument(
            "readtable: Sheet is only valid for spreadsheet files",
        ));
    }
    let bytes = read_file_bytes(path).await?;
    let text = decode_text_bytes(&bytes, &options.encoding)?;
    let mut raw_lines = text.lines().map(ToString::to_string).collect::<Vec<_>>();
    if let Some(first) = raw_lines.first_mut() {
        if first.starts_with('\u{FEFF}') {
            *first = first.trim_start_matches('\u{FEFF}').to_string();
        }
    }
    let delimiter = options
        .delimiter
        .clone()
        .or_else(|| detect_delimiter(&raw_lines))
        .unwrap_or(Delimiter::Whitespace);
    let mut rows = parse_text_records(&text, &delimiter, options.empty_line_rule);
    if options.num_header_lines > 0 {
        rows = rows.into_iter().skip(options.num_header_lines).collect();
    }
    if let Some(range) = options.range {
        rows = apply_import_range(rows, range);
    }
    import_rows_to_table(rows, options)
}

async fn read_spreadsheet_table(path: &Path, options: &ReadTableOptions) -> BuiltinResult<Value> {
    if options.delimiter.is_some() {
        return Err(invalid_argument(
            "readtable: Delimiter is only valid for text files",
        ));
    }
    let bytes = read_file_bytes(path).await?;
    let cursor = Cursor::new(bytes);
    let mut workbook = open_workbook_auto_from_rs(cursor).map_err(|err| {
        table_error(
            &TABLE_ERROR_UNSUPPORTED_FILE,
            format!(
                "readtable: unable to open spreadsheet '{}': {err}",
                path.display()
            ),
        )
    })?;
    let range = match &options.sheet {
        Some(SheetSelector::Name(name)) => workbook.worksheet_range(name).map_err(|err| {
            invalid_argument(format!("readtable: unable to read sheet '{name}': {err:?}"))
        })?,
        Some(SheetSelector::Index(index)) => workbook
            .worksheet_range_at(*index)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "readtable: sheet index {} exceeds bounds",
                    index + 1
                ))
            })?
            .map_err(|err| {
                invalid_argument(format!(
                    "readtable: unable to read sheet {}: {err:?}",
                    index + 1
                ))
            })?,
        None => workbook
            .worksheet_range_at(0)
            .ok_or_else(|| invalid_argument("readtable: spreadsheet contains no worksheets"))?
            .map_err(|err| {
                invalid_argument(format!("readtable: unable to read first sheet: {err:?}"))
            })?,
    };
    let rows = spreadsheet_range_to_rows(&range, options)?;
    import_rows_to_table(rows, options)
}

async fn read_file_bytes(path: &Path) -> BuiltinResult<Vec<u8>> {
    let mut file = File::open_async(path).await.map_err(|err| {
        table_error_with_source(
            &TABLE_ERROR_IO,
            format!("readtable: unable to open '{}': {err}", path.display()),
            err,
        )
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|err| {
        table_error_with_source(
            &TABLE_ERROR_IO,
            format!("readtable: unable to read '{}': {err}", path.display()),
            err,
        )
    })?;
    Ok(bytes)
}

fn is_spreadsheet_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref(),
        Some("xls") | Some("xlsx") | Some("xlsm") | Some("xlsb") | Some("ods")
    )
}

fn validate_encoding_label(label: &str) -> BuiltinResult<()> {
    encoding_for_label(label)
        .map(|_| ())
        .ok_or_else(|| invalid_argument(format!("readtable: unsupported Encoding '{label}'")))
}

fn encoding_for_label(label: &str) -> Option<&'static Encoding> {
    let label = label.trim();
    if label.is_empty()
        || label.eq_ignore_ascii_case("auto")
        || label.eq_ignore_ascii_case("default")
        || label.eq_ignore_ascii_case("system")
        || label.eq_ignore_ascii_case("native")
        || label.eq_ignore_ascii_case("utf-8")
        || label.eq_ignore_ascii_case("utf8")
        || label.eq_ignore_ascii_case("unicode")
    {
        return Some(UTF_8);
    }
    Encoding::for_label(label.as_bytes())
}

fn decode_text_bytes(bytes: &[u8], encoding: &str) -> BuiltinResult<String> {
    let (encoding, offset) = if encoding.trim().eq_ignore_ascii_case("auto") {
        Encoding::for_bom(bytes).unwrap_or((UTF_8, 0))
    } else {
        (
            encoding_for_label(encoding).ok_or_else(|| {
                invalid_argument(format!("readtable: unsupported Encoding '{encoding}'"))
            })?,
            0,
        )
    };
    let (decoded, _, had_errors) = encoding.decode(&bytes[offset..]);
    if had_errors {
        return Err(table_error(
            &TABLE_ERROR_IO,
            format!(
                "readtable: unable to decode file contents using encoding '{}'",
                encoding.name()
            ),
        ));
    }
    Ok(decoded.into_owned())
}

#[derive(Clone, Debug)]
enum ImportCell {
    Empty,
    Text(String),
    Number(f64),
    Logical(bool),
    DateTime(f64),
    Error(String),
}

impl ImportCell {
    fn from_text(text: String) -> Self {
        if text.trim().is_empty() {
            Self::Empty
        } else {
            Self::Text(text)
        }
    }

    fn display_text(&self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Text(text) => text.clone(),
            Self::Number(value) => format_key_number(*value),
            Self::Logical(value) => value.to_string(),
            Self::DateTime(serial) => format_key_number(*serial),
            Self::Error(text) => text.clone(),
        }
    }

    fn is_missing(&self, options: &ReadTableOptions) -> bool {
        match self {
            Self::Empty => true,
            Self::Text(text) => options.is_missing(text),
            _ => false,
        }
    }

    fn is_likely_data_token(&self, options: &ReadTableOptions) -> bool {
        match self {
            Self::Number(_) | Self::Logical(_) | Self::DateTime(_) => true,
            Self::Empty => false,
            Self::Text(text) => {
                let token = unquote(text.trim()).trim();
                options.is_missing(token)
                    || parse_numeric(token).is_some()
                    || parse_logical(token).is_some()
                    || parse_iso_datetime_to_datenum(token).is_some()
            }
            Self::Error(_) => true,
        }
    }
}

fn spreadsheet_cell_to_import(cell: &SpreadsheetData) -> ImportCell {
    match cell {
        SpreadsheetData::Empty => ImportCell::Empty,
        SpreadsheetData::Int(value) => ImportCell::Number(*value as f64),
        SpreadsheetData::Float(value) => ImportCell::Number(*value),
        SpreadsheetData::String(text) => ImportCell::Text(text.clone()),
        SpreadsheetData::Bool(value) => ImportCell::Logical(*value),
        SpreadsheetData::DateTime(value) => value
            .as_datetime()
            .map(crate::builtins::datetime::datenum_from_naive)
            .map(ImportCell::DateTime)
            .unwrap_or_else(|| ImportCell::Number(value.as_f64())),
        SpreadsheetData::DateTimeIso(text) => parse_iso_datetime_to_datenum(text)
            .map(ImportCell::DateTime)
            .unwrap_or_else(|| ImportCell::Text(text.clone())),
        SpreadsheetData::DurationIso(text) => ImportCell::Text(text.clone()),
        SpreadsheetData::Error(err) => ImportCell::Error(err.to_string()),
    }
}

fn spreadsheet_range_to_rows(
    range: &calamine::Range<SpreadsheetData>,
    options: &ReadTableOptions,
) -> BuiltinResult<Vec<Vec<ImportCell>>> {
    if range.is_empty() {
        return Ok(Vec::new());
    }
    let Some((range_start_row, range_start_col)) = range.start() else {
        return Ok(Vec::new());
    };
    let Some((range_end_row, range_end_col)) = range.end() else {
        return Ok(Vec::new());
    };
    let start_row = options
        .range
        .map(|spec| checked_u32(spec.start_row, "Range row"))
        .transpose()?
        .unwrap_or(range_start_row);
    let start_col = options
        .range
        .map(|spec| checked_u32(spec.start_col, "Range column"))
        .transpose()?
        .unwrap_or(range_start_col);
    let end_row = options
        .range
        .and_then(|spec| spec.end_row)
        .map(|row| checked_u32(row, "Range row"))
        .transpose()?
        .unwrap_or(range_end_row);
    let end_col = options
        .range
        .and_then(|spec| spec.end_col)
        .map(|col| checked_u32(col, "Range column"))
        .transpose()?
        .unwrap_or(range_end_col);
    if start_row > end_row || start_col > end_col {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for row_idx in start_row..=end_row {
        let mut row = Vec::new();
        for col_idx in start_col..=end_col {
            row.push(
                range
                    .get_value((row_idx, col_idx))
                    .map(spreadsheet_cell_to_import)
                    .unwrap_or(ImportCell::Empty),
            );
        }
        if matches!(options.empty_line_rule, EmptyLineRule::Skip)
            && row.iter().all(|cell| cell.is_missing(options))
        {
            continue;
        }
        rows.push(row);
    }
    if options.num_header_lines > 0 {
        Ok(rows.into_iter().skip(options.num_header_lines).collect())
    } else {
        Ok(rows)
    }
}

fn checked_u32(value: usize, context: &str) -> BuiltinResult<u32> {
    u32::try_from(value).map_err(|_| invalid_argument(format!("readtable: {context} overflow")))
}

fn detect_delimiter(lines: &[String]) -> Option<Delimiter> {
    let candidates = [',', '\t', ';', '|'];
    let mut best: Option<(f64, Delimiter)> = None;
    for candidate in candidates {
        let counts = lines
            .iter()
            .take(32)
            .filter(|line| line.contains(candidate))
            .map(|line| split_with_char_delim(line, candidate).len())
            .filter(|count| *count >= 2)
            .collect::<Vec<_>>();
        if counts.is_empty() {
            continue;
        }
        let avg = counts.iter().copied().sum::<usize>() as f64 / counts.len() as f64;
        if avg >= 2.0
            && best
                .as_ref()
                .map(|(best_avg, _)| avg > *best_avg)
                .unwrap_or(true)
        {
            best = Some((avg, Delimiter::Char(candidate)));
        }
    }
    best.map(|(_, delimiter)| delimiter).or_else(|| {
        lines
            .iter()
            .take(32)
            .any(|line| line.split_whitespace().count() > 1)
            .then_some(Delimiter::Whitespace)
    })
}

fn split_with_char_delim(line: &str, delimiter: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                current.push('"');
                chars.next();
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if ch == delimiter && !in_quotes {
            out.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    out.push(current);
    out
}

fn parse_text_records(
    text: &str,
    delimiter: &Delimiter,
    empty_line_rule: EmptyLineRule,
) -> Vec<Vec<ImportCell>> {
    match delimiter {
        Delimiter::Whitespace => parse_whitespace_records(text, empty_line_rule),
        Delimiter::Char(ch) => parse_delimited_records(text, &ch.to_string(), empty_line_rule),
        Delimiter::String(pattern) => parse_delimited_records(text, pattern, empty_line_rule),
    }
}

fn parse_delimited_records(
    text: &str,
    delimiter: &str,
    empty_line_rule: EmptyLineRule,
) -> Vec<Vec<ImportCell>> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut idx = 0usize;
    while idx < text.len() {
        let ch = text[idx..].chars().next().expect("valid char boundary");
        if ch == '"' {
            if in_quotes && text[idx + ch.len_utf8()..].starts_with('"') {
                current.push('"');
                idx += ch.len_utf8() + 1;
                continue;
            }
            in_quotes = !in_quotes;
            idx += ch.len_utf8();
            continue;
        }
        if !in_quotes && !delimiter.is_empty() && text[idx..].starts_with(delimiter) {
            row.push(ImportCell::from_text(std::mem::take(&mut current)));
            idx += delimiter.len();
            continue;
        }
        if !in_quotes && (ch == '\n' || ch == '\r') {
            row.push(ImportCell::from_text(std::mem::take(&mut current)));
            push_import_record(&mut records, std::mem::take(&mut row), empty_line_rule);
            idx += ch.len_utf8();
            if ch == '\r' && text[idx..].starts_with('\n') {
                idx += 1;
            }
            continue;
        }
        current.push(ch);
        idx += ch.len_utf8();
    }
    if !current.is_empty() || !row.is_empty() || text.ends_with(delimiter) {
        row.push(ImportCell::from_text(current));
        push_import_record(&mut records, row, empty_line_rule);
    }
    records
}

fn parse_whitespace_records(text: &str, empty_line_rule: EmptyLineRule) -> Vec<Vec<ImportCell>> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut field_open = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                current.push('"');
                chars.next();
            } else {
                in_quotes = !in_quotes;
            }
            field_open = true;
            continue;
        }
        if !in_quotes && (ch == '\n' || ch == '\r') {
            if field_open || !current.is_empty() {
                row.push(ImportCell::from_text(std::mem::take(&mut current)));
            }
            field_open = false;
            push_import_record(&mut records, std::mem::take(&mut row), empty_line_rule);
            if ch == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
            }
            continue;
        }
        if !in_quotes && ch.is_whitespace() {
            if field_open || !current.is_empty() {
                row.push(ImportCell::from_text(std::mem::take(&mut current)));
                field_open = false;
            }
            continue;
        }
        current.push(ch);
        field_open = true;
    }
    if field_open || !current.is_empty() {
        row.push(ImportCell::from_text(current));
    }
    if !row.is_empty() {
        push_import_record(&mut records, row, empty_line_rule);
    }
    records
}

fn push_import_record(
    records: &mut Vec<Vec<ImportCell>>,
    row: Vec<ImportCell>,
    empty_line_rule: EmptyLineRule,
) {
    if matches!(empty_line_rule, EmptyLineRule::Skip)
        && row.iter().all(|cell| matches!(cell, ImportCell::Empty))
    {
        return;
    }
    records.push(row);
}

fn apply_import_range(rows: Vec<Vec<ImportCell>>, range: RangeSpec) -> Vec<Vec<ImportCell>> {
    if rows.is_empty() {
        return rows;
    }
    let end_row = range
        .end_row
        .unwrap_or_else(|| rows.len().saturating_sub(1));
    let max_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let end_col = range.end_col.unwrap_or_else(|| max_cols.saturating_sub(1));
    rows.into_iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            if idx < range.start_row || idx > end_row {
                return None;
            }
            let selected = (range.start_col..=end_col)
                .map(|col| row.get(col).cloned().unwrap_or(ImportCell::Empty))
                .collect::<Vec<_>>();
            Some(selected)
        })
        .collect()
}

fn import_rows_to_table(
    mut rows: Vec<Vec<ImportCell>>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut variable_names = options.variable_names.clone();
    let read_variable_names = options
        .read_variable_names
        .unwrap_or_else(|| variable_names.is_none() && should_read_variable_names(&rows, options));
    if variable_names.is_none() && read_variable_names && !rows.is_empty() {
        variable_names = Some(
            rows.remove(0)
                .into_iter()
                .map(|cell| cell.display_text())
                .collect(),
        );
    }

    let mut row_names = options.row_names.clone();
    if options.read_row_names && !rows.is_empty() {
        row_names = Some(
            rows.iter_mut()
                .map(|row| {
                    if row.is_empty() {
                        String::new()
                    } else {
                        row.remove(0).display_text()
                    }
                })
                .collect(),
        );
        if let Some(names) = variable_names.as_mut() {
            if !names.is_empty() {
                names.remove(0);
            }
        }
    }

    let column_count = import_column_count(&rows, &variable_names, options)?;
    let names = import_variable_names(variable_names, column_count, options);

    let mut columns = Vec::with_capacity(names.len());
    for col in 0..names.len() {
        let values = rows
            .iter()
            .map(|row| row.get(col).cloned().unwrap_or(ImportCell::Empty))
            .collect::<Vec<_>>();
        let requested_type = options
            .variable_types
            .as_ref()
            .and_then(|types| types.get(col))
            .copied();
        columns.push(import_column(values, options, requested_type)?);
    }
    table_from_columns_with_properties(names, columns, row_names)
}

fn import_column_count(
    rows: &[Vec<ImportCell>],
    variable_names: &Option<Vec<String>>,
    options: &ReadTableOptions,
) -> BuiltinResult<usize> {
    let data_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let name_cols = variable_names.as_ref().map(Vec::len).unwrap_or(0);
    let type_cols = options.variable_types.as_ref().map(Vec::len).unwrap_or(0);
    if let Some(count) = options.num_variables {
        if name_cols > count {
            return Err(invalid_argument(
                "readtable: VariableNames length exceeds NumVariables",
            ));
        }
        if type_cols > count {
            return Err(invalid_argument(
                "readtable: VariableTypes length exceeds NumVariables",
            ));
        }
        return Ok(count);
    }
    Ok(data_cols.max(name_cols).max(type_cols))
}

fn import_variable_names(
    variable_names: Option<Vec<String>>,
    column_count: usize,
    options: &ReadTableOptions,
) -> Vec<String> {
    match variable_names {
        Some(mut names) => {
            while names.len() < column_count {
                names.push(format!("Var{}", names.len() + 1));
            }
            names.truncate(column_count);
            if options.preserve_variable_names {
                make_unique_names(names)
            } else {
                make_unique_variable_names(names)
            }
        }
        None => generated_variable_names(column_count),
    }
}

fn should_read_variable_names(rows: &[Vec<ImportCell>], options: &ReadTableOptions) -> bool {
    let Some(first) = rows.first() else {
        return false;
    };
    if first.is_empty() {
        return false;
    }
    let names = first
        .iter()
        .map(ImportCell::display_text)
        .map(|text| text.trim().to_string())
        .collect::<Vec<_>>();
    if names.iter().any(|name| name.is_empty()) {
        return false;
    }
    if first.iter().all(|cell| cell.is_likely_data_token(options)) {
        return false;
    }
    true
}

fn import_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    requested_type: Option<ImportVariableType>,
) -> BuiltinResult<Value> {
    match requested_type.unwrap_or(ImportVariableType::Auto) {
        ImportVariableType::Auto => infer_import_column(values, options),
        ImportVariableType::Numeric(dtype) => import_numeric_column(values, options, dtype),
        ImportVariableType::Logical => import_logical_column(values, options),
        ImportVariableType::Text(kind) => import_text_column(values, options, kind),
        ImportVariableType::CellStr => import_cellstr_column(values, options),
        ImportVariableType::Datetime => import_datetime_column(values, options),
        ImportVariableType::Duration => import_duration_column(values, options),
    }
}

fn import_numeric_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    let mut numeric = Vec::with_capacity(values.len());
    for value in &values {
        let parsed = numeric_from_import_cell(value, options, dtype.class_name())?;
        numeric.push(cast_import_numeric(parsed, dtype));
    }
    Tensor::new_with_dtype(numeric, vec![values.len(), 1], dtype)
        .map(Value::Tensor)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

fn numeric_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
    context: &str,
) -> BuiltinResult<f64> {
    match value {
        ImportCell::Empty => Ok(f64::NAN),
        ImportCell::Number(value) => Ok(*value),
        ImportCell::Logical(value) => Ok(if *value { 1.0 } else { 0.0 }),
        ImportCell::DateTime(serial) => Ok(*serial),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Ok(f64::NAN)
            } else {
                parse_numeric(token).ok_or_else(|| {
                    invalid_variable(format!("readtable: cannot import '{token}' as {context}"))
                })
            }
        }
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as {context}"
        ))),
    }
}

fn cast_import_numeric(value: f64, dtype: NumericDType) -> f64 {
    match dtype {
        NumericDType::F64 => value,
        NumericDType::F32 => (value as f32) as f64,
        NumericDType::U8 => {
            if value.is_finite() {
                value.round().clamp(0.0, u8::MAX as f64)
            } else {
                0.0
            }
        }
        NumericDType::U16 => {
            if value.is_finite() {
                value.round().clamp(0.0, u16::MAX as f64)
            } else {
                0.0
            }
        }
    }
}

fn import_logical_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut logical = Vec::with_capacity(values.len());
    for value in &values {
        logical.push(logical_from_import_cell(value, options)?);
    }
    LogicalArray::new(logical, vec![values.len(), 1])
        .map(Value::LogicalArray)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

fn logical_from_import_cell(value: &ImportCell, options: &ReadTableOptions) -> BuiltinResult<u8> {
    let flag = match value {
        ImportCell::Empty => false,
        ImportCell::Logical(value) => *value,
        ImportCell::Number(value) => *value != 0.0,
        ImportCell::DateTime(serial) => *serial != 0.0,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                false
            } else if let Some(value) = parse_logical(token) {
                value
            } else if let Some(value) = parse_numeric(token) {
                value != 0.0
            } else {
                return Err(invalid_variable(format!(
                    "readtable: cannot import '{token}' as logical"
                )));
            }
        }
        ImportCell::Error(text) => {
            return Err(invalid_variable(format!(
                "readtable: cannot import spreadsheet error '{text}' as logical"
            )));
        }
    };
    Ok(u8::from(flag))
}

fn import_text_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    kind: TextImportType,
) -> BuiltinResult<Value> {
    let strings = import_text_values(values, options);
    match kind {
        TextImportType::String => StringArray::new(strings.clone(), vec![strings.len(), 1])
            .map(Value::StringArray)
            .map_err(|err| invalid_variable(format!("readtable: {err}"))),
        TextImportType::Char => import_char_column(strings),
    }
}

fn import_text_values(values: Vec<ImportCell>, options: &ReadTableOptions) -> Vec<String> {
    values
        .into_iter()
        .map(|value| {
            if value.is_missing(options) {
                String::new()
            } else {
                unquote(value.display_text().trim()).to_string()
            }
        })
        .collect()
}

fn import_char_column(strings: Vec<String>) -> BuiltinResult<Value> {
    let rows = strings.len();
    let cols = strings
        .iter()
        .map(|text| text.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = vec![' '; rows * cols];
    for (row, text) in strings.iter().enumerate() {
        for (col, ch) in text.chars().enumerate() {
            data[row * cols + col] = ch;
        }
    }
    CharArray::new(data, rows, cols)
        .map(Value::CharArray)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

fn import_cellstr_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let strings = import_text_values(values, options);
    let rows = strings.len();
    let cells = strings
        .into_iter()
        .map(|text| Value::CharArray(CharArray::new_row(&text)))
        .collect::<Vec<_>>();
    CellArray::new(cells, rows, 1)
        .map(Value::Cell)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

fn import_datetime_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    if matches!(options.datetime_type, DatetimeImportType::Text) {
        return import_text_column(values, options, options.text_type);
    }

    let mut serials = Vec::with_capacity(values.len());
    for value in &values {
        serials.push(datetime_serial_from_import_cell(value, options)?);
    }
    let tensor = Tensor::new(serials, vec![values.len(), 1])
        .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
    if matches!(options.datetime_type, DatetimeImportType::ExcelDatenum) {
        Ok(Value::Tensor(tensor))
    } else {
        crate::builtins::datetime::datetime_object_from_serial_tensor(tensor, "yyyy-MM-dd HH:mm:ss")
    }
}

fn datetime_serial_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> BuiltinResult<f64> {
    match value {
        ImportCell::Empty => Ok(f64::NAN),
        ImportCell::DateTime(serial) => Ok(*serial),
        ImportCell::Number(value) => Ok(*value),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Ok(f64::NAN)
            } else if let Some(serial) = parse_iso_datetime_to_datenum(token) {
                Ok(serial)
            } else if let Some(serial) = parse_numeric(token) {
                Ok(serial)
            } else {
                Err(invalid_variable(format!(
                    "readtable: cannot import '{token}' as datetime"
                )))
            }
        }
        ImportCell::Logical(_) => Err(invalid_variable(
            "readtable: cannot import logical value as datetime",
        )),
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as datetime"
        ))),
    }
}

fn import_duration_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut days = Vec::with_capacity(values.len());
    for value in &values {
        days.push(duration_days_from_import_cell(value, options)?);
    }
    let tensor = Tensor::new(days, vec![values.len(), 1])
        .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
    crate::builtins::duration::duration_object_from_days_tensor(
        tensor,
        crate::builtins::duration::DEFAULT_DURATION_FORMAT,
    )
}

fn duration_days_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> BuiltinResult<f64> {
    match value {
        ImportCell::Empty => Ok(f64::NAN),
        ImportCell::Number(value) => Ok(*value),
        ImportCell::Logical(value) => Ok(if *value { 1.0 } else { 0.0 }),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Ok(f64::NAN)
            } else {
                parse_duration_to_days(token).ok_or_else(|| {
                    invalid_variable(format!("readtable: cannot import '{token}' as duration"))
                })
            }
        }
        ImportCell::DateTime(_) => Err(invalid_variable(
            "readtable: cannot import datetime value as duration",
        )),
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as duration"
        ))),
    }
}

fn infer_import_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut numeric = Vec::with_capacity(values.len());
    let mut all_numeric = true;
    for value in &values {
        match value {
            ImportCell::Empty => numeric.push(f64::NAN),
            ImportCell::Number(value) => numeric.push(*value),
            ImportCell::Text(text) => {
                let token = unquote(text.trim()).trim();
                if options.is_missing(token) {
                    numeric.push(f64::NAN);
                } else if let Some(value) = parse_numeric(token) {
                    numeric.push(value);
                } else {
                    all_numeric = false;
                    break;
                }
            }
            _ => {
                all_numeric = false;
                break;
            }
        }
    }
    if all_numeric {
        return Tensor::new(numeric, vec![values.len(), 1])
            .map(Value::Tensor)
            .map_err(|err| invalid_variable(format!("readtable: {err}")));
    }

    let mut logical = Vec::with_capacity(values.len());
    let mut all_logical = true;
    for value in &values {
        match value {
            ImportCell::Empty => logical.push(0),
            ImportCell::Logical(value) => logical.push(i32::from(*value) as u8),
            ImportCell::Text(text) => {
                let token = unquote(text.trim()).trim();
                if options.is_missing(token) {
                    logical.push(0);
                } else if let Some(value) = parse_logical(token) {
                    logical.push(i32::from(value) as u8);
                } else {
                    all_logical = false;
                    break;
                }
            }
            _ => {
                all_logical = false;
                break;
            }
        }
    }
    if all_logical {
        return LogicalArray::new(logical, vec![values.len(), 1])
            .map(Value::LogicalArray)
            .map_err(|err| invalid_variable(format!("readtable: {err}")));
    }

    if !matches!(options.datetime_type, DatetimeImportType::Text) {
        let mut serials = Vec::with_capacity(values.len());
        let mut all_datetime = true;
        for value in &values {
            match value {
                ImportCell::Empty => serials.push(f64::NAN),
                ImportCell::DateTime(serial) => serials.push(*serial),
                ImportCell::Text(text) => {
                    let token = unquote(text.trim()).trim();
                    if options.is_missing(token) {
                        serials.push(f64::NAN);
                    } else if let Some(serial) = parse_iso_datetime_to_datenum(token) {
                        serials.push(serial);
                    } else {
                        all_datetime = false;
                        break;
                    }
                }
                _ => {
                    all_datetime = false;
                    break;
                }
            }
        }
        if all_datetime {
            let tensor = Tensor::new(serials, vec![values.len(), 1])
                .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
            if matches!(options.datetime_type, DatetimeImportType::ExcelDatenum) {
                return Ok(Value::Tensor(tensor));
            }
            return crate::builtins::datetime::datetime_object_from_serial_tensor(
                tensor,
                "yyyy-MM-dd HH:mm:ss",
            );
        }
    }

    import_text_column(values, options, options.text_type)
}

fn parse_numeric(token: &str) -> Option<f64> {
    match token.to_ascii_lowercase().as_str() {
        "nan" => Some(f64::NAN),
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => token.parse::<f64>().ok(),
    }
}

fn parse_logical(token: &str) -> Option<bool> {
    match token.to_ascii_lowercase().as_str() {
        "true" | "t" | "yes" | "on" => Some(true),
        "false" | "f" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_duration_to_days(token: &str) -> Option<f64> {
    parse_numeric(token).or_else(|| parse_clock_duration_to_days(token))
}

fn parse_clock_duration_to_days(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (sign, body) = if let Some(rest) = trimmed.strip_prefix('-') {
        (-1.0, rest)
    } else if let Some(rest) = trimmed.strip_prefix('+') {
        (1.0, rest)
    } else {
        (1.0, trimmed)
    };
    let parts = body.split(':').collect::<Vec<_>>();
    let (hours, minutes, seconds) = match parts.as_slice() {
        [hours, minutes] => (
            hours.parse::<f64>().ok()?,
            minutes.parse::<f64>().ok()?,
            0.0,
        ),
        [hours, minutes, seconds] => (
            hours.parse::<f64>().ok()?,
            minutes.parse::<f64>().ok()?,
            seconds.parse::<f64>().ok()?,
        ),
        _ => return None,
    };
    if !hours.is_finite()
        || !minutes.is_finite()
        || !seconds.is_finite()
        || !(0.0..60.0).contains(&minutes)
        || !(0.0..60.0).contains(&seconds)
    {
        return None;
    }
    Some(sign * (hours * 3600.0 + minutes * 60.0 + seconds) / 86_400.0)
}

fn parse_iso_datetime_to_datenum(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S%.f",
        "%m/%d/%Y %H:%M:%S%.f",
    ] {
        if let Ok(value) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Some(crate::builtins::datetime::datenum_from_naive(value));
        }
    }
    for format in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"] {
        if let Ok(date) = NaiveDate::parse_from_str(trimmed, format) {
            return Some(crate::builtins::datetime::datenum_from_naive(
                date.and_time(NaiveTime::MIN),
            ));
        }
    }
    None
}

fn unquote(token: &str) -> &str {
    if token.len() >= 2 {
        let bytes = token.as_bytes();
        if (bytes[0] == b'"' && bytes[token.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[token.len() - 1] == b'\'')
        {
            return &token[1..token.len() - 1];
        }
    }
    token
}

fn default_properties(variable_names: Vec<String>, row_names: Option<Vec<String>>) -> StructValue {
    let mut props = StructValue::new();
    props.insert(
        VARIABLE_NAMES,
        Value::StringArray(
            StringArray::new(variable_names.clone(), vec![1, variable_names.len()])
                .expect("VariableNames shape is valid"),
        ),
    );
    props.insert(
        ROW_NAMES,
        row_names
            .map(|names| {
                Value::StringArray(
                    StringArray::new(names.clone(), vec![names.len(), 1])
                        .expect("RowNames shape is valid"),
                )
            })
            .unwrap_or_else(|| {
                Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap())
            }),
    );
    props.insert(
        DIMENSION_NAMES,
        Value::StringArray(
            StringArray::new(
                vec![
                    DEFAULT_ROW_DIM_NAME.to_string(),
                    DEFAULT_VARIABLE_DIM_NAME.to_string(),
                ],
                vec![1, 2],
            )
            .expect("DimensionNames shape is valid"),
        ),
    );
    props.insert(
        VARIABLE_UNITS,
        Value::StringArray(
            StringArray::new(
                vec![String::new(); variable_names.len()],
                vec![1, variable_names.len()],
            )
            .expect("VariableUnits shape is valid"),
        ),
    );
    props.insert(
        VARIABLE_DESCRIPTIONS,
        Value::StringArray(
            StringArray::new(
                vec![String::new(); variable_names.len()],
                vec![1, variable_names.len()],
            )
            .expect("VariableDescriptions shape is valid"),
        ),
    );
    props.insert(DESCRIPTION, Value::String(String::new()));
    props.insert(USER_DATA, Value::Tensor(Tensor::zeros(vec![0, 0])));
    props
}

pub fn table_from_columns(names: Vec<String>, columns: Vec<Value>) -> BuiltinResult<Value> {
    table_from_columns_with_properties(names, columns, None)
}

fn table_from_columns_with_properties(
    names: Vec<String>,
    columns: Vec<Value>,
    row_names: Option<Vec<String>>,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if names.len() != columns.len() {
        return Err(invalid_variable(
            "table: number of variable names must match number of variables",
        ));
    }
    let names = make_unique_names(names);
    let height = validate_column_heights(&names, &columns)?;
    if let Some(row_names) = &row_names {
        if row_names.len() != height {
            return Err(invalid_variable(
                "table: number of row names must match table height",
            ));
        }
    }
    let mut variables = StructValue::new();
    for (name, value) in names.iter().cloned().zip(columns) {
        variables.insert(name, value);
    }
    let props = default_properties(names, row_names);
    let mut object = ObjectInstance::new(TABLE_CLASS.to_string());
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(variables));
    object.properties.insert(
        TABLE_PROPERTIES_FIELD.to_string(),
        Value::Struct(props.clone()),
    );
    object
        .properties
        .insert(PROPERTIES_MEMBER.to_string(), Value::Struct(props));
    Ok(Value::Object(object))
}

fn validate_column_heights(names: &[String], columns: &[Value]) -> BuiltinResult<usize> {
    if columns.is_empty() {
        return Ok(0);
    }
    let height = value_row_count(&columns[0])?;
    for (name, value) in names.iter().zip(columns) {
        let rows = value_row_count(value)?;
        if rows != height {
            return Err(invalid_variable(format!(
                "table: variable '{name}' has {rows} rows but expected {height}"
            )));
        }
    }
    Ok(height)
}

pub fn is_table_value(value: &Value) -> bool {
    table_object(value).is_some()
}

fn table_object(value: &Value) -> Option<&ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(TABLE_CLASS) => Some(object),
        _ => None,
    }
}

fn into_table_object(value: Value, context: &str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(TABLE_CLASS) => Ok(object),
        other => Err(invalid_argument(format!(
            "{context}: expected table, got {other:?}"
        ))),
    }
}

pub fn table_variables(object: &ObjectInstance) -> BuiltinResult<StructValue> {
    match object.properties.get(TABLE_VARIABLES_FIELD) {
        Some(Value::Struct(st)) => Ok(st.clone()),
        Some(other) => Err(invalid_variable(format!(
            "table: invalid internal variable storage {other:?}"
        ))),
        None => Ok(StructValue::new()),
    }
}

pub fn table_variable_names_from_object(object: &ObjectInstance) -> BuiltinResult<Vec<String>> {
    let variables = table_variables(object)?;
    Ok(variables.fields.keys().cloned().collect())
}

pub fn table_height(object: &ObjectInstance) -> BuiltinResult<usize> {
    let variables = table_variables(object)?;
    match variables.fields.values().next() {
        Some(value) => value_row_count(value),
        None => Ok(0),
    }
}

pub fn table_width(object: &ObjectInstance) -> BuiltinResult<usize> {
    table_variables(object).map(|vars| vars.fields.len())
}

fn table_public_properties(object: &ObjectInstance) -> BuiltinResult<StructValue> {
    match object
        .properties
        .get(TABLE_PROPERTIES_FIELD)
        .or_else(|| object.properties.get(PROPERTIES_MEMBER))
    {
        Some(Value::Struct(st)) => Ok(st.clone()),
        Some(other) => Err(invalid_variable(format!(
            "table: invalid Properties storage {other:?}"
        ))),
        None => Ok(default_properties(
            table_variable_names_from_object(object)?,
            None,
        )),
    }
}

fn sync_table_properties(object: &mut ObjectInstance, props: StructValue) {
    object.properties.insert(
        TABLE_PROPERTIES_FIELD.to_string(),
        Value::Struct(props.clone()),
    );
    object
        .properties
        .insert(PROPERTIES_MEMBER.to_string(), Value::Struct(props));
}

fn table_member_get(object: &ObjectInstance, payload: &Value) -> BuiltinResult<Value> {
    let name = scalar_text(payload, "table member")?;
    if name == PROPERTIES_MEMBER {
        return Ok(Value::Struct(table_public_properties(object)?));
    }
    let variables = table_variables(object)?;
    variables
        .fields
        .get(&name)
        .cloned()
        .ok_or_else(|| invalid_variable(format!("table: unrecognized variable '{name}'")))
}

fn table_member_set(object: &mut ObjectInstance, field: &str, rhs: Value) -> BuiltinResult<()> {
    if field == PROPERTIES_MEMBER {
        let Value::Struct(props) = rhs else {
            return Err(invalid_variable(
                "table: Properties assignment expects a scalar struct",
            ));
        };
        apply_properties(object, props)?;
        return Ok(());
    }
    let mut variables = table_variables(object)?;
    let mut names = table_variable_names_from_object(object)?;
    let height = table_height(object)?;
    let rhs_rows = value_row_count(&rhs)?;
    if !variables.fields.is_empty() && rhs_rows != height {
        return Err(invalid_variable(format!(
            "table: variable '{field}' has {rhs_rows} rows but table has {height}"
        )));
    }
    if !variables.fields.contains_key(field) {
        names.push(field.to_string());
    }
    variables.insert(field.to_string(), rhs);
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(variables));
    let mut props = table_public_properties(object)?;
    update_variable_metadata_names(&mut props, names)?;
    sync_table_properties(object, props);
    Ok(())
}

fn apply_properties(object: &mut ObjectInstance, mut props: StructValue) -> BuiltinResult<()> {
    if let Some(value) = props.fields.get(VARIABLE_NAMES) {
        let names = variable_name_list(value)?;
        rename_table_variables(object, names.clone())?;
        update_variable_metadata_names(&mut props, names)?;
    }
    sync_table_properties(object, props);
    Ok(())
}

fn rename_table_variables(
    object: &mut ObjectInstance,
    new_names: Vec<String>,
) -> BuiltinResult<()> {
    let old_names = table_variable_names_from_object(object)?;
    if old_names.len() != new_names.len() {
        return Err(invalid_variable(
            "table: VariableNames assignment must preserve variable count",
        ));
    }
    let new_names = make_unique_variable_names(new_names);
    let variables = table_variables(object)?;
    let mut renamed = StructValue::new();
    for (old, new) in old_names.iter().zip(new_names.iter()) {
        let value = variables
            .fields
            .get(old)
            .cloned()
            .ok_or_else(|| invalid_variable(format!("table: missing variable '{old}'")))?;
        renamed.insert(new.clone(), value);
    }
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(renamed));
    Ok(())
}

fn update_variable_metadata_names(
    props: &mut StructValue,
    names: Vec<String>,
) -> BuiltinResult<()> {
    props.insert(
        VARIABLE_NAMES,
        Value::StringArray(
            StringArray::new(names.clone(), vec![1, names.len()])
                .map_err(|err| invalid_variable(format!("table: {err}")))?,
        ),
    );
    for field in [VARIABLE_UNITS, VARIABLE_DESCRIPTIONS] {
        let existing = props.fields.get(field).cloned();
        let values = match existing {
            Some(Value::StringArray(mut array)) => {
                array.data.resize(names.len(), String::new());
                array.data.truncate(names.len());
                array.data
            }
            _ => vec![String::new(); names.len()],
        };
        props.insert(
            field,
            Value::StringArray(
                StringArray::new(values, vec![1, names.len()])
                    .map_err(|err| invalid_variable(format!("table: {err}")))?,
            ),
        );
    }
    Ok(())
}

fn table_paren_get(object: &ObjectInstance, payload: &Value) -> BuiltinResult<Value> {
    let selectors = selector_values(payload)?;
    let rows = parse_row_selector(selectors.first(), table_height(object)?)?;
    let variable_names = table_variable_names_from_object(object)?;
    let selected_names = parse_variable_selector(selectors.get(1), &variable_names)?;
    let variables = table_variables(object)?;
    let mut out = Vec::with_capacity(selected_names.len());
    for name in &selected_names {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable(format!("table: missing variable '{name}'")))?;
        out.push(select_rows(value, &rows)?);
    }
    let row_names = selected_row_names(object, &rows)?;
    table_from_columns_with_properties(selected_names, out, row_names)
}

fn table_brace_get(object: &ObjectInstance, payload: &Value) -> BuiltinResult<Value> {
    let subset = table_paren_get(object, payload)?;
    let object = into_table_object(subset, "table brace indexing")?;
    let variables = table_variables(&object)?;
    if variables.fields.len() == 1 {
        return variables
            .fields
            .values()
            .next()
            .cloned()
            .ok_or_else(|| invalid_variable("table: missing selected variable"));
    }
    let values = variables.fields.values().collect::<Vec<_>>();
    if values.iter().all(|value| matches!(value, Value::Tensor(_))) {
        return concatenate_numeric_columns(&values);
    }
    CellArray::new(
        values.into_iter().cloned().collect(),
        1,
        variables.fields.len(),
    )
    .map(Value::Cell)
    .map_err(|err| invalid_variable(format!("table: {err}")))
}

fn table_paren_assign(
    mut object: ObjectInstance,
    payload: &Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let rhs_table = into_table_object(rhs, "table paren assignment")?;
    let selectors = selector_values(payload)?;
    let rows = parse_row_selector(selectors.first(), table_height(&object)?)?;
    let variable_names = table_variable_names_from_object(&object)?;
    let selected_names = parse_variable_selector(selectors.get(1), &variable_names)?;
    let rhs_names = table_variable_names_from_object(&rhs_table)?;
    if selected_names.len() != rhs_names.len() {
        return Err(invalid_variable(
            "table: assignment variable count must match selected variables",
        ));
    }
    let mut variables = table_variables(&object)?;
    let rhs_variables = table_variables(&rhs_table)?;
    for (target_name, rhs_name) in selected_names.iter().zip(rhs_names.iter()) {
        let current =
            variables.fields.get(target_name).cloned().ok_or_else(|| {
                invalid_variable(format!("table: missing variable '{target_name}'"))
            })?;
        let rhs_col =
            rhs_variables.fields.get(rhs_name).cloned().ok_or_else(|| {
                invalid_variable(format!("table: missing rhs variable '{rhs_name}'"))
            })?;
        variables.insert(target_name.clone(), assign_rows(current, &rows, rhs_col)?);
    }
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(variables));
    Ok(Value::Object(object))
}

fn table_brace_assign(
    mut object: ObjectInstance,
    payload: &Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let selectors = selector_values(payload)?;
    let rows = parse_row_selector(selectors.first(), table_height(&object)?)?;
    let variable_names = table_variable_names_from_object(&object)?;
    let selected_names = parse_variable_selector(selectors.get(1), &variable_names)?;
    if selected_names.len() != 1 {
        return Err(invalid_variable(
            "table: brace assignment supports one variable at a time",
        ));
    }
    let mut variables = table_variables(&object)?;
    let target = selected_names[0].clone();
    let current = variables
        .fields
        .get(&target)
        .cloned()
        .ok_or_else(|| invalid_variable(format!("table: missing variable '{target}'")))?;
    variables.insert(target, assign_rows(current, &rows, rhs)?);
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(variables));
    Ok(Value::Object(object))
}

fn selector_values(payload: &Value) -> BuiltinResult<Vec<Value>> {
    match payload {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for handle in &cell.data {
                out.push(handle.clone());
            }
            Ok(out)
        }
        other => Ok(vec![other.clone()]),
    }
}

fn parse_row_selector(selector: Option<&Value>, height: usize) -> BuiltinResult<Vec<usize>> {
    let Some(selector) = selector else {
        return Ok((0..height).collect());
    };
    if is_colon_selector(selector) {
        return Ok((0..height).collect());
    }
    if is_end_selector(selector) {
        return if height == 0 {
            Err(invalid_index(
                "table: end row index is invalid for empty table",
            ))
        } else {
            Ok(vec![height - 1])
        };
    }
    match selector {
        Value::Num(n) => Ok(vec![one_based_to_zero(*n, height, "row")?]),
        Value::Int(i) => Ok(vec![one_based_to_zero(i.to_f64(), height, "row")?]),
        Value::Tensor(tensor) => tensor
            .data
            .iter()
            .map(|value| one_based_to_zero(*value, height, "row"))
            .collect(),
        Value::LogicalArray(array) => {
            if array.data.len() != height {
                return Err(invalid_index(
                    "table: logical row selector length must match table height",
                ));
            }
            Ok(array
                .data
                .iter()
                .enumerate()
                .filter_map(|(idx, value)| (*value != 0).then_some(idx))
                .collect())
        }
        other => Err(invalid_index(format!(
            "table: unsupported row selector {other:?}"
        ))),
    }
}

fn parse_variable_selector(
    selector: Option<&Value>,
    names: &[String],
) -> BuiltinResult<Vec<String>> {
    let Some(selector) = selector else {
        return Ok(names.to_vec());
    };
    if is_colon_selector(selector) {
        return Ok(names.to_vec());
    }
    match selector {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) | Value::Cell(_) => {
            let selected = string_list(selector)?;
            for name in &selected {
                if !names.contains(name) {
                    return Err(invalid_variable(format!(
                        "table: unrecognized variable '{name}'"
                    )));
                }
            }
            Ok(selected)
        }
        Value::Num(n) => Ok(vec![name_at_index(names, *n)?]),
        Value::Int(i) => Ok(vec![name_at_index(names, i.to_f64())?]),
        Value::Tensor(tensor) => tensor
            .data
            .iter()
            .map(|value| name_at_index(names, *value))
            .collect(),
        Value::LogicalArray(array) => {
            if array.data.len() != names.len() {
                return Err(invalid_index(
                    "table: logical variable selector length must match table width",
                ));
            }
            Ok(array
                .data
                .iter()
                .zip(names.iter())
                .filter_map(|(flag, name)| (*flag != 0).then_some(name.clone()))
                .collect())
        }
        other => Err(invalid_index(format!(
            "table: unsupported variable selector {other:?}"
        ))),
    }
}

fn is_colon_selector(value: &Value) -> bool {
    scalar_text(value, "selector")
        .map(|text| text == ":")
        .unwrap_or(false)
}

fn is_end_selector(value: &Value) -> bool {
    scalar_text(value, "selector")
        .map(|text| text == "end")
        .unwrap_or(false)
}

fn name_at_index(names: &[String], value: f64) -> BuiltinResult<String> {
    let idx = one_based_to_zero(value, names.len(), "variable")?;
    Ok(names[idx].clone())
}

fn one_based_to_zero(value: f64, len: usize, context: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(invalid_index(format!(
            "table: {context} indices must be positive finite integers"
        )));
    }
    let idx = value.round() as usize - 1;
    if idx >= len {
        return Err(invalid_index(format!(
            "table: {context} index exceeds bounds"
        )));
    }
    Ok(idx)
}

fn selected_row_names(
    object: &ObjectInstance,
    rows: &[usize],
) -> BuiltinResult<Option<Vec<String>>> {
    let props = table_public_properties(object)?;
    let Some(value) = props.fields.get(ROW_NAMES) else {
        return Ok(None);
    };
    let names = string_list(value)?;
    if names.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        rows.iter()
            .filter_map(|row| names.get(*row).cloned())
            .collect(),
    ))
}

fn value_row_count(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.rows()),
        Value::ComplexTensor(tensor) => Ok(tensor.rows),
        Value::StringArray(array) => Ok(array.rows()),
        Value::LogicalArray(array) => Ok(array.shape.first().copied().unwrap_or(array.data.len())),
        Value::Cell(cell) => Ok(cell.rows),
        Value::CharArray(array) => Ok(array.rows),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .map(|tensor| tensor.rows())
        }
        Value::Object(obj) if obj.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .map(|tensor| tensor.rows())
        }
        Value::Object(obj) if obj.is_class(TABLE_CLASS) => table_height(obj),
        _ => Ok(1),
    }
}

fn select_rows(value: &Value, rows: &[usize]) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            let cols = tensor.cols();
            let mut data = Vec::with_capacity(rows.len() * cols);
            for col in 0..cols {
                for &row in rows {
                    data.push(tensor.get2(row, col).map_err(invalid_index)?);
                }
            }
            Tensor::new_with_dtype(data, vec![rows.len(), cols], tensor.dtype)
                .map(Value::Tensor)
                .map_err(invalid_variable)
        }
        Value::ComplexTensor(tensor) => {
            let mut data = Vec::with_capacity(rows.len() * tensor.cols);
            for col in 0..tensor.cols {
                for &row in rows {
                    let idx = row + col * tensor.rows;
                    data.push(*tensor.data.get(idx).ok_or_else(|| {
                        invalid_index("table: complex variable row index out of bounds")
                    })?);
                }
            }
            ComplexTensor::new(data, vec![rows.len(), tensor.cols])
                .map(Value::ComplexTensor)
                .map_err(invalid_variable)
        }
        Value::StringArray(array) => {
            let cols = array.cols();
            let mut data = Vec::with_capacity(rows.len() * cols);
            for col in 0..cols {
                for &row in rows {
                    let idx = row + col * array.rows();
                    data.push(array.data.get(idx).cloned().ok_or_else(|| {
                        invalid_index("table: string variable row index out of bounds")
                    })?);
                }
            }
            StringArray::new(data, vec![rows.len(), cols])
                .map(Value::StringArray)
                .map_err(invalid_variable)
        }
        Value::CharArray(array) => {
            let mut data = Vec::with_capacity(rows.len() * array.cols);
            for &row in rows {
                if row >= array.rows {
                    return Err(invalid_index(
                        "table: char variable row index out of bounds",
                    ));
                }
                let start = row * array.cols;
                data.extend_from_slice(&array.data[start..start + array.cols]);
            }
            CharArray::new(data, rows.len(), array.cols)
                .map(Value::CharArray)
                .map_err(invalid_variable)
        }
        Value::LogicalArray(array) => {
            let source_rows = array.shape.first().copied().unwrap_or(array.data.len());
            let cols = array.shape.get(1).copied().unwrap_or(1);
            let mut data = Vec::with_capacity(rows.len() * cols);
            for col in 0..cols {
                for &row in rows {
                    let idx = row + col * source_rows;
                    data.push(*array.data.get(idx).ok_or_else(|| {
                        invalid_index("table: logical variable row index out of bounds")
                    })?);
                }
            }
            LogicalArray::new(data, vec![rows.len(), cols])
                .map(Value::LogicalArray)
                .map_err(invalid_variable)
        }
        Value::Cell(cell) => {
            let mut data = Vec::with_capacity(rows.len() * cell.cols);
            for col in 0..cell.cols {
                for &row in rows {
                    data.push(cell.get(row, col).map_err(invalid_index)?);
                }
            }
            CellArray::new(data, rows.len(), cell.cols)
                .map(Value::Cell)
                .map_err(invalid_variable)
        }
        Value::Object(obj) if obj.is_class("datetime") => {
            let tensor = crate::builtins::datetime::serials_from_datetime_value(value)?;
            let selected = select_rows(&Value::Tensor(tensor), rows)?;
            match selected {
                Value::Tensor(tensor) => {
                    crate::builtins::datetime::datetime_object_from_serial_tensor(
                        tensor,
                        crate::builtins::datetime::datetime_format_from_value(value),
                    )
                }
                _ => unreachable!("select_rows tensor branch returns tensor"),
            }
        }
        Value::Object(obj) if obj.is_class("duration") => {
            let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
            let selected = select_rows(&Value::Tensor(tensor), rows)?;
            match selected {
                Value::Tensor(tensor) => {
                    crate::builtins::duration::duration_object_from_days_tensor(
                        tensor,
                        crate::builtins::duration::duration_format_from_value(value),
                    )
                }
                _ => unreachable!("select_rows tensor branch returns tensor"),
            }
        }
        _ if rows.len() == 1 && rows[0] == 0 => Ok(value.clone()),
        other => Err(invalid_variable(format!(
            "table: row selection unsupported for variable {other:?}"
        ))),
    }
}

fn assign_rows(mut current: Value, rows: &[usize], rhs: Value) -> BuiltinResult<Value> {
    if value_row_count(&rhs)? != rows.len() {
        return Err(invalid_variable(
            "table: assignment row count must match selected row count",
        ));
    }
    let replacing_all_rows = rows.len() == value_row_count(&current)?;
    match (&mut current, rhs) {
        (Value::Tensor(target), Value::Tensor(source)) => {
            if target.cols() != source.cols() {
                return Err(invalid_variable(
                    "table: tensor assignment column count mismatch",
                ));
            }
            for col in 0..target.cols() {
                for (src_row, &dst_row) in rows.iter().enumerate() {
                    let value = source.get2(src_row, col).map_err(invalid_index)?;
                    target.set2(dst_row, col, value).map_err(invalid_index)?;
                }
            }
            Ok(current)
        }
        (_, source) if replacing_all_rows => Ok(source),
        _ => Err(invalid_variable(
            "table: assignment for this variable type requires replacing all rows",
        )),
    }
}

fn concatenate_numeric_columns(values: &[&Value]) -> BuiltinResult<Value> {
    let rows = values
        .first()
        .and_then(|value| match value {
            Value::Tensor(t) => Some(t.rows()),
            _ => None,
        })
        .unwrap_or(0);
    let cols = values
        .iter()
        .map(|value| match value {
            Value::Tensor(t) => Ok(t.cols()),
            _ => Err(invalid_variable("table: expected numeric variable")),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let total_cols: usize = cols.iter().sum();
    let mut data = Vec::with_capacity(rows * total_cols);
    for value in values {
        let Value::Tensor(tensor) = value else {
            return Err(invalid_variable("table: expected numeric variable"));
        };
        for col in 0..tensor.cols() {
            for row in 0..rows {
                data.push(tensor.get2(row, col).map_err(invalid_index)?);
            }
        }
    }
    Tensor::new(data, vec![rows, total_cols])
        .map(Value::Tensor)
        .map_err(invalid_variable)
}

pub fn sortrows_table(value: Value, rest: &[Value]) -> BuiltinResult<(Value, Tensor)> {
    let object = into_table_object(value, "sortrows")?;
    let names = table_variable_names_from_object(&object)?;
    let sort_spec = SortSpec::parse(rest, &names)?;
    let height = table_height(&object)?;
    let variables = table_variables(&object)?;
    let mut indices: Vec<usize> = (0..height).collect();
    indices.sort_by(|&a, &b| {
        for key in &sort_spec.keys {
            let Some(value) = variables.fields.get(&key.name) else {
                continue;
            };
            let ord = compare_table_cells(value, a, b).unwrap_or(Ordering::Equal);
            let ord = if key.descending { ord.reverse() } else { ord };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        a.cmp(&b)
    });
    let mut sorted_columns = Vec::with_capacity(names.len());
    for name in &names {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable(format!("table: missing variable '{name}'")))?;
        sorted_columns.push(select_rows(value, &indices)?);
    }
    let row_names = selected_row_names(&object, &indices)?;
    let sorted = table_from_columns_with_properties(names, sorted_columns, row_names)?;
    let indices_tensor = Tensor::new(
        indices.iter().map(|idx| *idx as f64 + 1.0).collect(),
        vec![indices.len(), 1],
    )
    .map_err(invalid_variable)?;
    Ok((sorted, indices_tensor))
}

struct SortSpec {
    keys: Vec<SortKey>,
}

struct SortKey {
    name: String,
    descending: bool,
}

impl SortSpec {
    fn parse(rest: &[Value], names: &[String]) -> BuiltinResult<Self> {
        let mut keys = if rest.is_empty() {
            names
                .iter()
                .map(|name| SortKey {
                    name: name.clone(),
                    descending: false,
                })
                .collect::<Vec<_>>()
        } else {
            parse_variable_selector(rest.first(), names)?
                .into_iter()
                .map(|name| SortKey {
                    name,
                    descending: false,
                })
                .collect()
        };
        if let Some(direction) = rest.get(1) {
            let directions = string_list(direction)?;
            if directions.len() == 1 {
                let descending = directions[0].eq_ignore_ascii_case("descend")
                    || directions[0].eq_ignore_ascii_case("desc");
                for key in &mut keys {
                    key.descending = descending;
                }
            } else {
                for (key, direction) in keys.iter_mut().zip(directions.iter()) {
                    key.descending = direction.eq_ignore_ascii_case("descend")
                        || direction.eq_ignore_ascii_case("desc");
                }
            }
        }
        Ok(Self { keys })
    }
}

fn compare_table_cells(value: &Value, a: usize, b: usize) -> BuiltinResult<Ordering> {
    match value {
        Value::Tensor(tensor) => Ok(tensor
            .get2(a, 0)
            .map_err(invalid_index)?
            .partial_cmp(&tensor.get2(b, 0).map_err(invalid_index)?)
            .unwrap_or(Ordering::Greater)),
        Value::StringArray(array) => {
            let av = array.data.get(a).cloned().unwrap_or_default();
            let bv = array.data.get(b).cloned().unwrap_or_default();
            Ok(av.cmp(&bv))
        }
        Value::LogicalArray(array) => {
            let av = *array.data.get(a).unwrap_or(&0);
            let bv = *array.data.get(b).unwrap_or(&0);
            Ok(av.cmp(&bv))
        }
        Value::Object(obj) if obj.is_class("datetime") => {
            let tensor = crate::builtins::datetime::serials_from_datetime_value(value)?;
            Ok(tensor
                .data
                .get(a)
                .copied()
                .unwrap_or(f64::NAN)
                .partial_cmp(&tensor.data.get(b).copied().unwrap_or(f64::NAN))
                .unwrap_or(Ordering::Greater))
        }
        other => Ok(cell_key_string(other, a).cmp(&cell_key_string(other, b))),
    }
}

#[derive(Clone, Debug)]
enum GroupAtom {
    Number(f64),
    Text(String),
    Logical(bool),
    Missing,
}

impl GroupAtom {
    fn rank(&self) -> u8 {
        match self {
            Self::Missing => 0,
            Self::Logical(_) => 1,
            Self::Number(_) => 2,
            Self::Text(_) => 3,
        }
    }
}

impl PartialEq for GroupAtom {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for GroupAtom {}

impl PartialOrd for GroupAtom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GroupAtom {
    fn cmp(&self, other: &Self) -> Ordering {
        let rank = self.rank().cmp(&other.rank());
        if rank != Ordering::Equal {
            return rank;
        }
        match (self, other) {
            (Self::Missing, Self::Missing) => Ordering::Equal,
            (Self::Logical(a), Self::Logical(b)) => a.cmp(b),
            (Self::Number(a), Self::Number(b)) => a.total_cmp(b),
            (Self::Text(a), Self::Text(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

fn cell_group_atom(value: &Value, row: usize) -> GroupAtom {
    match value {
        Value::Tensor(tensor) => tensor
            .get2(row, 0)
            .map(GroupAtom::Number)
            .unwrap_or(GroupAtom::Missing),
        Value::StringArray(array) => array
            .data
            .get(row)
            .cloned()
            .map(GroupAtom::Text)
            .unwrap_or(GroupAtom::Missing),
        Value::LogicalArray(array) => array
            .data
            .get(row)
            .map(|value| GroupAtom::Logical(*value != 0))
            .unwrap_or(GroupAtom::Missing),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .ok()
                .and_then(|tensor| tensor.data.get(row).copied())
                .map(GroupAtom::Number)
                .unwrap_or(GroupAtom::Missing)
        }
        other => GroupAtom::Text(cell_key_string(other, row)),
    }
}

fn groupsummary_impl(
    table: Value,
    groupvars: Value,
    method: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let object = into_table_object(table, "groupsummary")?;
    let names = table_variable_names_from_object(&object)?;
    let group_names = parse_variable_selector(Some(&groupvars), &names)?;
    let methods = string_list(&method)?;
    if methods.is_empty() {
        return Err(invalid_argument(
            "groupsummary: method list must not be empty",
        ));
    }
    let data_names = if let Some(value) = rest.first() {
        parse_variable_selector(Some(value), &names)?
    } else {
        names
            .iter()
            .filter(|name| !group_names.contains(name))
            .filter(|name| {
                table_variables(&object)
                    .ok()
                    .and_then(|vars| vars.fields.get(*name).cloned())
                    .map(|value| matches!(value, Value::Tensor(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    };
    let variables = table_variables(&object)?;
    let height = table_height(&object)?;
    let mut groups: BTreeMap<Vec<GroupAtom>, Vec<usize>> = BTreeMap::new();
    for row in 0..height {
        let key = group_names
            .iter()
            .map(|name| {
                variables
                    .fields
                    .get(name)
                    .map(|value| cell_group_atom(value, row))
                    .unwrap_or(GroupAtom::Missing)
            })
            .collect::<Vec<_>>();
        groups.entry(key).or_default().push(row);
    }
    let group_rows = groups
        .values()
        .filter_map(|rows| rows.first().copied())
        .collect::<Vec<_>>();
    let mut out_names = Vec::new();
    let mut out_columns = Vec::new();
    for name in &group_names {
        let value = variables.fields.get(name).ok_or_else(|| {
            invalid_variable(format!("groupsummary: missing group variable '{name}'"))
        })?;
        out_names.push(name.clone());
        out_columns.push(select_rows(value, &group_rows)?);
    }
    out_names.push("GroupCount".to_string());
    out_columns.push(Value::Tensor(
        Tensor::new(
            groups.values().map(|rows| rows.len() as f64).collect(),
            vec![groups.len(), 1],
        )
        .map_err(invalid_variable)?,
    ));
    for method in &methods {
        for name in &data_names {
            let value = variables.fields.get(name).ok_or_else(|| {
                invalid_variable(format!("groupsummary: missing data variable '{name}'"))
            })?;
            let values = summarize_groups(value, groups.values(), method)?;
            out_names.push(format!("{}_{}", method.to_ascii_lowercase(), name));
            out_columns.push(Value::Tensor(
                Tensor::new(values, vec![groups.len(), 1]).map_err(invalid_variable)?,
            ));
        }
    }
    table_from_columns(out_names, out_columns)
}

fn summarize_groups<'a>(
    value: &Value,
    groups: impl Iterator<Item = &'a Vec<usize>>,
    method: &str,
) -> BuiltinResult<Vec<f64>> {
    let tensor = match value {
        Value::Tensor(tensor) if tensor.cols() == 1 => tensor,
        _ => {
            return Err(invalid_variable(
                "groupsummary: summary data variables must be numeric column vectors",
            ))
        }
    };
    groups
        .map(|rows| {
            let mut values = rows
                .iter()
                .map(|row| tensor.get2(*row, 0).map_err(invalid_index))
                .collect::<BuiltinResult<Vec<_>>>()?;
            values.retain(|value| !value.is_nan());
            let result = match method.to_ascii_lowercase().as_str() {
                "mean" => {
                    if values.is_empty() {
                        f64::NAN
                    } else {
                        values.iter().sum::<f64>() / values.len() as f64
                    }
                }
                "sum" => values.iter().sum(),
                "min" => values.into_iter().fold(f64::INFINITY, f64::min),
                "max" => values.into_iter().fold(f64::NEG_INFINITY, f64::max),
                "median" => {
                    if values.is_empty() {
                        f64::NAN
                    } else {
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let mid = values.len() / 2;
                        if values.len() % 2 == 0 {
                            (values[mid - 1] + values[mid]) / 2.0
                        } else {
                            values[mid]
                        }
                    }
                }
                "count" | "numel" => values.len() as f64,
                other => {
                    return Err(invalid_argument(format!(
                        "groupsummary: unsupported method '{other}'"
                    )))
                }
            };
            Ok(result)
        })
        .collect()
}

fn cell_key_string(value: &Value, row: usize) -> String {
    match value {
        Value::Tensor(tensor) => tensor
            .get2(row, 0)
            .map(format_key_number)
            .unwrap_or_default(),
        Value::StringArray(array) => array.data.get(row).cloned().unwrap_or_default(),
        Value::LogicalArray(array) => array
            .data
            .get(row)
            .map(|value| value.to_string())
            .unwrap_or_default(),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .ok()
                .and_then(|tensor| tensor.data.get(row).copied())
                .map(format_key_number)
                .unwrap_or_default()
        }
        other => format!("{other}"),
    }
}

pub fn table_display_text(value: &Value) -> BuiltinResult<String> {
    let object = match value {
        Value::Object(object) if object.is_class(TABLE_CLASS) => object,
        _ => return Err(invalid_argument("table display expects table object")),
    };
    let names = table_variable_names_from_object(object)?;
    let variables = table_variables(object)?;
    let rows = table_height(object)?;
    let preview = rows.min(12);
    let mut widths = names.iter().map(|name| name.len()).collect::<Vec<_>>();
    let rendered_cols = names
        .iter()
        .enumerate()
        .map(|(col, name)| {
            let value = variables
                .fields
                .get(name)
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
            let cells = (0..preview)
                .map(|row| render_table_cell(&value, row))
                .collect::<Vec<_>>();
            for cell in &cells {
                widths[col] = widths[col].max(cell.len());
            }
            cells
        })
        .collect::<Vec<_>>();

    let mut lines = Vec::new();
    lines.push(format!("{rows}x{} table", names.len()));
    if names.is_empty() {
        return Ok(lines.join("\n"));
    }
    let header = names
        .iter()
        .enumerate()
        .map(|(idx, name)| format!("{name:<width$}", width = widths[idx]))
        .collect::<Vec<_>>()
        .join("  ");
    lines.push(header);
    for row in 0..preview {
        lines.push(
            rendered_cols
                .iter()
                .enumerate()
                .map(|(col, cells)| format!("{:<width$}", cells[row], width = widths[col]))
                .collect::<Vec<_>>()
                .join("  "),
        );
    }
    if preview < rows {
        lines.push(format!("... {} more rows", rows - preview));
    }
    Ok(lines.join("\n"))
}

pub fn table_summary_text(value: &Value) -> BuiltinResult<String> {
    let object = match value {
        Value::Object(object) if object.is_class(TABLE_CLASS) => object,
        _ => return Err(invalid_argument("table display expects table object")),
    };
    Ok(format!(
        "{}x{} table",
        table_height(object)?,
        table_width(object)?
    ))
}

fn render_table_cell(value: &Value, row: usize) -> String {
    match value {
        Value::Tensor(tensor) => tensor
            .get2(row, 0)
            .map(format_table_number)
            .unwrap_or_default(),
        Value::StringArray(array) => array.data.get(row).cloned().unwrap_or_default(),
        Value::LogicalArray(array) => array
            .data
            .get(row)
            .map(|value| if *value != 0 { "true" } else { "false" }.to_string())
            .unwrap_or_default(),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::datetime_string_array(value)
                .ok()
                .flatten()
                .and_then(|array| array.data.get(row).cloned())
                .unwrap_or_else(|| value.to_string())
        }
        other => other.to_string(),
    }
}

fn format_table_number(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        trim_float(format!("{value:.6}"))
    }
}

fn format_key_number(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        value.to_string()
    } else {
        trim_float(format!("{value:.17}"))
    }
}

fn trim_float(mut text: String) -> String {
    if let Some(dot) = text.find('.') {
        let mut end = text.len();
        while end > dot + 1 && text.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if end == dot + 1 {
            end -= 1;
        }
        text.truncate(end);
    }
    text
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
    use futures::executor::block_on;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::io::Write;

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
            "Value,Flag,When,Elapsed\n1.5,true,2026-06-01,01:30:00\n2.25,false,2026-06-02,02:00:00\n",
        )
        .expect("write sample");
        let types = StringArray::new(
            vec![
                "single".to_string(),
                "logical".to_string(),
                "datetime".to_string(),
                "duration".to_string(),
            ],
            vec![1, 4],
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
        let categorical = StringArray::new(vec!["categorical".to_string()], vec![1, 1]).unwrap();
        let err = read_table_err(
            &path,
            vec![
                Value::from("VariableTypes"),
                Value::StringArray(categorical),
            ],
        );
        assert!(err
            .message()
            .contains("unsupported VariableTypes entry 'categorical'"));
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
}
