use super::*;
use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;

pub(crate) const DETECT_IMPORT_OPTIONS_INTEGER_NUM_HEADER_LINES_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "detectimportoptions-integer-num-header-lines",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "detectImportOptions with a typed-integer NumHeaderLines value is a RunMat extension",
    error_identifier: Some(
        "RunMat:compatibility:DetectImportOptionsIntegerNumHeaderLinesExtension",
    ),
};

pub const DETECT_IMPORT_OPTIONS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [DETECT_IMPORT_OPTIONS_INTEGER_NUM_HEADER_LINES_EXTENSION];

const DETECT_IMPORT_OPTIONS_INTEGER_LOCATION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Range, DataRange, Sheet, ExpectedNumVariables, or VariableNamesLine",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Supported text/spreadsheet location and size controls are parsed exactly from authoritative integer storage and validated before conversion to host indices; numeric DataRange supports scalar rows and one 1-by-2 interval, textual cell references remain available for detected-option replay, and multiple numeric Nx2 intervals remain a general gap.",
    }];
const DETECT_IMPORT_OPTIONS_INTEGER_HEADER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "NumHeaderLines",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public R2026a datatype list is single/double; typed-integer values are admitted only in RunMat mode after an explicit extension gate.",
    }];
pub const DETECT_IMPORT_OPTIONS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "opts = detectImportOptions(filename, integer_location_controls...)",
        inputs: &DETECT_IMPORT_OPTIONS_INTEGER_LOCATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The implemented text/spreadsheet subset preserves exact integer controls and returns host import metadata; JSON/XML/Word/HTML/archive families and concrete MATLAB options classes remain explicit general gaps.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "opts = detectImportOptions(filename, \"NumHeaderLines\", typed_integer)",
        inputs: &DETECT_IMPORT_OPTIONS_INTEGER_HEADER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat mode parses this pre-existing extension exactly; compatibility mode rejects before file or provider access.",
    },
];

#[runtime_builtin(
    name = "readtable",
    category = "io/tabular",
    summary = "Import tabular text or spreadsheet data into a table.",
    keywords = "readtable,table,csv,tsv,xlsx,xls,ods,spreadsheet,VariableNames,RowNames,Sheet,Range",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::READTABLE_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readtable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
    name = "parquetread",
    category = "io/tabular",
    summary = "Read Parquet columnar data into a table.",
    keywords = "parquetread,parquet,table,SelectedVariableNames,RowGroups,OutputType",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::PARQUETREAD_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn parquetread_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let args = gather_values(&rest).await?;
    let options = ParquetReadOptions::parse(&args)?;
    let resolved = resolve_path(&path_value)?;
    read_parquet_table(&resolved, &options).await
}

#[runtime_builtin(
    name = "parquetinfo",
    category = "io/tabular",
    summary = "Inspect Parquet file schema and row-group metadata.",
    keywords = "parquetinfo,parquet,schema,row groups,metadata",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::PARQUETINFO_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn parquetinfo_builtin(path: Value) -> BuiltinResult<Value> {
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let resolved = resolve_path(&path_value)?;
    parquet_file_info(&resolved).await
}

#[runtime_builtin(
    name = "spreadsheetImportOptions",
    category = "io/tabular",
    summary = "Create spreadsheet import options for readtable.",
    keywords = "spreadsheetImportOptions,readtable,spreadsheet,xlsx,xls,DataRange,VariableTypes,VariableNames,NumVariables",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn spreadsheet_import_options_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    extensions(crate::builtins::table::builtins::io::DETECT_IMPORT_OPTIONS_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::io::DETECT_IMPORT_OPTIONS_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn detect_import_options_builtin(
    path: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if crate::value_contains_gpu(&path) || rest.iter().any(crate::value_contains_gpu) {
        return Err(invalid_argument(
            "detectImportOptions: resident arguments are not supported",
        ));
    }
    if detect_option_has_typed_integer(&rest, "NumHeaderLines") {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DETECT_IMPORT_OPTIONS_INTEGER_NUM_HEADER_LINES_EXTENSION,
            "detectImportOptions",
        )?;
    }
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let args = gather_values(&rest).await?;
    let options = ReadTableOptions::parse(&args)?;
    let resolved = resolve_path(&path_value)?;
    detect_import_options_from_file(&resolved, &options).await
}

fn detect_option_has_typed_integer(args: &[Value], sought: &str) -> bool {
    if let Some(Value::Struct(options)) = args.first() {
        if options
            .fields
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case(sought) && is_typed_integer(value))
        {
            return true;
        }
    }
    args.windows(2).any(|pair| {
        scalar_text(&pair[0], "detectImportOptions option")
            .is_ok_and(|name| name.eq_ignore_ascii_case(sought))
            && is_typed_integer(&pair[1])
    })
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

#[runtime_builtin(
    name = "writetable",
    category = "io/tabular",
    summary = "Write a table to a delimited text file.",
    keywords = "writetable,table,csv,delimited text,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_WRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn writetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn writetimetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readcell_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let path = resolve_path(&path)?;
    let options = ReadTableOptions::parse(&rest)?;
    read_cell_from_file(&path, &options).await
}
