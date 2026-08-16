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

pub(crate) const READTABLE_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "readtable-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "readtable accepts typed-integer controls whose public datatype tables are floating-only as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReadtableTypedIntegerControlExtension"),
    };
pub(crate) const READCELL_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "readcell-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "readcell accepts typed-integer controls whose public datatype tables are floating-only as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReadcellTypedIntegerControlExtension"),
    };
pub(crate) const SPREADSHEET_OPTIONS_TYPED_INTEGER_CONTROL_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spreadsheetimportoptions-typed-integer-location-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spreadsheetImportOptions accepts typed-integer location controls outside documented NumVariables as a RunMat extension",
    error_identifier: Some(
        "RunMat:compatibility:SpreadsheetImportOptionsTypedIntegerControlExtension",
    ),
};
pub const READTABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [READTABLE_TYPED_INTEGER_CONTROL_EXTENSION];
pub const READCELL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [READCELL_TYPED_INTEGER_CONTROL_EXTENSION];
pub const SPREADSHEET_IMPORT_OPTIONS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [SPREADSHEET_OPTIONS_TYPED_INTEGER_CONTROL_EXTENSION];

const TABLE_IMPORT_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableTypes integer class",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer variable types parse decimal text directly into private native storage, including exact int64 and uint64 endpoints and saturating conversion of missing, infinite, or out-of-range values.",
    }];
const TABLE_IMPORT_VARIABLE_NAMES_LINE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableNamesLine",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public datatype table explicitly includes all eight integer classes. RunMat reads the authoritative scalar exactly and bounds-checks it before deriving the zero-based host header offset.",
    }];
const READCELL_NUM_HEADER_LINES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "NumHeaderLines",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public contract is a positive integer. RunMat accepts every native integer class, reads the authoritative scalar exactly, and bounds-checks it before skipping host input records.",
    }];
const TABLE_IMPORT_EXTENSION_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Range, Sheet, NumHeaderLines, or another implemented control whose public datatype list excludes native integers",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public R2026a datatype tables restrict these implemented controls to floating or nonnumeric classes. RunMat mode retains exact typed-integer admission; compatibility mode rejects before gather or file access.",
    }];
const SPREADSHEET_NUM_VARIABLES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "NumVariables",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive scalar integer is read exactly from every integer class and must fit usize before import-option metadata is allocated.",
    }];
pub const READTABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "T = readtable(filename, import_options_with_integer_VariableTypes)",
        inputs: &TABLE_IMPORT_INTEGER_VARIABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Each imported integer table variable retains its documented class and exact values inside the table.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = readtable(filename, 'VariableNamesLine', integer_line)",
        inputs: &TABLE_IMPORT_VARIABLE_NAMES_LINE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented integer line control affects only host import structure and never selects output class or residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = readtable(filename, typed_integer_extension_controls...)",
        inputs: &TABLE_IMPORT_EXTENSION_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The pre-existing typed-control extension is independently gated; ordinary resident arguments may still gather transparently for host I/O.",
    },
];
pub const READCELL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = readcell(filename, 'NumHeaderLines', integer_count)",
        inputs: &READCELL_NUM_HEADER_LINES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Imported numeric cells retain the documented double scalar representation; the integer header count affects only import structure.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = readcell(filename, typed_integer_extension_controls...)",
        inputs: &TABLE_IMPORT_EXTENSION_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Compatibility mode rejects typed floating-only controls before file access; RunMat mode validates them exactly.",
    },
];
pub const SPREADSHEET_IMPORT_OPTIONS_INTEGER_CAPABILITIES:
    [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "opts = spreadsheetImportOptions('NumVariables', integer_num_vars, ___)",
        inputs: &SPREADSHEET_NUM_VARIABLES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "VariableTypes may name all eight native integer output classes for later readtable use; the options structure itself is host metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "opts = spreadsheetImportOptions(typed_integer_location_controls...)",
        inputs: &TABLE_IMPORT_EXTENSION_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed location controls beyond NumVariables are a gated RunMat extension and are never rounded through binary64.",
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
    extensions(crate::builtins::table::builtins::io::READTABLE_EXTENSIONS),
    integer_capabilities(crate::builtins::table::builtins::io::READTABLE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readtable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    enforce_table_import_integer_control_gate(
        &rest,
        &READTABLE_TYPED_INTEGER_CONTROL_EXTENSION,
        "readtable",
        &["VariableNamesLine"],
    )?;
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
    extensions(crate::builtins::table::builtins::io::SPREADSHEET_IMPORT_OPTIONS_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::io::SPREADSHEET_IMPORT_OPTIONS_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn spreadsheet_import_options_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    enforce_table_import_integer_control_gate(
        &args,
        &SPREADSHEET_OPTIONS_TYPED_INTEGER_CONTROL_EXTENSION,
        "spreadsheetImportOptions",
        &["NumVariables"],
    )?;
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
    extensions(crate::builtins::table::builtins::io::READCELL_EXTENSIONS),
    integer_capabilities(crate::builtins::table::builtins::io::READCELL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readcell_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    enforce_table_import_integer_control_gate(
        &rest,
        &READCELL_TYPED_INTEGER_CONTROL_EXTENSION,
        "readcell",
        &["NumHeaderLines"],
    )?;
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let path = resolve_path(&path)?;
    let options = ReadTableOptions::parse(&rest)?;
    read_cell_from_file(&path, &options).await
}

pub(crate) fn enforce_table_import_integer_control_gate(
    args: &[Value],
    extension: &'static BuiltinExtensionDescriptor,
    builtin: &'static str,
    documented_integer_options: &[&str],
) -> BuiltinResult<()> {
    if typed_integer_option_names(args).iter().any(|name| {
        !documented_integer_options
            .iter()
            .any(|allowed| name.eq_ignore_ascii_case(allowed))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
}

fn typed_integer_option_names(args: &[Value]) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(Value::Struct(options)) = args.first() {
        out.extend(
            options.fields.iter().filter_map(|(name, value)| {
                value_contains_typed_integer(value).then(|| name.clone())
            }),
        );
    }
    for pair in args.windows(2) {
        if !value_contains_typed_integer(&pair[1]) {
            continue;
        }
        if let Ok(name) = scalar_text(&pair[0], "import option") {
            out.push(name);
        }
    }
    out
}

fn value_contains_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
        || matches!(value, Value::Cell(cell) if cell.data.iter().any(value_contains_typed_integer))
        || matches!(value, Value::Struct(structure) if structure.fields.values().any(value_contains_typed_integer))
}
