use super::*;
use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;

const ARRAY2TIMETABLE_BUILTIN_NAME: &str = "array2timetable";

pub(crate) const ARRAY2TIMETABLE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "array2timetable-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "array2timetable with an interactive resident GPU argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Array2TimetableGpuInputExtension"),
    };

pub const ARRAY2TIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ARRAY2TIMETABLE_GPU_INPUT_EXTENSION];

pub(crate) const READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "readtimetable-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "readtimetable accepts typed-integer controls whose public datatype tables are floating-only as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReadtimetableTypedIntegerControlExtension"),
    };
pub const READTIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION];
const READTIMETABLE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableTypes integer class",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer variable types are imported directly into exact native table-variable storage before timetable construction.",
    }];
const READTIMETABLE_INTEGER_LOCATION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "VariableNamesLine",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public datatype table explicitly includes all eight integer classes. RunMat reads the authoritative scalar exactly and bounds-checks it before deriving the host header offset.",
    }];
const READTIMETABLE_INTEGER_EXTENSION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Range, Sheet, NumHeaderLines, ExpectedNumVariables, boolean numeric flags, or related floating-only control",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed forms outside the public datatype tables remain a gated RunMat extension and are rejected in compatibility mode before gather or file access.",
    }];
pub const READTIMETABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = readtimetable(filename, import_options_with_integer_VariableTypes)",
        inputs: &READTIMETABLE_INTEGER_VARIABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Timetable containment preserves the exact integer class and payload of imported variables; row-time inference does not coerce unrelated data variables.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = readtimetable(filename, 'VariableNamesLine', integer_line)",
        inputs: &READTIMETABLE_INTEGER_LOCATION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented line control is an exact host index and does not select output residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = readtimetable(filename, typed_integer_extension_controls...)",
        inputs: &READTIMETABLE_INTEGER_EXTENSION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Automatic residency may gather transparently for host I/O, but unsupported typed controls cannot bypass the compatibility gate.",
    },
];

const ARRAY2TIMETABLE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented homogeneous array domain includes all eight real integer classes.",
    }];

const ARRAY2TIMETABLE_INTEGER_SAMPLE_RATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Fs",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive numeric scalar SampleRate control accepts every real integer class as well as floating scalars.",
    }];

pub const ARRAY2TIMETABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = array2timetable(integer_X, timingName,timingValue, Name,Value...)",
        inputs: &ARRAY2TIMETABLE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each X column becomes a timetable variable with X's exact authoritative integer storage and class; row times are stored separately. Interactive resident GPU arguments are mode-gated RunMat extensions that gather before timetable construction.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "TT = array2timetable(X, \"SampleRate\", integer_Fs, Name,Value...)",
        inputs: &ARRAY2TIMETABLE_INTEGER_SAMPLE_RATE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Fs is decoded as an exact positive scalar before the reciprocal seconds-to-days timing boundary; timetable variable classes are determined independently by X.",
    },
];

#[runtime_builtin(
    name = "timetable",
    category = "table",
    summary = "Create a timetable from row times and variables.",
    keywords = "timetable,table,RowTimes,TimeStep,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timetable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    keywords = "array2timetable,timetable,RowTimes,SampleRate,TimeStep,StartTime,VariableNames,DimensionNames",
    accel = "gather",
    descriptor(crate::builtins::table::ARRAY2TIMETABLE_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::timetable::ARRAY2TIMETABLE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::timetable::ARRAY2TIMETABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn array2timetable_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if matches!(value, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAY2TIMETABLE_GPU_INPUT_EXTENSION,
            ARRAY2TIMETABLE_BUILTIN_NAME,
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_array2timetable_options(&rest)?;
    let columns = split_value_columns(value)?;
    let names = options
        .variable_names
        .clone()
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    validate_array2timetable_names(&names, options.dimension_names.as_deref())?;
    let height = columns
        .first()
        .map(value_row_count)
        .transpose()?
        .unwrap_or(0);
    let row_times = array2timetable_row_times(&options, height)?;
    let mut out = table_from_columns_with_class(TIMETABLE_CLASS, names, columns, None)?;
    if let Value::Object(object) = &mut out {
        set_timetable_row_times(object, Some(row_times))?;
        if let Some(dimension_names) = options.dimension_names {
            set_table_dimension_names(object, dimension_names, ARRAY2TIMETABLE_BUILTIN_NAME)?;
        }
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2timetable_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timetable2table_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
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
    extensions(crate::builtins::table::builtins::timetable::READTIMETABLE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::timetable::READTIMETABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readtimetable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    super::io::enforce_table_import_integer_control_gate(
        &rest,
        &READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION,
        "readtimetable",
        &["VariableNamesLine"],
    )?;
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let (readtable_args, timetable_args) = split_readtimetable_options(&rest)?;
    let table = super::io::readtable_builtin(path, readtable_args).await?;
    table2timetable_builtin(table, timetable_args).await
}
