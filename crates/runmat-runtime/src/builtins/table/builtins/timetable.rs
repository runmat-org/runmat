use super::*;
pub(crate) use runmat_builtins::{
    ARRAY2TIMETABLE_GPU_INPUT_EXTENSION, READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION,
    TABLE2TIMETABLE_GENERATED_TIMES_EXTENSION, TABLE2TIMETABLE_GPU_INPUT_EXTENSION,
    TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION, TIMETABLE2TABLE_EXPLICIT_GPU_EXTENSION,
    TIMETABLE2TABLE_INTEGER_OPTION_EXTENSION, TIMETABLE_EXPLICIT_GPU_EXTENSION,
    TIMETABLE_GENERATED_ROW_TIMES_EXTENSION, TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
};
use runmat_macros::runtime_builtin;
const ARRAY2TIMETABLE_BUILTIN_NAME: &str = "array2timetable";
fn ensure_numeric_row_times_extension(args: &[Value], builtin_name: &str) -> BuiltinResult<()> {
    let has_numeric_vector = args.chunks_exact(2).any(|pair| {
        scalar_text(&pair[0], "RowTimes option")
            .is_ok_and(|name| name.eq_ignore_ascii_case("RowTimes"))
            && match &pair[1] {
                Value::Tensor(tensor) => tensor.len() != 1,
                Value::GpuTensor(handle) => handle.shape().iter().any(|&dimension| dimension != 1),
                _ => false,
            }
    });
    if has_numeric_vector {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
            builtin_name,
        )?;
    }
    Ok(())
}
#[runtime_builtin(
    name = "timetable",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timetable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMETABLE_EXPLICIT_GPU_EXTENSION,
            "timetable",
        )?;
    }
    let args = gather_values(&args).await?;
    let (explicit_row_times, variables, options) = split_timetable_constructor_args(args)?;
    let timing_forms = usize::from(explicit_row_times.is_some())
        + usize::from(options.sample_rate.is_some())
        + usize::from(options.time_step.is_some());
    if timing_forms > 1 {
        return Err(invalid_argument(
            "timetable: specify exactly one of RowTimes, SampleRate, or TimeStep",
        ));
    }
    if options.start_time.is_some() && options.sample_rate.is_none() && options.time_step.is_none()
    {
        return Err(invalid_argument(
            "timetable: StartTime requires SampleRate or TimeStep",
        ));
    }

    let preallocated = options.table.size.is_some() || options.table.variable_types.is_some();
    let (names, variables, row_names, height) = if preallocated {
        preallocated_table_columns(variables, options.table, "timetable")?
    } else {
        let names = options
            .table
            .variable_names
            .unwrap_or_else(|| generated_variable_names(variables.len()));
        let height = variables
            .first()
            .map(value_row_count)
            .transpose()?
            .unwrap_or(0);
        (names, variables, options.table.row_names, height)
    };

    let row_times = if let Some(row_times) = explicit_row_times {
        Some(row_times)
    } else if options.sample_rate.is_some() || options.time_step.is_some() {
        Some(array2timetable_row_times(
            &Array2TimetableOptions {
                row_times: None,
                sample_rate: options.sample_rate,
                time_step: options.time_step,
                start_time: options.start_time,
                variable_names: None,
                dimension_names: None,
            },
            height,
        )?)
    } else {
        None
    };

    if let Some(row_times) = row_times.as_ref() {
        if !is_time_like_value(row_times) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
                "timetable",
            )?;
            if !matches!(row_times, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
                return Err(invalid_argument(
                    "timetable: RowTimes must contain datetime or duration values",
                ));
            }
        }
    } else if height > 0 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMETABLE_GENERATED_ROW_TIMES_EXTENSION,
            "timetable",
        )?;
    }

    let mut value = table_from_columns_with_class(TIMETABLE_CLASS, names, variables, row_names)?;
    if let Value::Object(object) = &mut value {
        set_timetable_row_times(object, row_times)?;
    }
    Ok(value)
}

#[runtime_builtin(
    name = "array2timetable",
    binding_variant = "default",
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
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2timetable_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_numeric_row_times_extension(&rest, "table2timetable")?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TABLE2TIMETABLE_GPU_INPUT_EXTENSION,
            "table2timetable",
        )?;
    }
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_table2timetable_options(&rest)?;
    let object = into_table_object(host, "table2timetable")?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let row_names = selected_row_names(&object, &(0..table_height(&object)?).collect::<Vec<_>>())?;
    let height = table_height(&object)?;
    let (times, out_names) = if let Some(row_times) = options.row_times.as_ref() {
        if is_time_like_value(row_times) {
            validate_explicit_row_times(row_times, height)?;
            (Some(row_times.clone()), names)
        } else if matches!(row_times, Value::Tensor(tensor) if tensor.len() != 1) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
                "table2timetable",
            )?;
            let Value::Tensor(tensor) = row_times else {
                unreachable!("numeric row-time extension requires a tensor")
            };
            if tensor.len() != height || tensor.cols() != 1 {
                return Err(invalid_argument(format!(
                    "table2timetable: numeric RowTimes must be a {height}-by-1 vector"
                )));
            }
            (Some(row_times.clone()), names)
        } else {
            let selected = table2timetable_time_variable_name(row_times, &names)?;
            let selected_value = variables.fields.get(&selected).cloned().ok_or_else(|| {
                invalid_variable(format!("table2timetable: missing variable '{selected}'"))
            })?;
            if !is_time_like_value(&selected_value) {
                return Err(invalid_argument(
                    "table2timetable: RowTimes variable must contain datetime or duration values",
                ));
            }
            let out_names = names.into_iter().filter(|name| name != &selected).collect();
            (Some(selected_value), out_names)
        }
    } else if let Some(generated) = table2timetable_generated_row_times(&options, height)? {
        (Some(generated), names)
    } else if let Some(first_time_name) = names
        .iter()
        .find(|name| variables.fields.get(*name).is_some_and(is_time_like_value))
        .cloned()
    {
        let first_time_value = variables.fields.get(&first_time_name).cloned();
        let out_names = names
            .into_iter()
            .filter(|name| name != &first_time_name)
            .collect();
        (first_time_value, out_names)
    } else {
        (None, names)
    };
    if times.is_none() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TABLE2TIMETABLE_GENERATED_TIMES_EXTENSION,
            "table2timetable",
        )?;
    }
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

fn table2timetable_time_variable_name(selector: &Value, names: &[String]) -> BuiltinResult<String> {
    if let Ok(name) = scalar_text(selector, "RowTimes variable") {
        if names.iter().any(|candidate| candidate == &name) {
            return Ok(name);
        }
        return Err(invalid_variable(format!(
            "table2timetable: unknown RowTimes variable '{name}'"
        )));
    }
    let index = positive_usize(selector, "RowTimes variable index")?;
    names.get(index - 1).cloned().ok_or_else(|| {
        invalid_variable(format!(
            "table2timetable: RowTimes variable index {index} is out of range"
        ))
    })
}

#[runtime_builtin(
    name = "timetable2table",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timetable2table_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if rest.chunks_exact(2).any(|pair| {
        scalar_text(&pair[0], "timetable2table option").is_ok_and(|name| {
            name.eq_ignore_ascii_case("ConvertRowTimes")
                && crate::builtins::common::validation::value_contains_native_integer_class(
                    &pair[1],
                )
        })
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMETABLE2TABLE_INTEGER_OPTION_EXTENSION,
            "timetable2table",
        )?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMETABLE2TABLE_EXPLICIT_GPU_EXTENSION,
            "timetable2table",
        )?;
    }
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let convert_row_times = parse_timetable2table_convert_row_times(&rest)?;
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

fn parse_timetable2table_convert_row_times(args: &[Value]) -> BuiltinResult<bool> {
    if args.is_empty() {
        return Ok(true);
    }
    if args.len() != 2 {
        return Err(invalid_argument(
            "timetable2table: ConvertRowTimes must be supplied as one name-value pair",
        ));
    }
    let name = scalar_text(&args[0], "timetable2table option")?;
    if !name.eq_ignore_ascii_case("ConvertRowTimes") {
        return Err(invalid_argument(format!(
            "timetable2table: unsupported option '{name}'"
        )));
    }
    zero_one_bool_scalar(&args[1], "ConvertRowTimes")
}

#[runtime_builtin(
    name = "readtimetable",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readtimetable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_numeric_row_times_extension(&rest, "readtimetable")?;
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
