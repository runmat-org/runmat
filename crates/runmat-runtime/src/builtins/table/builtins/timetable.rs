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
fn ensure_numeric_row_times_extension(
    args: crate::call::FunctionArgumentView<'_>,
    builtin_name: &str,
) -> BuiltinResult<()> {
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
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timetable_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
    let args = gather_retained_values(&args, runtime.as_ref()).await?;
    let (explicit_row_times, variables, options) =
        split_timetable_constructor_args(&args, runtime.as_ref()).await?;
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
        if !variables.is_empty() {
            return Err(invalid_argument(
                "timetable: Size/VariableTypes preallocation cannot be combined with data variables",
            ));
        }
        let (names, columns, row_names, height) =
            preallocated_table_columns(Vec::new(), options.table, "timetable")?;
        let columns = crate::call::BuiltinArguments::from_borrowed(&columns, runtime.as_ref())?;
        (names, columns, row_names, height)
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

    let mut value = table_from_retained_columns_with_class(
        runtime.as_ref(),
        TIMETABLE_CLASS,
        names,
        variables.iter(),
        row_names,
    )?;
    if let Value::Object(object) = &mut value {
        set_timetable_row_times(runtime.as_ref(), object, row_times)?;
    }
    Ok(value)
}

#[runtime_builtin(
    name = "array2timetable",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn array2timetable_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
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
    let value = gather_retained_value(&value, runtime.as_ref()).await?;
    let rest = gather_retained_values(&rest, runtime.as_ref()).await?;
    let options = parse_array2timetable_options(rest.view())?;
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
    let mut out = table_from_retained_columns_with_class(
        runtime.as_ref(),
        TIMETABLE_CLASS,
        names,
        columns.iter(),
        None,
    )?;
    if let Value::Object(object) = &mut out {
        set_timetable_row_times(runtime.as_ref(), object, Some(row_times))?;
        if let Some(dimension_names) = options.dimension_names {
            set_table_dimension_names(
                runtime.as_ref(),
                object,
                dimension_names,
                ARRAY2TIMETABLE_BUILTIN_NAME,
            )?;
        }
    }
    Ok(out)
}

#[runtime_builtin(
    name = "table2timetable",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2timetable_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_numeric_row_times_extension(rest.view(), "table2timetable")?;
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
    let host = gather_retained_value(&value, runtime.as_ref()).await?;
    let rest = gather_retained_values(&rest, runtime.as_ref()).await?;
    let options = parse_table2timetable_options(rest.view())?;
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
        .find(|name| variables.fields.get(name).is_some_and(is_time_like_value))
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
    for name in &out_names {
        if !variables.fields.contains_key(name) {
            return Err(invalid_variable(format!(
                "table2timetable: missing variable '{name}'"
            )));
        }
    }
    let out_columns = out_names.iter().map(|name| {
        variables
            .fields
            .get(name)
            .expect("validated output variable")
    });
    let mut out = table_from_retained_columns_with_class(
        runtime.as_ref(),
        TIMETABLE_CLASS,
        out_names.clone(),
        out_columns,
        row_names,
    )?;
    if let Value::Object(object) = &mut out {
        set_timetable_row_times(runtime.as_ref(), object, times)?;
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
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timetable2table_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
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
    let host = gather_retained_value(&value, runtime.as_ref()).await?;
    let rest = gather_retained_values(&rest, runtime.as_ref()).await?;
    let convert_row_times = parse_timetable2table_convert_row_times(rest.view())?;
    let object = into_timetable_object(host, "timetable2table")?;
    let mut names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let source_names = table_variable_names_from_object(&object)?;
    let row_times = if convert_row_times {
        timetable_row_times(&object)?
    } else {
        None
    };
    if row_times.is_some() {
        names.insert(0, "Time".to_string());
    }
    for name in &source_names {
        if !variables.fields.contains_key(name) {
            return Err(invalid_variable(format!(
                "timetable2table: missing variable '{name}'"
            )));
        }
    }
    let columns = row_times.iter().chain(source_names.iter().map(|name| {
        variables
            .fields
            .get(name)
            .expect("validated source variable")
    }));
    let row_names = selected_row_names(&object, &(0..table_height(&object)?).collect::<Vec<_>>())?;
    table_from_retained_columns_with_class(runtime.as_ref(), TABLE_CLASS, names, columns, row_names)
}

fn parse_timetable2table_convert_row_times(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<bool> {
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
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readtimetable_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_numeric_row_times_extension(rest.view(), "readtimetable")?;
    super::io::enforce_table_import_integer_control_gate(
        rest.view(),
        &READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION,
        "readtimetable",
        &["VariableNamesLine"],
    )?;
    let path = gather_retained_value(&path, runtime.as_ref()).await?;
    let rest = gather_retained_values(&rest, runtime.as_ref()).await?;
    let (readtable_args, timetable_args) =
        split_readtimetable_options(&rest, runtime.as_ref()).await?;
    let table = super::io::readtable_builtin(runtime.clone(), path, readtable_args).await?;
    table2timetable_builtin(runtime, table, timetable_args).await
}
