use super::*;
use runmat_macros::runtime_builtin;

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
    keywords = "array2timetable,timetable,RowTimes,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn array2timetable_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readtimetable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let (readtable_args, timetable_args) = split_readtimetable_options(&rest)?;
    let table = super::io::readtable_builtin(path, readtable_args).await?;
    table2timetable_builtin(table, timetable_args).await
}
