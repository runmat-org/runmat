use super::*;

pub(in crate::builtins::table) fn split_timetable_constructor_args(
    args: Vec<Value>,
) -> BuiltinResult<(Option<Value>, Vec<Value>, TableConstructorOptions)> {
    if args.is_empty() {
        return Ok((None, Vec::new(), TableConstructorOptions::default()));
    }
    let mut variables = Vec::new();
    let mut row_times = None;
    let mut options = TableConstructorOptions::default();
    let mut idx = 0usize;
    let has_explicit_row_times = args.iter().any(|value| {
        scalar_text(value, "timetable option")
            .map(|name| name.eq_ignore_ascii_case("RowTimes"))
            .unwrap_or(false)
    });
    if !has_explicit_row_times && args.len() > 1 && !is_timetable_option_token(&args[0]) {
        row_times = Some(args[0].clone());
        idx = 1;
    }
    while idx < args.len() {
        if idx + 1 < args.len() {
            if let Ok(name) = scalar_text(&args[idx], "timetable option") {
                if name.eq_ignore_ascii_case("RowTimes") {
                    row_times = Some(args[idx + 1].clone());
                    idx += 2;
                    continue;
                }
                if name.eq_ignore_ascii_case("VariableNames") {
                    options.variable_names = Some(variable_name_list(&args[idx + 1])?);
                    idx += 2;
                    continue;
                }
                if name.eq_ignore_ascii_case("RowNames") {
                    options.row_names = Some(string_list(&args[idx + 1])?);
                    idx += 2;
                    continue;
                }
            }
        }
        variables.push(args[idx].clone());
        idx += 1;
    }
    Ok((row_times, variables, options))
}

pub(in crate::builtins::table) fn is_timetable_option_token(value: &Value) -> bool {
    scalar_text(value, "timetable option")
        .map(|name| {
            name.eq_ignore_ascii_case("RowTimes")
                || name.eq_ignore_ascii_case("VariableNames")
                || name.eq_ignore_ascii_case("RowNames")
        })
        .unwrap_or(false)
}

pub(in crate::builtins::table) fn is_time_like_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Object(obj) if obj.is_class("datetime") || obj.is_class("duration")
    )
}

pub(in crate::builtins::table) fn parse_timetable_options(
    args: &[Value],
    context: &str,
) -> BuiltinResult<(Option<Value>, TableConstructorOptions)> {
    let mut row_times = None;
    let mut table_options = TableConstructorOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let name = scalar_text(&args[idx], "timetable option")?;
        if name.eq_ignore_ascii_case("RowTimes") {
            row_times = Some(args[idx + 1].clone());
        } else if name.eq_ignore_ascii_case("VariableNames") {
            table_options.variable_names = Some(variable_name_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("RowNames") {
            table_options.row_names = Some(string_list(&args[idx + 1])?);
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok((row_times, table_options))
}

pub(in crate::builtins::table) fn set_timetable_row_times(
    object: &mut ObjectInstance,
    row_times: Option<Value>,
) -> BuiltinResult<()> {
    let height = table_height(object)?;
    let times = row_times.unwrap_or_else(|| {
        Value::Tensor(
            Tensor::new(
                (1..=height).map(|idx| idx as f64).collect(),
                vec![height, 1],
            )
            .expect("generated row-time tensor shape is valid"),
        )
    });
    let rows = value_row_count(&times)?;
    if rows != height {
        return Err(invalid_variable(format!(
            "timetable: RowTimes has {rows} rows but timetable has {height}"
        )));
    }
    let mut props = table_public_properties(object)?;
    props.insert(ROW_TIMES, times);
    props.insert(
        DIMENSION_NAMES,
        Value::StringArray(
            StringArray::new(
                vec!["Time".to_string(), DEFAULT_VARIABLE_DIM_NAME.to_string()],
                vec![1, 2],
            )
            .map_err(invalid_variable)?,
        ),
    );
    sync_table_properties(object, props);
    Ok(())
}

pub(crate) fn timetable_row_times(object: &ObjectInstance) -> BuiltinResult<Option<Value>> {
    if !object.is_class(TIMETABLE_CLASS) {
        return Ok(None);
    }
    let props = table_public_properties(object)?;
    Ok(props.fields.get(ROW_TIMES).cloned())
}
