use super::*;

#[derive(Default)]
pub(in crate::builtins::table) struct Array2TimetableOptions {
    pub(in crate::builtins::table) row_times: Option<Value>,
    pub(in crate::builtins::table) sample_rate: Option<Value>,
    pub(in crate::builtins::table) time_step: Option<Value>,
    pub(in crate::builtins::table) start_time: Option<Value>,
    pub(in crate::builtins::table) variable_names: Option<Vec<String>>,
    pub(in crate::builtins::table) dimension_names: Option<Vec<String>>,
}

#[derive(Default)]
pub(in crate::builtins::table) struct Table2TimetableOptions {
    pub(in crate::builtins::table) row_times: Option<Value>,
    pub(in crate::builtins::table) sample_rate: Option<Value>,
    pub(in crate::builtins::table) time_step: Option<Value>,
    pub(in crate::builtins::table) start_time: Option<Value>,
}

pub(in crate::builtins::table) fn parse_table2timetable_options(
    args: &[Value],
) -> BuiltinResult<Table2TimetableOptions> {
    let mut options = Table2TimetableOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "table2timetable: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "table2timetable option")?;
        let value = args[idx + 1].clone();
        if name.eq_ignore_ascii_case("RowTimes") {
            options.row_times = Some(value);
        } else if name.eq_ignore_ascii_case("SampleRate") {
            options.sample_rate = Some(value);
        } else if name.eq_ignore_ascii_case("TimeStep") {
            options.time_step = Some(value);
        } else if name.eq_ignore_ascii_case("StartTime") {
            options.start_time = Some(value);
        } else {
            return Err(invalid_argument(format!(
                "table2timetable: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    let timing_forms = usize::from(options.row_times.is_some())
        + usize::from(options.sample_rate.is_some())
        + usize::from(options.time_step.is_some());
    if timing_forms > 1 {
        return Err(invalid_argument(
            "table2timetable: specify at most one of RowTimes, SampleRate, or TimeStep",
        ));
    }
    if options.start_time.is_some() && options.sample_rate.is_none() && options.time_step.is_none()
    {
        return Err(invalid_argument(
            "table2timetable: StartTime requires SampleRate or TimeStep",
        ));
    }
    Ok(options)
}

pub(in crate::builtins::table) fn table2timetable_generated_row_times(
    options: &Table2TimetableOptions,
    height: usize,
) -> BuiltinResult<Option<Value>> {
    if options.sample_rate.is_none() && options.time_step.is_none() {
        return Ok(None);
    }
    let array_options = Array2TimetableOptions {
        row_times: None,
        sample_rate: options.sample_rate.clone(),
        time_step: options.time_step.clone(),
        start_time: options.start_time.clone(),
        variable_names: None,
        dimension_names: None,
    };
    array2timetable_row_times(&array_options, height).map(Some)
}

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

pub(in crate::builtins::table) fn parse_array2timetable_options(
    args: &[Value],
) -> BuiltinResult<Array2TimetableOptions> {
    let mut options = Array2TimetableOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "array2timetable: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "array2timetable option")?;
        let value = args[idx + 1].clone();
        if name.eq_ignore_ascii_case("RowTimes") {
            options.row_times = Some(value);
        } else if name.eq_ignore_ascii_case("SampleRate") {
            options.sample_rate = Some(value);
        } else if name.eq_ignore_ascii_case("TimeStep") {
            options.time_step = Some(value);
        } else if name.eq_ignore_ascii_case("StartTime") {
            options.start_time = Some(value);
        } else if name.eq_ignore_ascii_case("VariableNames") {
            options.variable_names = Some(raw_variable_name_list(&value)?);
        } else if name.eq_ignore_ascii_case("DimensionNames") {
            options.dimension_names = Some(string_list(&value)?);
        } else {
            return Err(invalid_argument(format!(
                "array2timetable: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    let timing_forms = usize::from(options.row_times.is_some())
        + usize::from(options.sample_rate.is_some())
        + usize::from(options.time_step.is_some());
    if timing_forms != 1 {
        return Err(invalid_argument(
            "array2timetable: specify exactly one of RowTimes, SampleRate, or TimeStep",
        ));
    }
    if options.start_time.is_some() && options.sample_rate.is_none() && options.time_step.is_none()
    {
        return Err(invalid_argument(
            "array2timetable: StartTime requires SampleRate or TimeStep",
        ));
    }
    Ok(options)
}

pub(in crate::builtins::table) fn array2timetable_row_times(
    options: &Array2TimetableOptions,
    height: usize,
) -> BuiltinResult<Value> {
    if let Some(row_times) = &options.row_times {
        validate_explicit_row_times(row_times, height)?;
        return Ok(row_times.clone());
    }
    if let Some(sample_rate) = &options.sample_rate {
        let sample_rate = numeric_sample_rate(sample_rate)?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(invalid_argument(
                "array2timetable: SampleRate must be a positive finite numeric scalar",
            ));
        }
        return fixed_step_row_times(
            height,
            1.0 / (sample_rate * 86_400.0),
            options.start_time.as_ref(),
        );
    }
    let time_step = options
        .time_step
        .as_ref()
        .expect("parser requires one timing form");
    if crate::builtins::duration::is_duration_object(time_step) {
        let tensor = crate::builtins::duration::duration_tensor_from_duration_value(time_step)?;
        if tensor.len() != 1 {
            return Err(invalid_argument(
                "array2timetable: TimeStep must be a scalar",
            ));
        }
        let step_days = crate::builtins::common::tensor::tensor_value_f64(&tensor, 0);
        if !step_days.is_finite() || step_days <= 0.0 {
            return Err(invalid_argument(
                "array2timetable: TimeStep must be a positive finite duration scalar",
            ));
        }
        return fixed_step_row_times(height, step_days, options.start_time.as_ref());
    }
    if crate::builtins::datetime::is_calendar_duration_object(time_step) {
        let start = options.start_time.as_ref().ok_or_else(|| {
            invalid_argument("array2timetable: calendar TimeStep requires a datetime StartTime")
        })?;
        if !crate::builtins::datetime::is_datetime_object(start) {
            return Err(invalid_argument(
                "array2timetable: calendar TimeStep requires a datetime StartTime",
            ));
        }
        return crate::builtins::datetime::datetime_row_times_from_calendar_step(
            start, time_step, height,
        )
        .map_err(|error| invalid_argument(error.message));
    }
    Err(invalid_argument(
        "array2timetable: TimeStep must be a duration or calendarDuration scalar",
    ))
}

pub(in crate::builtins::table) fn validate_explicit_row_times(
    value: &Value,
    height: usize,
) -> BuiltinResult<()> {
    let tensor = if crate::builtins::datetime::is_datetime_object(value) {
        crate::builtins::datetime::serials_from_datetime_value(value)?
    } else if crate::builtins::duration::is_duration_object(value) {
        crate::builtins::duration::duration_tensor_from_duration_value(value)?
    } else {
        return Err(invalid_argument(
            "array2timetable: RowTimes must be a datetime or duration vector",
        ));
    };
    if tensor.len() != height || tensor.cols() != 1 {
        return Err(invalid_argument(format!(
            "array2timetable: RowTimes must be a {height}-by-1 vector"
        )));
    }
    Ok(())
}

fn numeric_sample_rate(value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => positive_integer_sample_rate(value),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            if let Some(storage) = tensor.integer_storage() {
                return positive_integer_sample_rate(
                    &storage.value_at(0).expect("one-element integer storage"),
                );
            }
            Ok(crate::builtins::common::tensor::tensor_value_f64(tensor, 0))
        }
        Value::Tensor(tensor) => Err(invalid_argument(format!(
            "array2timetable: SampleRate must be scalar, got {} elements",
            tensor.len()
        ))),
        _ => Err(invalid_argument(
            "array2timetable: SampleRate must be a numeric scalar",
        )),
    }
}

fn positive_integer_sample_rate(value: &runmat_builtins::IntValue) -> BuiltinResult<f64> {
    if value.try_to_u64().is_some_and(|value| value > 0) {
        Ok(value.to_f64())
    } else {
        Err(invalid_argument(
            "array2timetable: SampleRate must be a positive finite numeric scalar",
        ))
    }
}

fn fixed_step_row_times(
    height: usize,
    step_days: f64,
    start_time: Option<&Value>,
) -> BuiltinResult<Value> {
    enum TimeKind {
        Duration(String),
        Datetime(String),
    }
    let (start_days, kind) = match start_time {
        None => (0.0, TimeKind::Duration("hh:mm:ss".to_string())),
        Some(value) if crate::builtins::duration::is_duration_object(value) => {
            let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
            if tensor.len() != 1 {
                return Err(invalid_argument(
                    "array2timetable: StartTime must be a scalar",
                ));
            }
            (
                crate::builtins::common::tensor::tensor_value_f64(&tensor, 0),
                TimeKind::Duration(crate::builtins::duration::duration_format_from_value(value)),
            )
        }
        Some(value) if crate::builtins::datetime::is_datetime_object(value) => {
            let tensor = crate::builtins::datetime::serials_from_datetime_value(value)?;
            if tensor.len() != 1 {
                return Err(invalid_argument(
                    "array2timetable: StartTime must be a scalar",
                ));
            }
            (
                crate::builtins::common::tensor::tensor_value_f64(&tensor, 0),
                TimeKind::Datetime(crate::builtins::datetime::datetime_format_from_value(value)),
            )
        }
        Some(_) => {
            return Err(invalid_argument(
                "array2timetable: StartTime must be a datetime or duration scalar",
            ))
        }
    };
    if !start_days.is_finite() {
        return Err(invalid_argument(
            "array2timetable: StartTime must be finite",
        ));
    }
    let values = (0..height)
        .map(|index| start_days + step_days * index as f64)
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_argument(
            "array2timetable: generated row times are outside the supported range",
        ));
    }
    let tensor = Tensor::new(values, vec![height, 1])
        .map_err(|error| invalid_argument(error.to_string()))?;
    match kind {
        TimeKind::Duration(format) => {
            crate::builtins::duration::duration_object_from_days_tensor(tensor, format)
        }
        TimeKind::Datetime(format) => {
            crate::builtins::datetime::datetime_object_from_serial_tensor(tensor, format)
        }
    }
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
