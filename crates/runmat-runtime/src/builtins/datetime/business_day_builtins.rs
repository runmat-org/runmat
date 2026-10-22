use super::capabilities::{BUILTIN_NAME, DEFAULT_DATE_FORMAT, MAX_BUSDAYS_OUTPUT_LEN};
use super::*;

fn ensure_business_day_input_policy(
    name: &'static str,
    inputs: &[&Value],
    date_inputs: &[&Value],
    numeric_inputs: &[&Value],
) -> BuiltinResult<()> {
    if inputs
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::BUSINESS_DAY_RESIDENT_INPUT_EXTENSION,
            name,
        )?;
    }
    if date_inputs
        .iter()
        .any(|value| is_typed_legacy_numeric(value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::BUSINESS_DAY_TYPED_LEGACY_SERIAL_EXTENSION,
            name,
        )?;
    }
    if numeric_inputs
        .iter()
        .any(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::BUSINESS_DAY_LOGICAL_NUMERIC_INPUT_EXTENSION,
            name,
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BusinessDayPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Semiannual,
    Annual,
}

impl BusinessDayPeriod {
    fn period_start(self, date: NaiveDate) -> BuiltinResult<NaiveDate> {
        let (year, month, day) = match self {
            Self::Daily => return Ok(date),
            Self::Weekly => {
                return date
                    .checked_sub_signed(Duration::days(i64::from(
                        (date.weekday().num_days_from_monday() + 2) % 7,
                    )))
                    .ok_or_else(|| datetime_error("busdays: date is outside supported range"));
            }
            Self::Monthly => (date.year(), date.month(), 1),
            Self::Quarterly => (date.year(), ((date.month() - 1) / 3) * 3 + 1, 1),
            Self::Semiannual => (date.year(), ((date.month() - 1) / 6) * 6 + 1, 1),
            Self::Annual => (date.year(), 1, 1),
        };
        NaiveDate::from_ymd_opt(year, month, day)
            .ok_or_else(|| datetime_error("busdays: date is outside supported range"))
    }

    fn period_end(self, date: NaiveDate) -> BuiltinResult<NaiveDate> {
        let (year, month, day) = match self {
            Self::Daily => return Ok(date),
            Self::Weekly => {
                let until_friday = (4 + 7 - date.weekday().num_days_from_monday()) % 7;
                return date
                    .checked_add_signed(Duration::days(i64::from(until_friday)))
                    .ok_or_else(|| datetime_error("busdays: date is outside supported range"));
            }
            Self::Monthly => (
                date.year(),
                date.month(),
                days_in_month(date.year(), date.month())?,
            ),
            Self::Quarterly => {
                let month = ((date.month() - 1) / 3 + 1) * 3;
                (date.year(), month, days_in_month(date.year(), month)?)
            }
            Self::Semiannual => {
                let month = ((date.month() - 1) / 6 + 1) * 6;
                (date.year(), month, days_in_month(date.year(), month)?)
            }
            Self::Annual => (date.year(), 12, 31),
        };
        NaiveDate::from_ymd_opt(year, month, day)
            .ok_or_else(|| datetime_error("busdays: date is outside supported range"))
    }

    fn previous_period_start(self, date: NaiveDate) -> BuiltinResult<Option<NaiveDate>> {
        Ok(self.period_start(date)?.pred_opt())
    }
}

fn business_day_period(value: Option<&Value>, context: &str) -> BuiltinResult<BusinessDayPeriod> {
    let Some(value) = value else {
        return Ok(BusinessDayPeriod::Daily);
    };
    if matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        return match scalar_text(value, "business-day frequency")?
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "daily" | "d" => Ok(BusinessDayPeriod::Daily),
            "weekly" | "w" => Ok(BusinessDayPeriod::Weekly),
            "monthly" | "m" => Ok(BusinessDayPeriod::Monthly),
            "quarterly" | "q" => Ok(BusinessDayPeriod::Quarterly),
            "semiannual" | "s" => Ok(BusinessDayPeriod::Semiannual),
            "annual" | "a" => Ok(BusinessDayPeriod::Annual),
            _ => Err(datetime_error(format!(
                "{context}: unsupported business-day frequency"
            ))),
        };
    }
    let number = tensor_from_numeric(value.clone(), context)?;
    if !tensor::is_scalar_tensor(&number) {
        return Err(datetime_error(format!(
            "{context}: frequency must be scalar"
        )));
    }
    match tensor::tensor_value_f64(&number, 0) {
        1.0 => Ok(BusinessDayPeriod::Daily),
        2.0 => Ok(BusinessDayPeriod::Weekly),
        3.0 => Ok(BusinessDayPeriod::Monthly),
        4.0 => Ok(BusinessDayPeriod::Quarterly),
        5.0 => Ok(BusinessDayPeriod::Semiannual),
        6.0 => Ok(BusinessDayPeriod::Annual),
        _ => Err(datetime_error(format!(
            "{context}: unsupported business-day frequency"
        ))),
    }
}

#[runmat_macros::runtime_builtin(
    name = "isbusday",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime",
    context = true
)]
pub(super) async fn isbusday_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(datetime_error(
            "isbusday: expected a date, optional holidays, and optional weekend mask",
        ));
    }
    ensure_business_day_input_policy(
        "isbusday",
        &std::iter::once(&value)
            .chain(rest.iter())
            .collect::<Vec<_>>(),
        &[Some(&value), rest.first()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        &[&value],
    )?;
    let value = gather_in_runtime(runtime.as_ref(), &value)
        .await
        .map_err(|err| datetime_error(format!("isbusday: {}", err.message())))?;
    let rest = gather_retained_args(rest, runtime.as_ref()).await?;
    let serials = numeric_or_datetime_serial_tensor(runtime.as_ref(), value, "isbusday")?;
    let (start_key, end_key) = date_key_range(&[&serials])?;
    let holidays = holiday_set_from_optional_or_default(
        runtime.as_ref(),
        rest.first().cloned(),
        "isbusday",
        start_key,
        end_key,
    )?;
    let weekend = weekend_mask(rest.get(1).cloned(), "isbusday")?;
    let serial_values = tensor::tensor_values_f64_cow(&serials);
    let mut out = Vec::with_capacity(serial_values.len());
    for serial in serial_values.iter() {
        out.push(u8::from(is_business_day_with_weekend(
            serial_date_key(*serial)?,
            &holidays,
            &weekend,
        )?));
    }
    if out.len() == 1 {
        Ok(Value::Bool(out[0] != 0))
    } else {
        runmat_value::LogicalArray::new(
            out,
            tensor::default_shape_for(serials.shape(), serial_values.len()),
        )
        .map(Value::LogicalArray)
        .map_err(|err| datetime_error(format!("isbusday: {err}")))
    }
}

#[runmat_macros::runtime_builtin(
    name = "holidays",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime",
    context = true
)]
pub(super) async fn holidays_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if args.len() > 3 {
        return Err(datetime_error(
            "holidays: expected zero, one, two, or three inputs",
        ));
    }
    if args.len() < 2 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::HOLIDAYS_LOCAL_YEAR_EXTENSION,
            "holidays",
        )?;
    }
    let date_inputs: Vec<&Value> = if args.len() >= 2 {
        args.iter().take(3).collect()
    } else {
        Vec::new()
    };
    let numeric_inputs: Vec<&Value> = if args.len() == 1 {
        args.iter().collect()
    } else {
        date_inputs.clone()
    };
    ensure_business_day_input_policy(
        "holidays",
        &args.iter().collect::<Vec<_>>(),
        &date_inputs,
        &numeric_inputs,
    )?;
    let output_datetime = args.len() < 2 || args.iter().any(is_datetime_object);
    let args = gather_retained_args(args, runtime.as_ref()).await?;
    let keys = match args.len() {
        0 => {
            let year = current_naive_local(runtime.as_ref())?.year();
            let start = key_from_date(NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
            let end = key_from_date(NaiveDate::from_ymd_opt(year, 12, 31).unwrap());
            holiday_keys_between(start, end)?
        }
        1 => {
            let tensor = tensor_from_numeric(args[0].clone(), "holidays")?;
            if !tensor::is_scalar_tensor(&tensor) {
                return Err(datetime_error("holidays: Year must be a scalar integer"));
            }
            let year = tensor::tensor_value_f64(&tensor, 0);
            if !year.is_finite() || year.fract() != 0.0 || !(1000.0..=9999.0).contains(&year) {
                return Err(datetime_error(
                    "holidays: Year must be an integer from 1000 through 9999",
                ));
            }
            let year = year as i32;
            let start = key_from_date(NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
            let end = key_from_date(NaiveDate::from_ymd_opt(year, 12, 31).unwrap());
            holiday_keys_between(start, end)?
        }
        2 | 3 => {
            let start =
                numeric_or_datetime_serial_tensor(runtime.as_ref(), args[0].clone(), "holidays")?;
            let end =
                numeric_or_datetime_serial_tensor(runtime.as_ref(), args[1].clone(), "holidays")?;
            if tensor::tensor_element_len(&start) != 1 || tensor::tensor_element_len(&end) != 1 {
                return Err(datetime_error(
                    "holidays: start and end dates must be scalar",
                ));
            }
            let start_key = serial_date_key(tensor::tensor_value_f64(&start, 0))?;
            let end_key = serial_date_key(tensor::tensor_value_f64(&end, 0))?;
            if let Some(alternate) = args.get(2) {
                let alternate = numeric_or_datetime_serial_tensor(
                    runtime.as_ref(),
                    alternate.clone(),
                    "holidays",
                )?;
                let mut keys = tensor::tensor_values_f64_cow(&alternate)
                    .iter()
                    .map(|serial| serial_date_key(*serial))
                    .collect::<BuiltinResult<Vec<_>>>()?;
                keys.retain(|key| *key >= start_key.min(end_key) && *key <= start_key.max(end_key));
                keys.sort_unstable();
                keys.dedup();
                keys
            } else {
                holiday_keys_between(start_key, end_key)?
            }
        }
        _ => {
            return Err(datetime_error(
                "holidays: expected zero, one, two, or three inputs",
            ))
        }
    };
    let len = keys.len();
    let serials: Vec<f64> = keys.into_iter().map(|key| key as f64).collect();
    if output_datetime {
        datetime_object_from_serials(runtime.as_ref(), serials, vec![len, 1], DEFAULT_DATE_FORMAT)
    } else {
        Tensor::new(serials, vec![len, 1])
            .map(Value::Tensor)
            .map_err(|err| datetime_error(format!("holidays: {err}")))
    }
}

#[runmat_macros::runtime_builtin(
    name = "busdays",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime",
    context = true
)]
pub(super) async fn busdays_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    start: Value,
    end: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(datetime_error(
            "busdays: expected a period and optional holiday list",
        ));
    }
    ensure_business_day_input_policy(
        "busdays",
        &std::iter::once(&start)
            .chain(std::iter::once(&end))
            .chain(rest.iter())
            .collect::<Vec<_>>(),
        &[Some(&start), Some(&end), rest.get(1)]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        &[Some(&start), Some(&end), rest.first(), rest.get(1)]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    )?;
    let start = gather_in_runtime(runtime.as_ref(), &start)
        .await
        .map_err(|err| datetime_error(format!("busdays: {}", err.message())))?;
    let end = gather_in_runtime(runtime.as_ref(), &end)
        .await
        .map_err(|err| datetime_error(format!("busdays: {}", err.message())))?;
    let rest = gather_retained_args(rest, runtime.as_ref()).await?;
    let period = business_day_period(rest.first(), "busdays")?;
    let output_datetime = is_datetime_object(&start)
        || is_datetime_object(&end)
        || rest.get(1).is_some_and(is_datetime_object);
    let start = numeric_or_datetime_serial_tensor(runtime.as_ref(), start, "busdays")?;
    let end = numeric_or_datetime_serial_tensor(runtime.as_ref(), end, "busdays")?;
    if tensor::tensor_element_len(&start) != 1 || tensor::tensor_element_len(&end) != 1 {
        return Err(datetime_error(
            "busdays: start and end dates must be scalar",
        ));
    }
    let mut key = serial_date_key(tensor::tensor_value_f64(&start, 0))?;
    let end_key = serial_date_key(tensor::tensor_value_f64(&end, 0))?;
    let span = key
        .max(end_key)
        .checked_sub(key.min(end_key))
        .and_then(|delta| delta.checked_add(1))
        .ok_or_else(|| datetime_error("busdays: date range is outside supported range"))?;
    if span > MAX_BUSDAYS_OUTPUT_LEN {
        return Err(datetime_error(format!(
            "busdays: output would exceed {MAX_BUSDAYS_OUTPUT_LEN} dates"
        )));
    }
    let calendar_start = if period == BusinessDayPeriod::Daily {
        key.min(end_key)
    } else {
        key_from_date(period.period_start(date_from_key(key.min(end_key))?)?)
    };
    let calendar_end = if period == BusinessDayPeriod::Daily {
        key.max(end_key)
    } else {
        key_from_date(period.period_end(date_from_key(key.max(end_key))?)?)
    };
    let holidays = holiday_set_from_optional_or_default(
        runtime.as_ref(),
        rest.get(1).cloned(),
        "busdays",
        calendar_start,
        calendar_end,
    )?;
    let step = if key <= end_key { 1 } else { -1 };
    let mut out = Vec::new();
    if period == BusinessDayPeriod::Daily {
        loop {
            if is_business_day_key(key, &holidays)? {
                out.push(key as f64);
            }
            if key == end_key {
                break;
            }
            key = key
                .checked_add(step)
                .ok_or_else(|| datetime_error("busdays: date range is outside supported range"))?;
        }
    } else {
        let first_period = period.period_end(date_from_key(key)?)?;
        let last_period = period.period_end(date_from_key(end_key)?)?;
        let mut period_end = first_period;
        loop {
            let mut candidate = key_from_date(period_end);
            while !is_business_day_key(candidate, &holidays)? {
                candidate = candidate.checked_sub(1).ok_or_else(|| {
                    datetime_error("busdays: date range is outside supported range")
                })?;
            }
            out.push(candidate as f64);
            if period_end == last_period {
                break;
            }
            let next = if step > 0 {
                period_end.succ_opt()
            } else {
                period.previous_period_start(period_end)?
            }
            .ok_or_else(|| datetime_error("busdays: date range is outside supported range"))?;
            period_end = period.period_end(next)?;
        }
    }
    let len = out.len();
    if output_datetime {
        return datetime_object_from_serials(
            runtime.as_ref(),
            out,
            vec![len, 1],
            DEFAULT_DATE_FORMAT,
        );
    }
    Tensor::new(out, vec![len, 1])
        .map(Value::Tensor)
        .map_err(|err| datetime_error(format!("busdays: {err}")))
}

#[runmat_macros::runtime_builtin(
    name = "days252bus",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime",
    context = true
)]
pub(super) async fn days252bus_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    start: Value,
    end: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(datetime_error(
            "days252bus: expected at most one holiday list",
        ));
    }
    ensure_business_day_input_policy(
        "days252bus",
        &std::iter::once(&start)
            .chain(std::iter::once(&end))
            .chain(rest.iter())
            .collect::<Vec<_>>(),
        &[Some(&start), Some(&end), rest.first()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        &[Some(&start), Some(&end), rest.first()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    )?;
    let start = gather_in_runtime(runtime.as_ref(), &start)
        .await
        .map_err(|err| datetime_error(format!("days252bus: {}", err.message())))?;
    let end = gather_in_runtime(runtime.as_ref(), &end)
        .await
        .map_err(|err| datetime_error(format!("days252bus: {}", err.message())))?;
    let rest = gather_retained_args(rest, runtime.as_ref()).await?;
    let starts = numeric_or_datetime_serial_tensor(runtime.as_ref(), start, "days252bus")?;
    let ends = numeric_or_datetime_serial_tensor(runtime.as_ref(), end, "days252bus")?;
    let (start_key, end_key) = date_key_range(&[&starts, &ends])?;
    let holidays = holiday_set_from_optional_or_default(
        runtime.as_ref(),
        rest.first().cloned(),
        "days252bus",
        start_key,
        end_key,
    )?;
    let (start_data, end_data, shape) =
        tensor::binary_numeric_tensors(&starts, &ends, "days252bus", BUILTIN_NAME)?;
    let counts = start_data
        .iter()
        .zip(end_data.iter())
        .map(|(start, end)| {
            Ok(
                count_business_days(serial_date_key(*start)?, serial_date_key(*end)?, &holidays)?
                    as f64,
            )
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    tensor_or_scalar(counts, shape)
}

#[runmat_macros::runtime_builtin(
    name = "daysdif",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime",
    context = true
)]
pub(super) async fn daysdif_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    start: Value,
    end: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(datetime_error(
            "daysdif: expected at most one basis argument",
        ));
    }
    ensure_business_day_input_policy(
        "daysdif",
        &std::iter::once(&start)
            .chain(std::iter::once(&end))
            .chain(rest.iter())
            .collect::<Vec<_>>(),
        &[&start, &end],
        &[Some(&start), Some(&end), rest.first()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    )?;
    let start = gather_in_runtime(runtime.as_ref(), &start)
        .await
        .map_err(|err| datetime_error(format!("daysdif: {}", err.message())))?;
    let end = gather_in_runtime(runtime.as_ref(), &end)
        .await
        .map_err(|err| datetime_error(format!("daysdif: {}", err.message())))?;
    let rest = gather_retained_args(rest, runtime.as_ref()).await?;
    let basis = match rest.first() {
        None => 0,
        Some(value) => {
            let tensor = tensor_from_numeric(value.clone(), "daysdif")?;
            if !tensor::is_scalar_tensor(&tensor) {
                return Err(datetime_error("daysdif: Basis must be a scalar integer"));
            }
            let value = tensor::tensor_value_f64(&tensor, 0);
            if !value.is_finite() || value.fract() != 0.0 || !matches!(value as i64, 0 | 1) {
                return Err(datetime_error(
                    "daysdif: only Basis 0 (actual) and 1 (30/360 SIA) are supported",
                ));
            }
            value as i64
        }
    };
    let starts = numeric_or_datetime_serial_tensor(runtime.as_ref(), start, "daysdif")?;
    let ends = numeric_or_datetime_serial_tensor(runtime.as_ref(), end, "daysdif")?;
    let (start_data, end_data, shape) =
        tensor::binary_numeric_tensors(&starts, &ends, "daysdif", BUILTIN_NAME)?;
    let out = start_data
        .iter()
        .zip(end_data.iter())
        .map(|(start, end)| {
            let start_key = serial_date_key(*start)?;
            let end_key = serial_date_key(*end)?;
            if basis == 1 {
                let s = date_from_key(start_key)?;
                let e = date_from_key(end_key)?;
                let start_last_february = s.month() == 2 && s.day() == days_in_month(s.year(), 2)?;
                let end_last_february = e.month() == 2 && e.day() == days_in_month(e.year(), 2)?;
                let sd = if s.day() == 31 || start_last_february {
                    30
                } else {
                    s.day()
                } as i32;
                let ed =
                    if (start_last_february && end_last_february) || (sd == 30 && e.day() == 31) {
                        30
                    } else {
                        e.day()
                    } as i32;
                Ok(f64::from(
                    (e.year() - s.year()) * 360
                        + (e.month() as i32 - s.month() as i32) * 30
                        + (ed - sd),
                ))
            } else {
                Ok((end_key - start_key) as f64)
            }
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "fbusdate",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime",
    context = true
)]
pub(super) async fn fbusdate_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    year: Value,
    month: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    month_business_date_builtin(
        runtime.as_ref(),
        year,
        month,
        rest,
        "fbusdate",
        first_business_day_with_weekend,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "lbusdate",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime",
    context = true
)]
pub(super) async fn lbusdate_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    year: Value,
    month: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    month_business_date_builtin(
        runtime.as_ref(),
        year,
        month,
        rest,
        "lbusdate",
        last_business_day_with_weekend,
    )
    .await
}

type MonthBusinessDate = fn(i32, u32, &HashSet<i64>, &[bool; 7]) -> BuiltinResult<i64>;

async fn month_business_date_builtin(
    runtime: Option<&crate::context::RuntimeContext>,
    year: Value,
    month: Value,
    rest: crate::call::BuiltinArguments,
    name: &'static str,
    find: MonthBusinessDate,
) -> BuiltinResult<Value> {
    if rest.len() > 3 {
        return Err(datetime_error(format!(
            "{name}: expected optional Holiday, Weekend, and outputType"
        )));
    }
    ensure_business_day_input_policy(
        name,
        &std::iter::once(&year)
            .chain(std::iter::once(&month))
            .chain(rest.iter())
            .collect::<Vec<_>>(),
        &rest.first().into_iter().collect::<Vec<_>>(),
        &[Some(&year), Some(&month), rest.first()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    )?;
    let output_datetime = match rest.get(2) {
        Some(value) => match scalar_text(value, "outputType")?
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "datenum" => false,
            "datetime" => true,
            _ => {
                return Err(datetime_error(format!(
                    "{name}: outputType must be datenum or datetime"
                )))
            }
        },
        None => rest.first().is_some_and(is_datetime_object),
    };
    let year = gather_in_runtime(runtime, &year)
        .await
        .map_err(|err| datetime_error(format!("{name}: {}", err.message())))?;
    let month = gather_in_runtime(runtime, &month)
        .await
        .map_err(|err| datetime_error(format!("{name}: {}", err.message())))?;
    let rest = gather_retained_args(rest, runtime).await?;
    let years = component_tensor(year, name)?;
    let months = component_tensor(month, name)?;
    let (year_data, month_data, shape) =
        tensor::binary_numeric_tensors(&years, &months, name, BUILTIN_NAME)?;
    let weekend = weekend_mask(rest.get(1).cloned(), name)?;
    let mut bounds: Option<(i64, i64)> = None;
    for (year, month) in year_data.iter().zip(month_data.iter()) {
        let year = round_component(*year, "year", -262_000, 262_000)? as i32;
        let month = round_component(*month, "month", 1, 12)? as u32;
        let first = key_from_date(
            NaiveDate::from_ymd_opt(year, month, 1)
                .ok_or_else(|| datetime_error(format!("{name}: invalid year/month")))?,
        );
        let last = key_from_date(
            NaiveDate::from_ymd_opt(year, month, days_in_month(year, month)?)
                .ok_or_else(|| datetime_error(format!("{name}: invalid year/month")))?,
        );
        bounds = Some(match bounds {
            Some((min, max)) => (min.min(first), max.max(last)),
            None => (first, last),
        });
    }
    let (min, max) = bounds.unwrap_or((0, 0));
    let holidays =
        holiday_set_from_optional_or_default(runtime, rest.first().cloned(), name, min, max)?;
    let out = year_data
        .iter()
        .zip(month_data.iter())
        .map(|(year, month)| {
            Ok(find(
                round_component(*year, "year", -262_000, 262_000)? as i32,
                round_component(*month, "month", 1, 12)? as u32,
                &holidays,
                &weekend,
            )? as f64)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    if output_datetime {
        datetime_object_from_serials(runtime, out, shape, DEFAULT_DATE_FORMAT)
    } else {
        tensor_or_scalar(out, shape)
    }
}
#[cfg(test)]
pub(crate) async fn isbusday_builtin_source(
    value: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    isbusday_builtin(
        None,
        value,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}
#[cfg(test)]
pub(crate) async fn holidays_builtin_source(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    holidays_builtin(
        None,
        crate::call::BuiltinArguments::from_borrowed(&args, None)?,
    )
    .await
}
#[cfg(test)]
pub(crate) async fn busdays_builtin_source(
    start: Value,
    end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    busdays_builtin(
        None,
        start,
        end,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}
#[cfg(test)]
pub(crate) async fn days252bus_builtin_source(
    start: Value,
    end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    days252bus_builtin(
        None,
        start,
        end,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}
#[cfg(test)]
pub(crate) async fn daysdif_builtin_source(
    start: Value,
    end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    daysdif_builtin(
        None,
        start,
        end,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}
#[cfg(test)]
pub(crate) async fn fbusdate_builtin_source(
    year: Value,
    month: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    fbusdate_builtin(
        None,
        year,
        month,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}
#[cfg(test)]
pub(crate) async fn lbusdate_builtin_source(
    year: Value,
    month: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    lbusdate_builtin(
        None,
        year,
        month,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}
