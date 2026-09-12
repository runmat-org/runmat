use super::capabilities::{BUILTIN_NAME, DEFAULT_DATE_FORMAT, MAX_BUSDAYS_OUTPUT_LEN};
use super::*;

#[runmat_macros::runtime_builtin(
    name = "isbusday",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return true where date values are business days.",
    keywords = "isbusday,business day,datetime,financial"
)]
pub(super) async fn isbusday_builtin(
    value: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("isbusday: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "isbusday: expected at most one holiday list",
        ));
    }
    let serials = numeric_or_datetime_serial_tensor(value, "isbusday")?;
    let (start_key, end_key) = date_key_range(&[&serials])?;
    let holidays = holiday_set_from_optional_or_default(
        rest.into_iter().next(),
        "isbusday",
        start_key,
        end_key,
    )?;
    let serial_values = tensor::tensor_values_f64_cow(&serials);
    let mut out = Vec::with_capacity(serial_values.len());
    for serial in serial_values.iter() {
        out.push(
            if is_business_day_key(serial_date_key(*serial)?, &holidays)? {
                1.0
            } else {
                0.0
            },
        );
    }
    tensor_or_scalar(
        out,
        tensor::default_shape_for(&serials.shape, serial_values.len()),
    )
}

#[runmat_macros::runtime_builtin(
    name = "holidays",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return exchange-style market holidays for a year or date range.",
    keywords = "holidays,business day,datetime,financial"
)]
pub(super) async fn holidays_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let args = gather_args(&args).await?;
    let keys = match args.len() {
        0 => {
            let year = current_naive_local().year();
            let start = key_from_date(NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
            let end = key_from_date(NaiveDate::from_ymd_opt(year, 12, 31).unwrap());
            holiday_keys_between(start, end)?
        }
        1 => {
            let tensor = tensor_from_numeric(args[0].clone(), "holidays");
            if let Ok(tensor) = tensor {
                let year =
                    tensor::is_scalar_tensor(&tensor).then(|| tensor::tensor_value_f64(&tensor, 0));
                if let Some(year) = year.filter(|year| (1000.0..=9999.0).contains(year)) {
                    let keys = market_holiday_keys_for_year(year.round() as i32)?;
                    let len = keys.len();
                    return datetime_object_from_serials(
                        keys.into_iter().map(|key| key as f64).collect(),
                        vec![len, 1],
                        DEFAULT_DATE_FORMAT,
                    );
                }
            }
            let serials = numeric_or_datetime_serial_tensor(args[0].clone(), "holidays")?;
            let year =
                date_from_key(serial_date_key(tensor::tensor_value_f64(&serials, 0))?)?.year();
            let start = key_from_date(NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
            let end = key_from_date(NaiveDate::from_ymd_opt(year, 12, 31).unwrap());
            holiday_keys_between(start, end)?
        }
        2 => {
            let start = numeric_or_datetime_serial_tensor(args[0].clone(), "holidays")?;
            let end = numeric_or_datetime_serial_tensor(args[1].clone(), "holidays")?;
            if tensor::tensor_element_len(&start) != 1 || tensor::tensor_element_len(&end) != 1 {
                return Err(datetime_error(
                    "holidays: start and end dates must be scalar",
                ));
            }
            holiday_keys_between(
                serial_date_key(tensor::tensor_value_f64(&start, 0))?,
                serial_date_key(tensor::tensor_value_f64(&end, 0))?,
            )?
        }
        _ => {
            return Err(datetime_error(
                "holidays: expected zero, one, or two inputs",
            ))
        }
    };
    let len = keys.len();
    datetime_object_from_serials(
        keys.into_iter().map(|key| key as f64).collect(),
        vec![len, 1],
        DEFAULT_DATE_FORMAT,
    )
}

#[runmat_macros::runtime_builtin(
    name = "busdays",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return serial date numbers for business days in a scalar date range.",
    keywords = "busdays,business day,datetime,financial"
)]
pub(super) async fn busdays_builtin(
    start: Value,
    end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let start = gather_if_needed_async(&start)
        .await
        .map_err(|err| datetime_error(format!("busdays: {}", err.message())))?;
    let end = gather_if_needed_async(&end)
        .await
        .map_err(|err| datetime_error(format!("busdays: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error("busdays: expected at most one holiday list"));
    }
    let start = numeric_or_datetime_serial_tensor(start, "busdays")?;
    let end = numeric_or_datetime_serial_tensor(end, "busdays")?;
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
    let holidays =
        holiday_set_from_optional_or_default(rest.into_iter().next(), "busdays", key, end_key)?;
    let step = if key <= end_key { 1 } else { -1 };
    let mut out = Vec::new();
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
    let len = out.len();
    Tensor::new(out, vec![len, 1])
        .map(Value::Tensor)
        .map_err(|err| datetime_error(format!("busdays: {err}")))
}

#[runmat_macros::runtime_builtin(
    name = "days252bus",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Count business days between date values using a 252-business-day calendar.",
    keywords = "days252bus,business day,datetime,financial"
)]
pub(super) async fn days252bus_builtin(
    start: Value,
    end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let start = gather_if_needed_async(&start)
        .await
        .map_err(|err| datetime_error(format!("days252bus: {}", err.message())))?;
    let end = gather_if_needed_async(&end)
        .await
        .map_err(|err| datetime_error(format!("days252bus: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "days252bus: expected at most one holiday list",
        ));
    }
    let starts = numeric_or_datetime_serial_tensor(start, "days252bus")?;
    let ends = numeric_or_datetime_serial_tensor(end, "days252bus")?;
    let (start_key, end_key) = date_key_range(&[&starts, &ends])?;
    let holidays = holiday_set_from_optional_or_default(
        rest.into_iter().next(),
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
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return date differences using actual or 30/360 day-count bases.",
    keywords = "daysdif,date difference,datetime,financial"
)]
pub(super) async fn daysdif_builtin(
    start: Value,
    end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let start = gather_if_needed_async(&start)
        .await
        .map_err(|err| datetime_error(format!("daysdif: {}", err.message())))?;
    let end = gather_if_needed_async(&end)
        .await
        .map_err(|err| datetime_error(format!("daysdif: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "daysdif: expected at most one basis argument",
        ));
    }
    let basis = rest
        .first()
        .map(|value| tensor_from_numeric(value.clone(), "daysdif"))
        .transpose()?
        .and_then(|tensor| {
            tensor::is_scalar_tensor(&tensor).then(|| tensor::tensor_value_f64(&tensor, 0))
        })
        .unwrap_or(0.0)
        .round() as i64;
    let starts = numeric_or_datetime_serial_tensor(start, "daysdif")?;
    let ends = numeric_or_datetime_serial_tensor(end, "daysdif")?;
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
                let sd = s.day().min(30) as i32;
                let ed = if sd == 30 { e.day().min(30) } else { e.day() } as i32;
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
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return first business day serial date numbers for month/year pairs.",
    keywords = "fbusdate,business day,datetime,financial"
)]
pub(super) async fn fbusdate_builtin(
    year: Value,
    month: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let year = gather_if_needed_async(&year)
        .await
        .map_err(|err| datetime_error(format!("fbusdate: {}", err.message())))?;
    let month = gather_if_needed_async(&month)
        .await
        .map_err(|err| datetime_error(format!("fbusdate: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "fbusdate: expected at most one holiday list",
        ));
    }
    let years = component_tensor(year, "fbusdate year")?;
    let months = component_tensor(month, "fbusdate month")?;
    let (year_data, month_data, shape) =
        tensor::binary_numeric_tensors(&years, &months, "fbusdate", BUILTIN_NAME)?;
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    for (year, month) in year_data.iter().zip(month_data.iter()) {
        let year = round_component(*year, "year", -262_000, 262_000)? as i32;
        let month = round_component(*month, "month", 1, 12)? as u32;
        min_key = min_key.min(key_from_date(
            NaiveDate::from_ymd_opt(year, month, 1)
                .ok_or_else(|| datetime_error("fbusdate: invalid year/month"))?,
        ));
        max_key = max_key.max(key_from_date(
            NaiveDate::from_ymd_opt(year, month, days_in_month(year, month)?)
                .ok_or_else(|| datetime_error("fbusdate: invalid year/month"))?,
        ));
    }
    let holidays = holiday_set_from_optional_or_default(
        rest.into_iter().next(),
        "fbusdate",
        min_key,
        max_key,
    )?;
    let out = year_data
        .iter()
        .zip(month_data.iter())
        .map(|(year, month)| {
            Ok(first_business_day_key(
                round_component(*year, "year", -262_000, 262_000)? as i32,
                round_component(*month, "month", 1, 12)? as u32,
                &holidays,
            )? as f64)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "lbusdate",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return last business day serial date numbers for month/year pairs.",
    keywords = "lbusdate,business day,datetime,financial"
)]
pub(super) async fn lbusdate_builtin(
    year: Value,
    month: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let year = gather_if_needed_async(&year)
        .await
        .map_err(|err| datetime_error(format!("lbusdate: {}", err.message())))?;
    let month = gather_if_needed_async(&month)
        .await
        .map_err(|err| datetime_error(format!("lbusdate: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "lbusdate: expected at most one holiday list",
        ));
    }
    let years = component_tensor(year, "lbusdate year")?;
    let months = component_tensor(month, "lbusdate month")?;
    let (year_data, month_data, shape) =
        tensor::binary_numeric_tensors(&years, &months, "lbusdate", BUILTIN_NAME)?;
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    for (year, month) in year_data.iter().zip(month_data.iter()) {
        let year = round_component(*year, "year", -262_000, 262_000)? as i32;
        let month = round_component(*month, "month", 1, 12)? as u32;
        min_key = min_key.min(key_from_date(
            NaiveDate::from_ymd_opt(year, month, 1)
                .ok_or_else(|| datetime_error("lbusdate: invalid year/month"))?,
        ));
        max_key = max_key.max(key_from_date(
            NaiveDate::from_ymd_opt(year, month, days_in_month(year, month)?)
                .ok_or_else(|| datetime_error("lbusdate: invalid year/month"))?,
        ));
    }
    let holidays = holiday_set_from_optional_or_default(
        rest.into_iter().next(),
        "lbusdate",
        min_key,
        max_key,
    )?;
    let out = year_data
        .iter()
        .zip(month_data.iter())
        .map(|(year, month)| {
            Ok(last_business_day_key(
                round_component(*year, "year", -262_000, 262_000)? as i32,
                round_component(*month, "month", 1, 12)? as u32,
                &holidays,
            )? as f64)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    tensor_or_scalar(out, shape)
}
