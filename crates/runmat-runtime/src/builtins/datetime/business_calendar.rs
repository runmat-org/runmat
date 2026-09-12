use super::capabilities::MAX_HOLIDAY_YEAR_SPAN;
use super::*;

pub(super) fn tensor_from_datevec_like(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor::integer_tensor_to_f64(tensor_from_numeric(value, context)?)
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let shape = tensor::default_shape_for(&tensor.shape, tensor::tensor_element_len(&tensor));
    let values = tensor::tensor_into_values_f64(tensor);
    let normalize = |rows: usize, cols: usize, data: Vec<f64>| -> BuiltinResult<Tensor> {
        if cols == 6 {
            return Tensor::new(data, vec![rows, 6])
                .map_err(|err| datetime_error(format!("{context}: {err}")));
        }
        let mut padded = vec![0.0; rows.saturating_mul(6)];
        for col in 0..3 {
            for row in 0..rows {
                padded[col * rows + row] = data[col * rows + row];
            }
        }
        Tensor::new(padded, vec![rows, 6])
            .map_err(|err| datetime_error(format!("{context}: {err}")))
    };
    if values.len() == 3 {
        return normalize(1, 3, values);
    }
    if values.len() == 6 {
        return normalize(1, 6, values);
    }
    if shape.len() >= 2 && (shape[1] == 3 || shape[1] == 6) {
        return normalize(shape[0], shape[1], values);
    }
    Err(datetime_error(format!(
        "{context}: expected a date vector with three or six columns"
    )))
}

pub(super) fn datenum_from_datevec_tensor(tensor: &Tensor, context: &str) -> BuiltinResult<Tensor> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    if cols != 6 {
        return Err(datetime_error(format!(
            "{context}: date vectors must have six columns"
        )));
    }
    let values = tensor::tensor_values_f64_cow(tensor);
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let component = |col: usize| values[col * rows + row];
        let naive = naive_from_components(
            component(0),
            component(1),
            component(2),
            component(3),
            component(4),
            component(5),
        )?;
        out.push(datenum_from_naive(naive));
    }
    Tensor::new(out, vec![rows, 1]).map_err(|err| datetime_error(format!("{context}: {err}")))
}

pub(super) fn char_array_from_rows(rows: &[String], context: &str) -> BuiltinResult<CharArray> {
    let width = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = vec![' '; rows.len().saturating_mul(width)];
    for (row_idx, row) in rows.iter().enumerate() {
        for (col, ch) in row.chars().enumerate() {
            data[row_idx * width + col] = ch;
        }
    }
    CharArray::new(data, rows.len(), width)
        .map_err(|err| datetime_error(format!("{context}: {err}")))
}

pub(super) fn broadcast_three_numeric_tensors(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    context: &str,
) -> BuiltinResult<Broadcast3> {
    let mut output_shape = Vec::new();
    let mut output_len = 1usize;
    for operand in [a, b, c] {
        let len = tensor::tensor_element_len(operand);
        if len == 1 {
            continue;
        }
        let shape = tensor::default_shape_for(&operand.shape, len);
        if output_shape.is_empty() {
            output_len = len;
            output_shape = shape;
        } else if len != output_len || shape != output_shape {
            return Err(datetime_error(format!(
                "{context}: operands must be scalar or have matching sizes"
            )));
        }
    }
    if output_shape.is_empty() {
        output_shape = vec![1, 1];
        output_len = 1;
    }

    let expand = |operand: &Tensor| -> BuiltinResult<Vec<f64>> {
        match tensor::tensor_element_len(operand) {
            1 => Ok(vec![tensor::tensor_value_f64(operand, 0); output_len]),
            len if len == output_len => Ok(tensor::tensor_values_f64(operand)),
            _ => Err(datetime_error(format!(
                "{context}: operands must be scalar or have matching sizes"
            ))),
        }
    };

    Ok((expand(a)?, expand(b)?, expand(c)?, output_shape))
}

pub(super) fn serial_date_key(serial: f64) -> BuiltinResult<i64> {
    if !serial.is_finite() {
        return Err(datetime_error("date values must be finite"));
    }
    let key = serial.floor();
    if key < i64::MIN as f64 || key > i64::MAX as f64 {
        return Err(datetime_error("date value is outside supported range"));
    }
    Ok(key as i64)
}

pub(super) fn date_from_key(key: i64) -> BuiltinResult<NaiveDate> {
    Ok(naive_from_datenum(key as f64)?.date())
}

pub(super) fn key_from_date(date: NaiveDate) -> i64 {
    datenum_from_naive(midnight(date)).floor() as i64
}

pub(super) fn observed_fixed_holiday(year: i32, month: u32, day: u32) -> BuiltinResult<i64> {
    let date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| datetime_error("holidays: invalid fixed holiday date"))?;
    let observed = match date.weekday() {
        Weekday::Sat => date - Duration::days(1),
        Weekday::Sun => date + Duration::days(1),
        _ => date,
    };
    Ok(key_from_date(observed))
}

pub(super) fn nth_weekday(year: i32, month: u32, weekday: Weekday, n: u32) -> BuiltinResult<i64> {
    let mut date = NaiveDate::from_ymd_opt(year, month, 1)
        .ok_or_else(|| datetime_error("holidays: invalid nth weekday month"))?;
    while date.weekday() != weekday {
        date += Duration::days(1);
    }
    date += Duration::days(i64::from(n.saturating_sub(1)) * 7);
    Ok(key_from_date(date))
}

pub(super) fn last_weekday(year: i32, month: u32, weekday: Weekday) -> BuiltinResult<i64> {
    let last_day = days_in_month(year, month)?;
    let mut date = NaiveDate::from_ymd_opt(year, month, last_day)
        .ok_or_else(|| datetime_error("holidays: invalid last weekday month"))?;
    while date.weekday() != weekday {
        date -= Duration::days(1);
    }
    Ok(key_from_date(date))
}

pub(super) fn easter_sunday(year: i32) -> BuiltinResult<NaiveDate> {
    let a = year.rem_euclid(19);
    let b = year.div_euclid(100);
    let c = year.rem_euclid(100);
    let d = b.div_euclid(4);
    let e = b.rem_euclid(4);
    let f = (b + 8).div_euclid(25);
    let g = (b - f + 1).div_euclid(3);
    let h = (19 * a + b - d - g + 15).rem_euclid(30);
    let i = c.div_euclid(4);
    let k = c.rem_euclid(4);
    let l = (32 + 2 * e + 2 * i - h - k).rem_euclid(7);
    let m = (a + 11 * h + 22 * l).div_euclid(451);
    let month = (h + l - 7 * m + 114).div_euclid(31) as u32;
    let day = ((h + l - 7 * m + 114).rem_euclid(31) + 1) as u32;
    NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| datetime_error("holidays: invalid computed Easter date"))
}

pub(super) fn market_holiday_keys_for_year(year: i32) -> BuiltinResult<Vec<i64>> {
    let mut keys = vec![
        observed_fixed_holiday(year, 1, 1)?,
        nth_weekday(year, 1, Weekday::Mon, 3)?,
        nth_weekday(year, 2, Weekday::Mon, 3)?,
        key_from_date(easter_sunday(year)? - Duration::days(2)),
        last_weekday(year, 5, Weekday::Mon)?,
        observed_fixed_holiday(year, 6, 19)?,
        observed_fixed_holiday(year, 7, 4)?,
        nth_weekday(year, 9, Weekday::Mon, 1)?,
        nth_weekday(year, 11, Weekday::Thu, 4)?,
        observed_fixed_holiday(year, 12, 25)?,
    ];
    keys.sort_unstable();
    keys.dedup();
    Ok(keys)
}

pub(super) fn holiday_keys_between(start_key: i64, end_key: i64) -> BuiltinResult<Vec<i64>> {
    let start_year = date_from_key(start_key.min(end_key))?
        .year()
        .checked_sub(1)
        .ok_or_else(|| datetime_error("holidays: date range is outside supported range"))?;
    let end_year = date_from_key(start_key.max(end_key))?
        .year()
        .checked_add(1)
        .ok_or_else(|| datetime_error("holidays: date range is outside supported range"))?;
    if end_year - start_year > MAX_HOLIDAY_YEAR_SPAN {
        return Err(datetime_error(format!(
            "holidays: date range spans more than {MAX_HOLIDAY_YEAR_SPAN} years"
        )));
    }
    let mut keys = Vec::new();
    for year in start_year..=end_year {
        keys.extend(market_holiday_keys_for_year(year)?);
    }
    keys.sort_unstable();
    keys.dedup();
    Ok(keys
        .into_iter()
        .filter(|key| *key >= start_key.min(end_key) && *key <= start_key.max(end_key))
        .collect())
}

pub(super) fn holiday_set_for_range(start_key: i64, end_key: i64) -> BuiltinResult<HashSet<i64>> {
    Ok(holiday_keys_between(start_key, end_key)?
        .into_iter()
        .collect())
}

pub(super) fn holiday_set_from_optional_or_default(
    value: Option<Value>,
    context: &str,
    start_key: i64,
    end_key: i64,
) -> BuiltinResult<HashSet<i64>> {
    if let Some(value) = value {
        let serials = numeric_or_datetime_serial_tensor(value, context)?;
        let values = tensor::tensor_values_f64_cow(&serials);
        return values
            .iter()
            .map(|serial| serial_date_key(*serial))
            .collect::<BuiltinResult<HashSet<_>>>();
    }
    holiday_set_for_range(start_key, end_key)
}

pub(super) fn date_key_range(tensors: &[&Tensor]) -> BuiltinResult<(i64, i64)> {
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    let mut found = false;
    for tensor in tensors {
        let values = tensor::tensor_values_f64_cow(tensor);
        for serial in values.iter() {
            let key = serial_date_key(*serial)?;
            min_key = min_key.min(key);
            max_key = max_key.max(key);
            found = true;
        }
    }
    if found {
        Ok((min_key, max_key))
    } else {
        Ok((0, 0))
    }
}

pub(super) fn is_business_day_key(key: i64, holidays: &HashSet<i64>) -> BuiltinResult<bool> {
    let date = date_from_key(key)?;
    Ok(!matches!(date.weekday(), Weekday::Sat | Weekday::Sun) && !holidays.contains(&key))
}

pub(super) fn count_weekdays_forward(start_key: i64, end_key: i64) -> BuiltinResult<i64> {
    let total_days = end_key
        .checked_sub(start_key)
        .and_then(|delta| delta.checked_add(1))
        .ok_or_else(|| datetime_error("business-day date range is outside supported range"))?;
    let full_weeks = total_days / 7;
    let mut count = full_weeks * 5;
    let remainder = total_days % 7;
    for offset in 0..remainder {
        let key = start_key
            .checked_add(offset)
            .ok_or_else(|| datetime_error("business-day date range is outside supported range"))?;
        if !matches!(date_from_key(key)?.weekday(), Weekday::Sat | Weekday::Sun) {
            count += 1;
        }
    }
    Ok(count)
}

pub(super) fn count_business_days(
    start_key: i64,
    end_key: i64,
    holidays: &HashSet<i64>,
) -> BuiltinResult<i64> {
    if start_key > end_key {
        return Ok(-count_business_days(end_key, start_key, holidays)?);
    }
    let mut count = count_weekdays_forward(start_key, end_key)?;
    for holiday in holidays {
        if *holiday >= start_key
            && *holiday <= end_key
            && !matches!(
                date_from_key(*holiday)?.weekday(),
                Weekday::Sat | Weekday::Sun
            )
        {
            count -= 1;
        }
    }
    Ok(count)
}

pub(super) fn first_business_day_key(
    year: i32,
    month: u32,
    holidays: &HashSet<i64>,
) -> BuiltinResult<i64> {
    let mut date = NaiveDate::from_ymd_opt(year, month, 1)
        .ok_or_else(|| datetime_error("fbusdate: invalid year/month"))?;
    loop {
        let key = key_from_date(date);
        if is_business_day_key(key, holidays)? {
            return Ok(key);
        }
        date += Duration::days(1);
    }
}

pub(super) fn last_business_day_key(
    year: i32,
    month: u32,
    holidays: &HashSet<i64>,
) -> BuiltinResult<i64> {
    let mut date = NaiveDate::from_ymd_opt(year, month, days_in_month(year, month)?)
        .ok_or_else(|| datetime_error("lbusdate: invalid year/month"))?;
    loop {
        let key = key_from_date(date);
        if is_business_day_key(key, holidays)? {
            return Ok(key);
        }
        date -= Duration::days(1);
    }
}
