use super::capabilities::MAX_DATESHIFT_DAY_OCCURRENCE;
use super::*;
pub(super) use runmat_builtins::{DateShiftBoundary, DateShiftUnit};
use runmat_builtins::{DateShiftDayTarget, DateShiftRuleKeyword};

pub(super) fn parse_dateshift_boundary(value: &Value) -> BuiltinResult<DateShiftBoundary> {
    let text = scalar_text(value, "dateshift boundary")?;
    parse_boundary_text(&text)
}

pub(super) fn parse_boundary_text(text: &str) -> BuiltinResult<DateShiftBoundary> {
    DateShiftBoundary::parse_text(text)
        .ok_or_else(|| datetime_error(format!("dateshift: unsupported boundary '{text}'")))
}
pub(super) fn parse_dateshift_unit(value: &Value) -> BuiltinResult<DateShiftUnit> {
    let text = scalar_text(value, "dateshift unit")?;
    parse_unit_text(&text)
}

pub(super) fn parse_unit_text(text: &str) -> BuiltinResult<DateShiftUnit> {
    DateShiftUnit::parse_text(text)
        .ok_or_else(|| datetime_error(format!("dateshift: unsupported unit '{text}'")))
}
pub(super) fn parse_dateshift_day_target(text: &str) -> BuiltinResult<DayTarget> {
    let target = DateShiftDayTarget::parse_text(text)
        .ok_or_else(|| datetime_error(format!("dateshift: unsupported weekday '{text}'")))?;
    Ok(match target {
        DateShiftDayTarget::Sunday => DayTarget::Exact(Weekday::Sun),
        DateShiftDayTarget::Monday => DayTarget::Exact(Weekday::Mon),
        DateShiftDayTarget::Tuesday => DayTarget::Exact(Weekday::Tue),
        DateShiftDayTarget::Wednesday => DayTarget::Exact(Weekday::Wed),
        DateShiftDayTarget::Thursday => DayTarget::Exact(Weekday::Thu),
        DateShiftDayTarget::Friday => DayTarget::Exact(Weekday::Fri),
        DateShiftDayTarget::Saturday => DayTarget::Exact(Weekday::Sat),
        DateShiftDayTarget::Weekday => DayTarget::Weekday,
        DateShiftDayTarget::Weekend => DayTarget::Weekend,
    })
}

pub(super) fn weekday_from_matlab_index(index: i64) -> BuiltinResult<Weekday> {
    match index {
        1 => Ok(Weekday::Sun),
        2 => Ok(Weekday::Mon),
        3 => Ok(Weekday::Tue),
        4 => Ok(Weekday::Wed),
        5 => Ok(Weekday::Thu),
        6 => Ok(Weekday::Fri),
        7 => Ok(Weekday::Sat),
        _ => Err(datetime_error(
            "dateshift: numeric weekdays must be in the range 1..7",
        )),
    }
}

pub(super) fn midnight(date: NaiveDate) -> NaiveDateTime {
    date.and_hms_opt(0, 0, 0).unwrap()
}

pub(super) fn start_of_week(
    value: NaiveDateTime,
    week_start: Weekday,
) -> BuiltinResult<NaiveDateTime> {
    let current = value.weekday().num_days_from_monday() as i64;
    let start = week_start.num_days_from_monday() as i64;
    let delta = (current - start).rem_euclid(7);
    value
        .date()
        .checked_sub_signed(Duration::days(delta))
        .map(midnight)
        .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))
}

pub(super) fn start_of_unit(
    value: NaiveDateTime,
    unit: DateShiftUnit,
    week_start: Weekday,
) -> BuiltinResult<NaiveDateTime> {
    let start = match unit {
        DateShiftUnit::Year => midnight(NaiveDate::from_ymd_opt(value.year(), 1, 1).unwrap()),
        DateShiftUnit::Quarter => {
            let month = ((value.month() - 1) / 3) * 3 + 1;
            midnight(NaiveDate::from_ymd_opt(value.year(), month, 1).unwrap())
        }
        DateShiftUnit::Month => {
            midnight(NaiveDate::from_ymd_opt(value.year(), value.month(), 1).unwrap())
        }
        DateShiftUnit::Week => return start_of_week(value, week_start),
        DateShiftUnit::Day => midnight(value.date()),
        DateShiftUnit::Hour => value
            .date()
            .and_hms_nano_opt(value.hour(), 0, 0, 0)
            .unwrap(),
        DateShiftUnit::Minute => value
            .date()
            .and_hms_nano_opt(value.hour(), value.minute(), 0, 0)
            .unwrap(),
        DateShiftUnit::Second => value
            .date()
            .and_hms_nano_opt(value.hour(), value.minute(), value.second(), 0)
            .unwrap(),
    };
    Ok(start)
}

pub(super) fn next_unit_start(
    start: NaiveDateTime,
    unit: DateShiftUnit,
) -> BuiltinResult<NaiveDateTime> {
    let out_of_range = || datetime_error("dateshift: result is outside the supported range");
    match unit {
        DateShiftUnit::Year => {
            let year = start.year().checked_add(1).ok_or_else(out_of_range)?;
            return NaiveDate::from_ymd_opt(year, 1, 1)
                .map(midnight)
                .ok_or_else(out_of_range);
        }
        DateShiftUnit::Quarter | DateShiftUnit::Month => {
            let delta = if matches!(unit, DateShiftUnit::Quarter) {
                3
            } else {
                1
            };
            let month_index = i64::from(start.year())
                .checked_mul(12)
                .and_then(|base| base.checked_add(i64::from(start.month0())))
                .and_then(|base| base.checked_add(delta))
                .ok_or_else(out_of_range)?;
            let year = i32::try_from(month_index.div_euclid(12)).map_err(|_| out_of_range())?;
            let month = month_index.rem_euclid(12) as u32 + 1;
            return NaiveDate::from_ymd_opt(year, month, 1)
                .map(midnight)
                .ok_or_else(out_of_range);
        }
        DateShiftUnit::Week => start.checked_add_signed(Duration::days(7)),
        DateShiftUnit::Day => start.checked_add_signed(Duration::days(1)),
        DateShiftUnit::Hour => start.checked_add_signed(Duration::hours(1)),
        DateShiftUnit::Minute => start.checked_add_signed(Duration::minutes(1)),
        DateShiftUnit::Second => start.checked_add_signed(Duration::seconds(1)),
    }
    .ok_or_else(out_of_range)
}

#[derive(Clone, Copy)]
pub(super) enum DateShiftRule {
    Current,
    Next,
    Previous,
    Nearest,
    Occurrence(i64),
}

pub(super) use runmat_builtins::dateshift_exact_f64_to_i64 as exact_f64_to_i64;

pub(super) fn exact_integer_values(
    value: &Value,
    context: &str,
) -> BuiltinResult<(Vec<i64>, Vec<usize>)> {
    match value {
        Value::Int(value) => value
            .try_to_i64()
            .map(|value| (vec![value], vec![1, 1]))
            .ok_or_else(|| datetime_error(format!("dateshift: {context} is outside int64 range"))),
        Value::Num(value) => exact_f64_to_i64(*value)
            .map(|value| (vec![value], vec![1, 1]))
            .ok_or_else(|| {
                datetime_error(format!(
                    "dateshift: {context} must be a representable integer"
                ))
            }),
        Value::Tensor(array) => {
            let shape = array.shape().to_vec();
            if let Some(storage) = array.integer_storage() {
                let mut out = Vec::with_capacity(storage.len());
                for value in storage.exact_values() {
                    out.push(value.try_to_i64().ok_or_else(|| {
                        datetime_error(format!("dateshift: {context} is outside int64 range"))
                    })?);
                }
                Ok((out, shape))
            } else {
                let mut out = Vec::with_capacity(tensor::tensor_element_len(array));
                for value in tensor::tensor_values_f64_cow(array).iter().copied() {
                    out.push(exact_f64_to_i64(value).ok_or_else(|| {
                        datetime_error(format!(
                            "dateshift: {context} values must be representable integers"
                        ))
                    })?);
                }
                Ok((out, shape))
            }
        }
        Value::Bool(_) | Value::LogicalArray(_) => Err(datetime_error(format!(
            "dateshift: {context} does not accept logical values"
        ))),
        _ => Err(datetime_error(format!(
            "dateshift: {context} must be an integer scalar or array"
        ))),
    }
}

pub(super) fn parse_rules(
    value: Option<&Value>,
) -> BuiltinResult<(Vec<DateShiftRule>, Vec<usize>)> {
    let Some(value) = value else {
        return Ok((vec![DateShiftRule::Current], vec![1, 1]));
    };
    if matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        let text = scalar_text(value, "dateshift rule")?;
        let rule = parse_rule_text(&text)?;
        return Ok((vec![rule], vec![1, 1]));
    }
    let (values, shape) = exact_integer_values(value, "rule")?;
    Ok((
        values.into_iter().map(DateShiftRule::Occurrence).collect(),
        shape,
    ))
}

pub(super) fn parse_rule_text(text: &str) -> BuiltinResult<DateShiftRule> {
    let keyword = DateShiftRuleKeyword::parse_text(text)
        .ok_or_else(|| datetime_error(format!("dateshift: unsupported rule '{text}'")))?;
    Ok(match keyword {
        DateShiftRuleKeyword::Current => DateShiftRule::Current,
        DateShiftRuleKeyword::Next => DateShiftRule::Next,
        DateShiftRuleKeyword::Previous => DateShiftRule::Previous,
        DateShiftRuleKeyword::Nearest => DateShiftRule::Nearest,
    })
}

pub(super) fn unit_step(
    value: NaiveDateTime,
    unit: DateShiftUnit,
    steps: i64,
) -> BuiltinResult<NaiveDateTime> {
    match unit {
        DateShiftUnit::Year | DateShiftUnit::Quarter | DateShiftUnit::Month => {
            let factor = match unit {
                DateShiftUnit::Year => 12,
                DateShiftUnit::Quarter => 3,
                _ => 1,
            };
            let delta = steps
                .checked_mul(factor)
                .ok_or_else(|| datetime_error("dateshift: rule is outside the supported range"))?;
            let month_index = i64::from(value.year())
                .checked_mul(12)
                .and_then(|base| base.checked_add(i64::from(value.month0())))
                .and_then(|base| base.checked_add(delta))
                .ok_or_else(|| {
                    datetime_error("dateshift: result is outside the supported range")
                })?;
            let year = i32::try_from(month_index.div_euclid(12))
                .map_err(|_| datetime_error("dateshift: result is outside the supported range"))?;
            let month = month_index.rem_euclid(12) as u32 + 1;
            return Ok(midnight(
                NaiveDate::from_ymd_opt(year, month, 1).ok_or_else(|| {
                    datetime_error("dateshift: result is outside the supported range")
                })?,
            ));
        }
        DateShiftUnit::Week
        | DateShiftUnit::Day
        | DateShiftUnit::Hour
        | DateShiftUnit::Minute
        | DateShiftUnit::Second => {
            let duration = match unit {
                DateShiftUnit::Week => Duration::try_weeks(steps),
                DateShiftUnit::Day => Duration::try_days(steps),
                DateShiftUnit::Hour => Duration::try_hours(steps),
                DateShiftUnit::Minute => Duration::try_minutes(steps),
                DateShiftUnit::Second => Duration::try_seconds(steps),
                _ => unreachable!("calendar units returned above"),
            }
            .ok_or_else(|| datetime_error("dateshift: rule is outside the supported range"))?;
            value.checked_add_signed(duration)
        }
    }
    .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))
}

pub(super) fn unit_end(start: NaiveDateTime, unit: DateShiftUnit) -> BuiltinResult<NaiveDateTime> {
    let next = next_unit_start(start, unit)?;
    match unit {
        DateShiftUnit::Year
        | DateShiftUnit::Quarter
        | DateShiftUnit::Month
        | DateShiftUnit::Week => next.checked_sub_signed(Duration::days(1)),
        DateShiftUnit::Day
        | DateShiftUnit::Hour
        | DateShiftUnit::Minute
        | DateShiftUnit::Second => Some(next),
    }
    .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))
}

pub(super) fn apply_boundary_rule(
    value: NaiveDateTime,
    boundary: DateShiftBoundary,
    unit: DateShiftUnit,
    rule: DateShiftRule,
) -> BuiltinResult<NaiveDateTime> {
    let start = start_of_unit(value, unit, Weekday::Sun)?;
    let current = if boundary == DateShiftBoundary::Start {
        start
    } else {
        unit_end(start, unit)?
    };
    let shifted = |steps| {
        let shifted_start = unit_step(start, unit, steps)?;
        Ok(if boundary == DateShiftBoundary::Start {
            shifted_start
        } else {
            unit_end(shifted_start, unit)?
        })
    };
    match rule {
        DateShiftRule::Current => Ok(current),
        DateShiftRule::Next => shifted(1),
        DateShiftRule::Previous => shifted(-1),
        DateShiftRule::Nearest => {
            let previous = if value >= current {
                current
            } else {
                shifted(-1)?
            };
            let next = if value <= current {
                current
            } else {
                shifted(1)?
            };
            if value - previous <= next - value {
                Ok(previous)
            } else {
                Ok(next)
            }
        }
        DateShiftRule::Occurrence(0) => Ok(current),
        DateShiftRule::Occurrence(n) => shifted(n),
    }
}

#[derive(Clone, Copy)]
pub(super) enum DayTarget {
    Exact(Weekday),
    Weekend,
    Weekday,
}

pub(super) fn target_matches(target: DayTarget, weekday: Weekday) -> bool {
    match target {
        DayTarget::Exact(expected) => weekday == expected,
        DayTarget::Weekend => matches!(weekday, Weekday::Sat | Weekday::Sun),
        DayTarget::Weekday => !matches!(weekday, Weekday::Sat | Weekday::Sun),
    }
}

pub(super) fn current_week_target(
    origin: NaiveDateTime,
    target: DayTarget,
) -> BuiltinResult<NaiveDateTime> {
    let sunday = start_of_week(origin, Weekday::Sun)?;
    let add_days = |days| {
        sunday
            .checked_add_signed(Duration::days(days))
            .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))
    };
    match target {
        DayTarget::Exact(weekday) => add_days(i64::from(weekday.num_days_from_sunday())),
        DayTarget::Weekend => {
            if matches!(origin.weekday(), Weekday::Sat | Weekday::Sun) {
                Ok(origin)
            } else {
                add_days(6)
            }
        }
        DayTarget::Weekday => match origin.weekday() {
            Weekday::Sun => add_days(1),
            Weekday::Sat => add_days(5),
            _ => Ok(origin),
        },
    }
}

pub(super) fn shift_day_target(
    value: NaiveDateTime,
    target: DayTarget,
    rule: DateShiftRule,
) -> BuiltinResult<NaiveDateTime> {
    let origin = midnight(value.date());
    let seek = |direction: i64, occurrence: u64| -> BuiltinResult<NaiveDateTime> {
        if occurrence == 0 || occurrence > MAX_DATESHIFT_DAY_OCCURRENCE {
            return Err(datetime_error(
                "dateshift: day occurrence rule is outside the supported range",
            ));
        }

        let mut date = origin;
        for _ in 0..7 {
            if target_matches(target, date.weekday()) {
                break;
            }
            date = date
                .checked_add_signed(Duration::days(direction))
                .ok_or_else(|| {
                    datetime_error("dateshift: result is outside the supported range")
                })?;
        }
        if !target_matches(target, date.weekday()) {
            return Err(datetime_error(
                "dateshift: result is outside the supported range",
            ));
        }

        let matches_per_week = match target {
            DayTarget::Exact(_) => 1,
            DayTarget::Weekend => 2,
            DayTarget::Weekday => 5,
        };
        let remaining = occurrence - 1;
        let whole_weeks = remaining / matches_per_week;
        let residual = remaining % matches_per_week;
        let whole_week_days = i64::try_from(whole_weeks.checked_mul(7).ok_or_else(|| {
            datetime_error("dateshift: day occurrence rule is outside the supported range")
        })?)
        .map_err(|_| {
            datetime_error("dateshift: day occurrence rule is outside the supported range")
        })?;
        let signed_days = whole_week_days
            .checked_mul(direction)
            .ok_or_else(|| datetime_error("dateshift: rule is outside the supported range"))?;
        let delta = Duration::try_days(signed_days)
            .ok_or_else(|| datetime_error("dateshift: rule is outside the supported range"))?;
        date = date
            .checked_add_signed(delta)
            .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))?;

        let mut residual_found = 0;
        while residual_found < residual {
            date = date
                .checked_add_signed(Duration::days(direction))
                .ok_or_else(|| {
                    datetime_error("dateshift: result is outside the supported range")
                })?;
            if target_matches(target, date.weekday()) {
                residual_found += 1;
            }
        }
        Ok(date)
    };
    match rule {
        DateShiftRule::Current => current_week_target(origin, target),
        DateShiftRule::Next => seek(1, 1),
        DateShiftRule::Previous => seek(-1, 1),
        DateShiftRule::Nearest => {
            let previous = seek(-1, 1)?;
            let next = seek(1, 1)?;
            if origin - previous <= next - origin {
                Ok(previous)
            } else {
                Ok(next)
            }
        }
        DateShiftRule::Occurrence(0) => current_week_target(origin, target),
        DateShiftRule::Occurrence(n) if n > 0 => seek(1, n as u64),
        DateShiftRule::Occurrence(n) => seek(-1, n.unsigned_abs()),
    }
}
