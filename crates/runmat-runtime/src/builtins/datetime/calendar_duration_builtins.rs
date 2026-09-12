use super::capabilities::SECONDS_PER_DAY;
use super::*;

#[runmat_macros::runtime_builtin(
    name = "calendarDuration",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar duration values from calendar components.",
    keywords = "calendarDuration,caldays,calmonths,calyears,datetime"
)]
pub(super) async fn calendar_duration_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let args = gather_args(&args).await?;
    if args.is_empty() {
        return calendar_duration_object_from_components(vec![0.0], vec![0.0], vec![1, 1]);
    }
    if args.len() == 1 && is_calendar_duration_object(&args[0]) {
        return Ok(args[0].clone());
    }

    let labels = ["years", "months", "days", "hours", "minutes", "seconds"];
    let positional = match args.len() {
        1 => {
            let days = component_tensor(args[0].clone(), "calendarDuration")?;
            let shape = tensor::default_shape_for(&days.shape, tensor::tensor_element_len(&days));
            let day_values = tensor::tensor_into_values_f64(days);
            return calendar_duration_object_from_components(
                vec![0.0; day_values.len()],
                day_values,
                shape,
            );
        }
        3..=6 => args,
        _ => {
            return Err(datetime_error(
                "calendarDuration: expected no input, days, or Y/M/D[/H/M/S] components",
            ))
        }
    };

    let mut arrays = Vec::with_capacity(6);
    for (idx, arg) in positional.into_iter().enumerate() {
        arrays.push(component_tensor(arg, labels[idx])?);
    }
    while arrays.len() < 6 {
        arrays.push(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    }
    let (broadcasted, shape) = broadcast_component_data(&arrays, &labels)?;
    let len = broadcasted[0].len();
    let mut months = Vec::with_capacity(len);
    let mut days = Vec::with_capacity(len);
    for idx in 0..len {
        let month_value = broadcasted[0][idx] * 12.0 + broadcasted[1][idx];
        let day_value = broadcasted[2][idx]
            + broadcasted[3][idx] / 24.0
            + broadcasted[4][idx] / (24.0 * 60.0)
            + broadcasted[5][idx] / SECONDS_PER_DAY;
        if !month_value.is_finite() || !day_value.is_finite() {
            return Err(datetime_error(
                "calendarDuration: resulting calendar duration is outside supported range",
            ));
        }
        months.push(month_value);
        days.push(day_value);
    }
    calendar_duration_object_from_components(months, days, shape)
}

#[runmat_macros::runtime_builtin(
    name = "caldays",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from days or convert calendar durations to day counts.",
    keywords = "caldays,calendarDuration,datetime"
)]
pub(super) async fn caldays_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("caldays: {}", err.message())))?;
    calendar_duration_unit_value(value, "caldays", 0.0, 1.0)
}

#[runmat_macros::runtime_builtin(
    name = "calweeks",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from weeks or convert calendar durations to week counts.",
    keywords = "calweeks,calendarDuration,datetime"
)]
pub(super) async fn calweeks_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("calweeks: {}", err.message())))?;
    calendar_duration_unit_value(value, "calweeks", 0.0, 7.0)
}

#[runmat_macros::runtime_builtin(
    name = "calmonths",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from months or convert calendar durations to month counts.",
    keywords = "calmonths,calendarDuration,datetime"
)]
pub(super) async fn calmonths_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("calmonths: {}", err.message())))?;
    calendar_duration_unit_value(value, "calmonths", 1.0, 0.0)
}

#[runmat_macros::runtime_builtin(
    name = "calquarters",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from quarters or convert calendar durations to quarter counts.",
    keywords = "calquarters,calendarDuration,datetime"
)]
pub(super) async fn calquarters_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("calquarters: {}", err.message())))?;
    calendar_duration_unit_value(value, "calquarters", 3.0, 0.0)
}

#[runmat_macros::runtime_builtin(
    name = "calyears",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from years or convert calendar durations to year counts.",
    keywords = "calyears,calendarDuration,datetime"
)]
pub(super) async fn calyears_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("calyears: {}", err.message())))?;
    calendar_duration_unit_value(value, "calyears", 12.0, 0.0)
}

#[runmat_macros::runtime_builtin(
    name = "iscalendarduration",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return true for calendarDuration values.",
    keywords = "iscalendarduration,calendarDuration,predicate"
)]
pub(super) fn iscalendarduration_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(is_calendar_duration_object(&value)))
}
