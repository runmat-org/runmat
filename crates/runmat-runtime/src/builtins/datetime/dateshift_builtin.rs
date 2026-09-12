use super::capabilities::DATETIME_GPU_INPUT_EXTENSION;
use super::*;

#[runmat_macros::runtime_builtin(
    name = "dateshift",
    descriptor(crate::builtins::datetime::DATESHIFT_DESCRIPTOR),
    extensions(crate::builtins::datetime::DATESHIFT_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::DATESHIFT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Shift datetime values to calendar or clock boundaries.",
    keywords = "dateshift,datetime,start,end,dayofweek,weekday,weekend,rule",
    related = "datetime,year,month,day"
)]
pub(super) async fn dateshift_builtin(
    value: Value,
    boundary: Value,
    unit: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if std::iter::once(&value)
        .chain(std::iter::once(&boundary))
        .chain(std::iter::once(&unit))
        .chain(rest.iter())
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            "dateshift",
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("dateshift: {}", err.message())))?;
    let boundary = gather_if_needed_async(&boundary)
        .await
        .map_err(|err| datetime_error(format!("dateshift: {}", err.message())))?;
    let unit = gather_if_needed_async(&unit)
        .await
        .map_err(|err| datetime_error(format!("dateshift: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    let serials = serials_from_datetime_value(&value)?;
    let format = datetime_format_from_value(&value);
    let boundary = DateShiftBoundary::parse(&boundary)?;

    let serial_values = tensor::tensor_values_f64_cow(&serials);
    if rest.len() > 1 {
        return Err(datetime_error(
            "dateshift: expected at most one rule argument",
        ));
    }
    let default_rule = if boundary == DateShiftBoundary::DayOfWeek {
        DateShiftRule::Next
    } else {
        DateShiftRule::Current
    };
    let (rules, rule_shape) = if rest.is_empty() {
        (vec![default_rule], vec![1, 1])
    } else {
        parse_rules(rest.first())?
    };
    let mut targets = vec![DayTarget::Exact(Weekday::Sun)];
    let mut target_shape = vec![1, 1];
    if boundary == DateShiftBoundary::DayOfWeek {
        if matches!(
            unit,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            let text = scalar_text(&unit, "dateshift weekday")?;
            targets[0] = match text.trim().to_ascii_lowercase().as_str() {
                "weekend" => DayTarget::Weekend,
                "weekday" => DayTarget::Weekday,
                _ => DayTarget::Exact(parse_weekday(&unit)?),
            };
        } else {
            let (indices, shape) = exact_integer_values(&unit, "weekday")?;
            targets = indices
                .into_iter()
                .map(weekday_from_matlab_index)
                .collect::<BuiltinResult<Vec<_>>>()?
                .into_iter()
                .map(DayTarget::Exact)
                .collect();
            target_shape = shape;
        }
    }
    let lengths = [serial_values.len(), targets.len(), rules.len()];
    let output_len = *lengths.iter().max().unwrap_or(&1);
    if lengths.iter().any(|len| *len != 1 && *len != output_len) {
        return Err(datetime_error(
            "dateshift: datetime, weekday, and rule arrays must have matching sizes or be scalar",
        ));
    }
    let serial_shape = tensor::default_shape_for(&serials.shape, serial_values.len());
    let mut non_scalar_shapes = Vec::new();
    if serial_values.len() > 1 {
        non_scalar_shapes.push(&serial_shape);
    }
    if targets.len() > 1 {
        non_scalar_shapes.push(&target_shape);
    }
    if rules.len() > 1 {
        non_scalar_shapes.push(&rule_shape);
    }
    if non_scalar_shapes.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(datetime_error(
            "dateshift: non-scalar datetime, weekday, and rule arrays must have matching sizes",
        ));
    }
    let output_shape = if serial_values.len() > 1 {
        serial_shape
    } else if targets.len() > 1 {
        target_shape
    } else if rules.len() > 1 {
        rule_shape
    } else {
        vec![1, 1]
    };
    let parsed_unit = if boundary == DateShiftBoundary::DayOfWeek {
        None
    } else {
        Some(DateShiftUnit::parse(&unit)?)
    };
    let mut out = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let serial = serial_values[if serial_values.len() == 1 { 0 } else { index }];
        if !serial.is_finite() {
            out.push(serial);
            continue;
        }
        let value = naive_from_datenum(serial)?;
        let rule = rules[if rules.len() == 1 { 0 } else { index }];
        let shifted = if boundary == DateShiftBoundary::DayOfWeek {
            let target = targets[if targets.len() == 1 { 0 } else { index }];
            shift_day_target(value, target, rule)?
        } else {
            apply_boundary_rule(value, boundary, parsed_unit.unwrap(), rule)?
        };
        out.push(datenum_from_naive(shifted));
    }
    datetime_object_from_serials(out, output_shape, format)
}
