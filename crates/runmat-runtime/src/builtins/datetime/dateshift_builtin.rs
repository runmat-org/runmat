use super::capabilities::DATETIME_GPU_INPUT_EXTENSION;
use super::*;

#[runmat_macros::runtime_builtin(
    name = "dateshift",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn dateshift_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    boundary: Value,
    unit: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let owner =
        crate::context::host_allocation::HostAllocationReceipt::capture_owner(runtime.as_ref())
            .map_err(|source| {
                let mut error = crate::context::host_allocation::host_allocation_runtime_error(
                    runtime.as_ref(),
                    "RunMat:dateshift:CalendarWorkspace",
                    "dateshift: calendar allocation owner unavailable",
                    source,
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error
            })?;
    if let Some(owner) = owner {
        let runtime = runtime.as_ref().expect("managed owner has runtime context");
        return super::allocation::dateshift::managed(runtime, owner, value, boundary, unit, rest)
            .await;
    }
    let work = super::allocation::UnboundedCalendarWork::admit(runtime.as_ref(), "dateshift")?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::DATESHIFT_DESCRIPTOR,
        &runmat_builtins::DATESHIFT_ERROR_TOO_MANY_OUTPUTS,
        "dateshift",
    )?;
    if rest.len() > 1 {
        return Err(datetime_argument_error(
            "dateshift: expected at most one rule argument",
        ));
    }
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
    let mut budget = runtime
        .as_ref()
        .map(crate::context::RuntimeContext::cpu_cooperation);
    let value = crate::dispatcher::gather_if_needed_owned_async(value, budget.as_mut())
        .await
        .map_err(|err| crate::builtins::common::map_control_flow_with_builtin(err, "dateshift"))?;
    work.check()?;
    let boundary = crate::dispatcher::gather_if_needed_owned_async(boundary, budget.as_mut())
        .await
        .map_err(|err| crate::builtins::common::map_control_flow_with_builtin(err, "dateshift"))?;
    work.check()?;
    let unit = crate::dispatcher::gather_if_needed_owned_async(unit, budget.as_mut())
        .await
        .map_err(|err| crate::builtins::common::map_control_flow_with_builtin(err, "dateshift"))?;
    work.check()?;
    let rest = rest
        .try_map_preserving_roots_async(rest.len(), runtime.as_ref(), async |index| {
            work.check()?;
            let value = crate::dispatcher::gather_if_needed_owned_async(
                rest[index].clone(),
                budget.as_mut(),
            )
            .await
            .map_err(|err| {
                crate::builtins::common::map_control_flow_with_builtin(err, "dateshift")
            })?;
            work.check()?;
            Ok(value)
        })
        .await?;
    work.check()?;
    let serials = serials_from_datetime_value(&value)?;
    let format = datetime_format_from_value(&value)?;
    let boundary = parse_dateshift_boundary(&boundary)?;

    let serial_values = tensor::tensor_values_f64_cow(&serials);
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
            targets[0] = parse_dateshift_day_target(&text)?;
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
    // Empty is a real non-scalar cardinality, not a request for one element.
    let output_len = lengths.iter().copied().find(|len| *len != 1).unwrap_or(1);
    if lengths.iter().any(|len| *len != 1 && *len != output_len) {
        return Err(datetime_error(
            "dateshift: datetime, weekday, and rule arrays must have matching sizes or be scalar",
        ));
    }
    let serial_shape = serials.shape().to_vec();
    let mut non_scalar_shapes = Vec::new();
    if serial_values.len() != 1 {
        non_scalar_shapes.push(&serial_shape);
    }
    if targets.len() != 1 {
        non_scalar_shapes.push(&target_shape);
    }
    if rules.len() != 1 {
        non_scalar_shapes.push(&rule_shape);
    }
    if non_scalar_shapes.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(datetime_error(
            "dateshift: non-scalar datetime, weekday, and rule arrays must have matching sizes",
        ));
    }
    let output_shape = if serial_values.len() != 1 {
        serial_shape
    } else if targets.len() != 1 {
        target_shape
    } else if rules.len() != 1 {
        rule_shape
    } else {
        vec![1, 1]
    };
    let parsed_unit = if boundary == DateShiftBoundary::DayOfWeek {
        None
    } else {
        Some(parse_dateshift_unit(&unit)?)
    };
    let mut out = Vec::with_capacity(output_len);
    for index in 0..output_len {
        if let Some(budget) = budget.as_mut() {
            budget
                .checkpoint(1)
                .await
                .map_err(|source| datetime_execution_error(runtime.as_ref(), source))?;
        }
        work.check()?;
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
    work.check()?;
    work.finish(datetime_object_from_serials(
        runtime.as_ref(),
        out,
        output_shape,
        format,
    ))
}
