//! Selected managed `dateshift`: borrowed controls and exact retained datetime output.
use super::super::{
    conversion::{BorrowedDatetimeSerial, CalendarWork, PreparedControl},
    dateshift_model::{
        apply_boundary_rule, parse_boundary_text, parse_dateshift_day_target, parse_rule_text,
        parse_unit_text, shift_day_target, DateShiftRule,
    },
    datetime_error, datetime_gather_error, naive_from_datenum,
};
use crate::call::BuiltinArguments;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationCheckpoint, HostAllocationError,
    HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::RootedValueLease;
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_builtins::DateShiftBoundary;
use runmat_value::{NumericDType, Value};

#[path = "dateshift/input.rs"]
mod input;
#[path = "dateshift/output.rs"]
mod output;
#[path = "dateshift/plan.rs"]
mod plan;
use input::IntegerView;
use plan::{Plan, Rule, Target};

fn allocation_error(runtime: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    let mut error = host_allocation_runtime_error(
        Some(runtime),
        "RunMat:dateshift:CalendarWorkspace",
        "dateshift: managed calendar allocation failed",
        source,
    );
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

fn never(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

async fn gathered(
    runtime: &RuntimeContext,
    work: &mut CalendarWork<'_>,
    source: &Value,
) -> BuiltinResult<Value> {
    work.check()?;
    let result = runtime
        .scope(crate::gather_if_needed_async(source))
        .await
        .map_err(|source| datetime_gather_error(Some(runtime), source))?;
    work.check()?;
    Ok(result)
}

pub(in crate::builtins::datetime) async fn managed(
    runtime: &RuntimeContext,
    original: HostAllocationCheckpoint,
    value: Value,
    boundary: Value,
    unit: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut work = CalendarWork::with_owner(Some(runtime), "dateshift", Some(original))?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::DATESHIFT_DESCRIPTOR,
        &runmat_builtins::DATESHIFT_ERROR_TOO_MANY_OUTPUTS,
        "dateshift",
    )
    .map_err(never)?;
    if rest.len() > 1 {
        return Err(never(super::super::datetime_argument_error(
            "dateshift: expected at most one rule argument",
        )));
    }
    if std::iter::once(&value)
        .chain(std::iter::once(&boundary))
        .chain(std::iter::once(&unit))
        .chain(rest.iter())
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &super::super::capabilities::DATETIME_GPU_INPUT_EXTENSION,
            "dateshift",
        )
        .map_err(never)?;
    }
    let _roots = RootedValueLease::for_values_iterator_in_context(
        [&value, &boundary, &unit].into_iter().chain(rest.iter()),
        runtime,
    )
    .map_err(never)?;
    let gathered_value = gathered(runtime, &mut work, &value).await?;
    let gathered_boundary = gathered(runtime, &mut work, &boundary).await?;
    let gathered_unit = gathered(runtime, &mut work, &unit).await?;
    let rule_value = if let Some(rule) = rest.first() {
        Some(gathered(runtime, &mut work, rule).await?)
    } else {
        None
    };
    let serials = BorrowedDatetimeSerial::borrow(&gathered_value).map_err(never)?;
    let boundary_control =
        PreparedControl::new(runtime, &mut work, Some(&gathered_boundary)).await?;
    let selected_boundary =
        parse_boundary_text(boundary_control.text().expect("validated text control"))
            .map_err(never)?;
    let rule_control = if rule_value.as_ref().is_some_and(|value| {
        matches!(
            value,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        )
    }) {
        Some(PreparedControl::new(runtime, &mut work, rule_value.as_ref()).await?)
    } else {
        None
    };
    let default_rule = if selected_boundary == DateShiftBoundary::DayOfWeek {
        DateShiftRule::Next
    } else {
        DateShiftRule::Current
    };
    let rule = if let Some(control) = &rule_control {
        Rule::Scalar(
            parse_rule_text(control.text().expect("validated text control")).map_err(never)?,
        )
    } else if let Some(value) = &rule_value {
        Rule::Numeric(IntegerView::borrow(value, "rule").map_err(never)?)
    } else {
        Rule::Scalar(default_rule)
    };
    // The ordinary parser validates the rule before the weekday target,
    // including supplied controls when the final datetime output is empty.
    for index in 0..rule.len() {
        rule.at(index).map_err(never)?;
        work.charge(1).await?;
    }
    let unit_control = if matches!(
        &gathered_unit,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        Some(PreparedControl::new(runtime, &mut work, Some(&gathered_unit)).await?)
    } else {
        None
    };
    let target = if selected_boundary == DateShiftBoundary::DayOfWeek {
        if let Some(unit_control) = &unit_control {
            Target::Text(
                parse_dateshift_day_target(unit_control.text().expect("validated text control"))
                    .map_err(never)?,
            )
        } else {
            Target::Numeric(IntegerView::borrow(&gathered_unit, "weekday").map_err(never)?)
        }
    } else {
        Target::Default
    };
    if selected_boundary == DateShiftBoundary::DayOfWeek {
        for index in 0..target.len() {
            target.at(index).map_err(never)?;
            work.charge(1).await?;
        }
    }
    let mut plan = Plan::new(serials, target, rule, None).map_err(never)?;
    if selected_boundary != DateShiftBoundary::DayOfWeek {
        plan.unit = Some(
            parse_unit_text(
                unit_control
                    .as_ref()
                    .and_then(PreparedControl::text)
                    .ok_or_else(|| {
                        never(datetime_error(
                            "datetime: dateshift unit must be a string scalar or character vector",
                        ))
                    })?,
            )
            .map_err(never)?,
        );
    }
    work.flush().await?;
    work.check()?;
    let retained = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, plan.shape)
        .map_err(|source| allocation_error(runtime, source))?;
    let quote =
        HostAllocationPlan::new(0, retained).map_err(|source| allocation_error(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), quote)
        .map_err(|source| allocation_error(runtime, source))?
        .ok_or_else(|| allocation_error(runtime, HostAllocationError::StaleAuthority))?;
    work.check()?;
    let mut output = receipt
        .exact_output_tensor(runtime, NumericDType::F64, plan.shape)
        .map_err(|source| allocation_error(runtime, source))?;
    for index in 0..plan.len {
        let serial = plan.serial_at(index);
        let shifted = if !serial.is_finite() {
            serial
        } else {
            let instant = naive_from_datenum(serial).map_err(never)?;
            let rule = plan.rule_at(index).map_err(never)?;
            let instant = if selected_boundary == DateShiftBoundary::DayOfWeek {
                shift_day_target(instant, plan.target_at(index).map_err(never)?, rule)
            } else {
                apply_boundary_rule(
                    instant,
                    selected_boundary,
                    plan.unit.expect("non-weekday boundary has selected unit"),
                    rule,
                )
            }
            .map_err(never)?;
            super::super::datenum_from_naive(instant)
        };
        output
            .as_f64_slice_mut()
            .expect("dateshift output is double")[index] = shifted;
        work.charge(1).await?;
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation_error(runtime, source))?;
    }
    work.flush().await?;
    work.check()?;
    let tensor = output
        .into_tensor()
        .map_err(|source| allocation_error(runtime, source.into()))?;
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation_error(runtime, source))?;
    let result = output::publish(runtime, &mut work, &gathered_value, tensor)
        .await
        .map_err(never)?;
    work.check()?;
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation_error(runtime, source))?;
    Ok(result)
}

#[cfg(test)]
#[path = "dateshift/tests.rs"]
mod tests;
