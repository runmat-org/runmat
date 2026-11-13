//! Selected-context residual datetime calls; existing calendar scratch remains fenced.
use super::super::*;
use runmat_builtins::CalendarResidualOperation as Op;
#[cfg(test)]
#[path = "remaining/tests.rs"]
mod tests;

pub(super) fn terminal(mut error: crate::RuntimeError, op: Op) -> crate::RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error.context.builtin = Some(op.name().into());
    if error.identifier.is_none() {
        error.identifier = runmat_builtins::CALENDAR_RESIDUAL_ERROR_INPUT
            .identifier
            .map(Into::into);
    }
    error
}
fn preflight(args: crate::call::FunctionArgumentView<'_>, op: Op) -> BuiltinResult<()> {
    crate::output_count::validate_fixed_outputs(
        op.entry().descriptor,
        &runmat_builtins::CALENDAR_RESIDUAL_ERROR_OUTPUTS,
        op.name(),
    )
    .map_err(|error| terminal(error, op))?;
    if op.input_count().is_some_and(|count| count != args.len()) {
        return Err(terminal(
            crate::runtime_descriptor_error_with_detail(
                op.name(),
                &runmat_builtins::CALENDAR_RESIDUAL_ERROR_ARGUMENT,
                "invalid input count",
            ),
            op,
        ));
    }
    Ok(())
}
pub(in crate::builtins::datetime) fn predicate(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    preflight(args, Op::IsDatetime)?;
    if runtime.is_some_and(crate::context::RuntimeContext::is_cancelled) {
        return Err(terminal(
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:Cancelled",
                "isdatetime: invocation cancelled",
                crate::context::host_allocation::HostAllocationError::Cancelled,
            ),
            Op::IsDatetime,
        ));
    }
    Ok(Value::Bool(is_datetime_object(&args[0])))
}
pub(super) async fn tick(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    preflight(args.view(), Op::DateTick)?;
    let work = allocation::UnboundedCalendarWork::admit(runtime, Op::DateTick.name())
        .map_err(|error| terminal(error, Op::DateTick))?;
    // Reuse the admitted argument owner. A host-only call allocates no second
    // collection; resident transfers retain the original roots and selected context.
    let _gathered = crate::dispatcher::gather_builtin_arguments_async(args, runtime)
        .await
        .map_err(|error| terminal(error, Op::DateTick))?;
    work.finish(Ok(Value::Num(0.)))
        .map_err(|error| terminal(error, Op::DateTick))
}
pub(super) fn calendar(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
    op: Op,
) -> BuiltinResult<crate::execution::RootedValue> {
    preflight(args, op)?;
    let work = allocation::UnboundedCalendarWork::admit(runtime, op.name())
        .map_err(|error| terminal(error, op))?;
    let result = (|| {
        let lhs = &args[0];
        let rhs = &args[1];
        let output = match op {
            Op::Plus if is_datetime_object(rhs) => {
                let (months, days) = calendar_duration_tensors_from_value(lhs)?;
                let serials = serials_from_datetime_value(rhs)?;
                let (serials, shape) =
                    apply_calendar_duration_to_serials(&serials, &months, &days, 1., "plus")?;
                datetime_object_from_serials(
                    runtime,
                    serials,
                    shape,
                    datetime_format_from_value(rhs)?,
                )?
            }
            Op::Plus => super::combine_calendar_durations(runtime, lhs, rhs, 1., "plus")?,
            Op::Minus => super::combine_calendar_durations(runtime, lhs, rhs, -1., "minus")?,
            Op::Eq | Op::Ne => compare(lhs, rhs, op == Op::Ne)?,
            _ => unreachable!("only calendar operations use this kernel"),
        };
        crate::execution::RootedValue::in_optional_context(output, runtime)
    })();
    work.finish(result).map_err(|error| terminal(error, op))
}
fn compare(lhs: &Value, rhs: &Value, invert: bool) -> BuiltinResult<Value> {
    let (lhs_months, lhs_days) = calendar_duration_tensors_from_value(lhs)?;
    let (rhs_months, rhs_days) = calendar_duration_tensors_from_value(rhs)?;
    let (left_months, right_months, shape) = tensor::binary_numeric_tensors(
        &lhs_months,
        &rhs_months,
        "eq",
        super::super::capabilities::BUILTIN_NAME,
    )?;
    let (left_days, right_days, day_shape) = tensor::binary_numeric_tensors(
        &lhs_days,
        &rhs_days,
        "eq",
        super::super::capabilities::BUILTIN_NAME,
    )?;
    if day_shape != shape {
        return Err(datetime_error(
            "eq: calendarDuration operands must have matching component sizes",
        ));
    }
    let values = left_months
        .iter()
        .zip(right_months.iter())
        .zip(left_days.iter().zip(right_days.iter()))
        .map(|((lm, rm), (ld, rd))| {
            let equal = super::calendar_component_equal(*lm, *rm)
                && super::calendar_component_equal(*ld, *rd);
            if equal != invert {
                1.
            } else {
                0.
            }
        })
        .collect();
    tensor_or_scalar(values, shape)
}
