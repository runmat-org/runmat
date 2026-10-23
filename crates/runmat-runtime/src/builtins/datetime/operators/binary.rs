//! Existing scalar-or-exact host calendar kernel, fenced by selected Runtime authority.
use super::super::capabilities::{BUILTIN_NAME, DATETIME_CLASS};
use super::super::*;
use runmat_builtins::DatetimeBinaryOperation;
#[cfg(test)]
#[path = "binary/tests.rs"]
mod tests;

pub(super) fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
    operation: DatetimeBinaryOperation,
) -> crate::BuiltinResult<Value> {
    let _scope = runtime.map(|runtime| runtime.enter());
    crate::output_count::validate_fixed_outputs(
        operation.catalog_entry().descriptor,
        operation.output_error(),
        operation.name(),
    )
    .map_err(|error| terminal(error, operation))?;
    if args.len() != 2 {
        return Err(terminal(
            crate::runtime_descriptor_error_with_detail(
                operation.name(),
                operation.argument_error(),
                "expected datetime receiver and RHS",
            ),
            operation,
        ));
    }
    let work = allocation::UnboundedCalendarWork::admit(runtime, operation.name())
        .map_err(|error| terminal(error, operation))?;
    let result = (|| {
        if args
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
        {
            return Err(crate::runtime_descriptor_error_with_detail(
                operation.name(),
                &runmat_builtins::DATETIME_OPERATOR_ERROR_INVALID_INPUT,
                "resident operands are not admitted and are not gathered",
            ));
        }
        let lhs = &args[0];
        let rhs = &args[1];
        match operation {
            DatetimeBinaryOperation::Plus => plus(runtime, lhs, rhs),
            DatetimeBinaryOperation::Minus => minus(runtime, lhs, rhs),
            _ => compare(lhs, rhs, operation),
        }
    })();
    work.finish(result.map_err(|error| terminal(error, operation)))
        .map_err(|error| terminal(error, operation))
}
fn terminal(
    mut error: crate::RuntimeError,
    operation: DatetimeBinaryOperation,
) -> crate::RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error.context.builtin = Some(operation.name().into());
    if error.identifier.is_none() {
        error.identifier = runmat_builtins::DATETIME_OPERATOR_ERROR_INVALID_INPUT
            .identifier
            .map(Into::into);
    }
    error
}

fn datetime_binary_serials(
    lhs: &Value,
    rhs: &Value,
    context: &str,
) -> BuiltinResult<(Tensor, Tensor, Vec<usize>)> {
    let lhs_serials = serials_from_datetime_value(lhs)?;
    let rhs_serials = match rhs {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj)?,
        _ => serial_tensor_from_value(rhs.clone(), context)?,
    };
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_serials, &rhs_serials, context, BUILTIN_NAME)?;
    let left_tensor = Tensor::new(left, shape.clone())
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let right_tensor = Tensor::new(right, shape.clone())
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    Ok((left_tensor, right_tensor, shape))
}

fn compare(lhs: &Value, rhs: &Value, operation: DatetimeBinaryOperation) -> BuiltinResult<Value> {
    let (left, right, shape) = datetime_binary_serials(lhs, rhs, operation.method())?;
    let left_values = tensor::tensor_values_f64_cow(&left);
    let right_values = tensor::tensor_values_f64_cow(&right);
    let out = left_values
        .iter()
        .zip(right_values.iter())
        .map(|(&left, &right)| {
            let matches = match operation {
                DatetimeBinaryOperation::Eq => super::calendar_component_equal(left, right),
                DatetimeBinaryOperation::Ne => !super::calendar_component_equal(left, right),
                DatetimeBinaryOperation::Lt => left < right,
                DatetimeBinaryOperation::Le => left <= right,
                DatetimeBinaryOperation::Gt => left > right,
                DatetimeBinaryOperation::Ge => left >= right,
                _ => unreachable!("only comparison operations enter comparison kernel"),
            };
            if matches {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    tensor_or_scalar(out, shape)
}
fn plus(
    runtime: Option<&crate::context::RuntimeContext>,
    lhs: &Value,
    rhs: &Value,
) -> crate::BuiltinResult<Value> {
    let lhs_serials = serials_from_datetime_value(lhs)?;
    if is_calendar_duration_object(rhs) {
        let (months, days) = calendar_duration_tensors_from_value(rhs)?;
        let (serials, shape) =
            apply_calendar_duration_to_serials(&lhs_serials, &months, &days, 1.0, "plus")?;
        return datetime_object_from_serials(
            runtime,
            serials,
            shape,
            datetime_format_from_value(lhs)?,
        );
    }
    let rhs_numeric = if crate::builtins::duration::is_duration_object(rhs) {
        crate::builtins::duration::duration_tensor_from_duration_value(rhs)?
    } else {
        serial_tensor_from_value(rhs.clone(), "plus")?
    };
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_serials, &rhs_numeric, "plus", BUILTIN_NAME)?;
    let serials = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    datetime_object_from_serials(runtime, serials, shape, datetime_format_from_value(lhs)?)
}
fn minus(
    runtime: Option<&crate::context::RuntimeContext>,
    lhs: &Value,
    rhs: &Value,
) -> crate::BuiltinResult<Value> {
    let lhs_serials = serials_from_datetime_value(lhs)?;
    match rhs {
        _ if is_calendar_duration_object(rhs) => {
            let (months, days) = calendar_duration_tensors_from_value(rhs)?;
            let (serials, shape) =
                apply_calendar_duration_to_serials(&lhs_serials, &months, &days, -1.0, "minus")?;
            datetime_object_from_serials(runtime, serials, shape, datetime_format_from_value(lhs)?)
        }
        _ if crate::builtins::duration::is_duration_object(rhs) => {
            let rhs_days = crate::builtins::duration::duration_tensor_from_duration_value(rhs)?;
            let (left, right, shape) =
                tensor::binary_numeric_tensors(&lhs_serials, &rhs_days, "minus", BUILTIN_NAME)?;
            let serials = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            datetime_object_from_serials(runtime, serials, shape, datetime_format_from_value(lhs)?)
        }
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => {
            let rhs_serials = serial_tensor_for_object(obj)?;
            let (left, right, shape) =
                tensor::binary_numeric_tensors(&lhs_serials, &rhs_serials, "minus", BUILTIN_NAME)?;
            let deltas = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            tensor_or_scalar(deltas, shape)
        }
        _ => {
            let rhs_numeric = serial_tensor_from_value(rhs.clone(), "minus")?;
            let (left, right, shape) =
                tensor::binary_numeric_tensors(&lhs_serials, &rhs_numeric, "minus", BUILTIN_NAME)?;
            let serials = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            datetime_object_from_serials(runtime, serials, shape, datetime_format_from_value(lhs)?)
        }
    }
}
