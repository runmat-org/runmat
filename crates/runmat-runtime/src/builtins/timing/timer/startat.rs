//! Absolute timer admission: date decoding is owned by `datetime`, while this
//! module maps an admitted calendar instant onto the session timing service.

use super::*;
use std::time::Duration;

const MAX_STARTAT_DELAY: Duration = Duration::from_secs(25 * 24 * 60 * 60);

pub(super) fn is_legacy_relative_scalar(args: crate::call::FunctionArgumentView<'_>) -> bool {
    args.len() == 1 && matches!(args.first(), Some(Value::Num(_) | Value::Int(_)))
        || args.len() == 1
            && matches!(args.first(), Some(Value::Tensor(tensor)) if tensor::is_scalar_tensor(tensor))
}

pub(super) async fn start_at_absolute(
    runtime: &RuntimeContext,
    receiver: &Value,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<()> {
    let (handles, receiver_shape) = receiver_handles(runtime, receiver)?;
    let timing = runtime.service_ports().timing()?;
    let now = timing.wall_time_since_epoch();
    let reference = crate::context::RuntimeCalendarInstant::from_epoch_duration(now)
        .and_then(|instant| timing.local_datetime_at_instant(instant))
        .ok_or_else(|| {
            timer_error(
                &TIMER_ERROR_INVALID_INPUT,
                "startat: host-local calendar is unavailable for this wall-clock instant",
            )
        })?;
    let serials =
        crate::builtins::datetime::timer_firing_arguments(Some(runtime), args, reference)?;
    let serial_values = serials.materialize_f64();
    let scalar = serial_values.len() == 1;
    if !scalar && serials.shape() != receiver_shape.as_slice() {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            "startat: firing-time array must match the timer array size",
        ));
    }
    let mut delays = Vec::with_capacity(handles.len());
    for index in 0..handles.len() {
        let serial_index = if scalar { 0 } else { index };
        let naive = crate::builtins::datetime::naive_from_datenum(serial_values[serial_index])
            .map_err(|error| startat_date_error(error))?;
        let epoch = timing.local_datetime_to_instant(naive).ok_or_else(|| {
            timer_error(
                &TIMER_ERROR_INVALID_INPUT,
                "startat: firing time is ambiguous, nonexistent, or outside the host timezone range",
            )
        })?;
        let delay = epoch
            .to_epoch_duration()
            .and_then(|epoch| epoch.checked_sub(now))
            .ok_or_else(|| {
                timer_error(
                    &TIMER_ERROR_INVALID_INPUT,
                    "startat: firing time is in the past",
                )
            })?;
        if delay > MAX_STARTAT_DELAY {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_INPUT,
                "startat: firing time must be within 25 days",
            ));
        }
        delays.push(delay);
    }
    for (handle, delay) in handles.into_iter().zip(delays) {
        start_one_timer(runtime, handle, Some(delay)).await?;
    }
    Ok(())
}

pub(super) fn receiver_handles(
    runtime: &RuntimeContext,
    receiver: &Value,
) -> BuiltinResult<(Vec<HandleRef>, Vec<usize>)> {
    match receiver {
        Value::HandleObject(handle) => {
            ensure_valid_timer(handle)?;
            if runtime.timer_session().borrow().record(handle).is_none() {
                return Err(timer_error(
                    &TIMER_ERROR_INVALID_HANDLE,
                    "startat: timer belongs to another session",
                ));
            }
            Ok((vec![handle.clone()], vec![1, 1]))
        }
        Value::Cell(cell) => {
            let mut handles = Vec::with_capacity(cell.data().len());
            for item in cell.iter_column_major() {
                let Value::HandleObject(handle) = item else {
                    return Err(timer_error(
                        &TIMER_ERROR_INVALID_HANDLE,
                        "startat: timer array must contain timer handles",
                    ));
                };
                ensure_valid_timer(handle)?;
                if runtime.timer_session().borrow().record(handle).is_none() {
                    return Err(timer_error(
                        &TIMER_ERROR_INVALID_HANDLE,
                        "startat: timer belongs to another session",
                    ));
                }
                handles.push(handle.clone());
            }
            Ok((handles, cell.shape.to_vec()))
        }
        Value::ObjectArray(array) if array.class_name().is(TIMER_CLASS) => {
            let mut handles = Vec::with_capacity(array.len());
            for item in array.data() {
                let Value::HandleObject(handle) = item else {
                    return Err(timer_error(
                        &TIMER_ERROR_INVALID_HANDLE,
                        "startat: timer object array must contain handles",
                    ));
                };
                ensure_valid_timer(handle)?;
                if runtime.timer_session().borrow().record(handle).is_none() {
                    return Err(timer_error(
                        &TIMER_ERROR_INVALID_HANDLE,
                        "startat: timer belongs to another session",
                    ));
                }
                handles.push(handle.clone());
            }
            Ok((handles, array.shape().to_vec()))
        }
        _ => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "startat: expected a timer handle or timer array",
        )),
    }
}

fn startat_date_error(error: RuntimeError) -> RuntimeError {
    build_runtime_error(format!("startat: {}", error.message))
        .with_builtin(BUILTIN_TIMER)
        .with_identifier("RunMat:timer:InvalidInput")
        .with_source(error)
        .build()
}
