//! Absolute timer admission: date decoding is owned by `datetime`, while this
//! module maps an admitted calendar instant onto the session timing service.

use super::*;
use std::time::Duration;

const MAX_STARTAT_DELAY: Duration = Duration::from_secs(25 * 24 * 60 * 60);

pub(super) fn is_legacy_relative_scalar(args: &[Value]) -> bool {
    matches!(args, [Value::Num(_) | Value::Int(_)])
        || matches!(args, [Value::Tensor(tensor)] if tensor::is_scalar_tensor(tensor))
}

pub(super) async fn start_at_absolute(
    runtime: &RuntimeContext,
    receiver: &Value,
    args: Vec<Value>,
) -> BuiltinResult<()> {
    let (handles, receiver_shape) = receiver_handles(runtime, receiver)?;
    let timing = runtime.service_ports().timing();
    let now = timing.wall_time_since_epoch();
    let reference = timing.local_datetime_at_epoch(now).ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            "startat: host-local calendar is unavailable for this wall-clock instant",
        )
    })?;
    let serials = crate::builtins::datetime::timer_firing_serials(args, reference)?;
    let serial_values = serials.materialize_f64();
    let scalar = serial_values.len() == 1;
    if !scalar && serials.shape != receiver_shape {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            "startat: firing-time array must match the timer array size",
        ));
    }
    let mut delays = Vec::with_capacity(handles.len());
    for index in 0..handles.len() {
        let serial_index = if scalar {
            0
        } else {
            column_major_index(index, &receiver_shape)?
        };
        let naive = crate::builtins::datetime::naive_from_datenum(serial_values[serial_index])
            .map_err(|error| startat_date_error(error))?;
        let epoch = timing.local_datetime_to_epoch(naive).ok_or_else(|| {
            timer_error(
                &TIMER_ERROR_INVALID_INPUT,
                "startat: firing time is ambiguous, nonexistent, or outside the host timezone range",
            )
        })?;
        let delay = epoch.checked_sub(now).ok_or_else(|| {
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

fn receiver_handles(
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
            let mut handles = Vec::with_capacity(cell.data.len());
            for item in &cell.data {
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
            Ok((handles, cell.shape.clone()))
        }
        _ => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "startat: expected a timer handle or timer cell array",
        )),
    }
}

fn column_major_index(row_major_index: usize, shape: &[usize]) -> BuiltinResult<usize> {
    let rows = shape[0];
    let cols = shape[1];
    let page_len = rows.checked_mul(cols).ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            "startat: timer array size overflow",
        )
    })?;
    if page_len == 0 {
        return Ok(0);
    }
    let page = row_major_index / page_len;
    let within_page = row_major_index % page_len;
    let row = within_page / cols;
    let col = within_page % cols;
    Ok(page * page_len + col * rows + row)
}

fn startat_date_error(error: RuntimeError) -> RuntimeError {
    build_runtime_error(format!("startat: {}", error.message))
        .with_builtin(BUILTIN_TIMER)
        .with_identifier("RunMat:timer:InvalidInput")
        .with_source(error)
        .build()
}
