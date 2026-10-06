//! Timer-handle method dispatch in MATLAB linear order over scalar, cell, and
//! homogeneous object-array receivers.

use super::*;

pub(super) async fn start_timer_values(
    runtime: &RuntimeContext,
    value: &Value,
    override_delay: Option<std::time::Duration>,
) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_timer_handle(handle) => {
            start_one_timer(runtime, handle.clone(), override_delay).await
        }
        Value::Cell(cell) => {
            for item in cell.iter_column_major() {
                Box::pin(start_timer_values(runtime, item, override_delay)).await?;
            }
            Ok(())
        }
        Value::ObjectArray(array) if array.class_name().is(TIMER_CLASS) => {
            for item in array.data() {
                Box::pin(start_timer_values(runtime, item, override_delay)).await?;
            }
            Ok(())
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: expected timer handle, got {other:?}"),
        )),
    }
}

pub(super) async fn stop_timer_values(
    runtime: &RuntimeContext,
    value: &Value,
    call_stop: bool,
) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_timer_handle(handle) => {
            stop_one_timer(runtime, handle, call_stop).await
        }
        Value::Cell(cell) => {
            for item in cell.iter_column_major() {
                Box::pin(stop_timer_values(runtime, item, call_stop)).await?;
            }
            Ok(())
        }
        Value::ObjectArray(array) if array.class_name().is(TIMER_CLASS) => {
            for item in array.data() {
                Box::pin(stop_timer_values(runtime, item, call_stop)).await?;
            }
            Ok(())
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: expected timer handle, got {other:?}"),
        )),
    }
}

pub(super) async fn wait_timer_values(
    runtime: &RuntimeContext,
    value: &Value,
) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_timer_handle(handle) => {
            wait_one_timer(runtime, handle).await
        }
        Value::Cell(cell) => {
            for item in cell.iter_column_major() {
                Box::pin(wait_timer_values(runtime, item)).await?;
            }
            Ok(())
        }
        Value::ObjectArray(array) if array.class_name().is(TIMER_CLASS) => {
            for item in array.data() {
                Box::pin(wait_timer_values(runtime, item)).await?;
            }
            Ok(())
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: expected timer handle, got {other:?}"),
        )),
    }
}

pub(super) async fn delete_timer_values(
    runtime: &RuntimeContext,
    value: &Value,
) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_timer_handle(handle) => {
            stop_one_timer(runtime, handle, false).await?;
            crate::set_handle_valid(handle, false);
            let pending = runtime.timer_session().borrow_mut().delete(handle);
            if let Some(lease) = pending {
                lease.cancel();
            }
            Ok(())
        }
        Value::Cell(cell) => {
            for item in cell.iter_column_major() {
                Box::pin(delete_timer_values(runtime, item)).await?;
            }
            Ok(())
        }
        Value::ObjectArray(array) if array.class_name().is(TIMER_CLASS) => {
            for item in array.data() {
                Box::pin(delete_timer_values(runtime, item)).await?;
            }
            Ok(())
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: expected timer handle, got {other:?}"),
        )),
    }
}
