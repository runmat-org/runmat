//! Timer callback admission and event delivery through shared executor ports.

use super::*;
use crate::context::SessionSourceEvaluation;
use runmat_value::StructValue;
use std::sync::Arc;

pub(super) fn update_average_period(
    handle: &HandleRef,
    instant_period: f64,
    idx: usize,
) -> BuiltinResult<()> {
    let current = numeric_property(handle, "AveragePeriod").unwrap_or(f64::NAN);
    let average = if current.is_finite() {
        (current * (idx as f64 - 1.0) + instant_period) / idx as f64
    } else {
        instant_period
    };
    set_property(handle, "AveragePeriod", Value::Num(average))
}

pub(super) async fn run_callback(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    property_name: &str,
    event_type: &str,
) -> BuiltinResult<()> {
    run_callback_with_error(runtime, handle, property_name, event_type, None).await
}

pub(super) async fn run_error_callback(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    error: Arc<RuntimeError>,
) -> BuiltinResult<()> {
    run_callback_with_error(runtime, handle, "ErrorFcn", "ErrorFcn", Some(error)).await
}

async fn run_callback_with_error(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    property_name: &str,
    event_type: &str,
    error: Option<Arc<RuntimeError>>,
) -> BuiltinResult<()> {
    let callback = property(handle, property_name)?;
    if callback_is_empty(&callback) {
        return Ok(());
    }
    match callback {
        Value::String(text) if !text.trim().is_empty() => {
            text_callback_request(runtime, handle, property_name, text)?
                .evaluate(runtime.clone())
                .await?;
        }
        Value::CharArray(chars) if !chars.data.is_empty() => {
            let text: String = chars.data.iter().collect();
            if !text.trim().is_empty() {
                text_callback_request(runtime, handle, property_name, text)?
                    .evaluate(runtime.clone())
                    .await?;
            }
        }
        Value::Cell(cell) => {
            let Some(function) = cell.data.first().cloned() else {
                return Ok(());
            };
            let mut args = callback_base_args(runtime, handle, event_type, error.as_ref());
            args.extend(cell.data.iter().skip(1).cloned());
            crate::call_feval_async_with_outputs_in_context(runtime, function, &args, 0).await?;
        }
        function @ (Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)) => {
            let args = callback_base_args(runtime, handle, event_type, error.as_ref());
            crate::call_feval_async_with_outputs_in_context(runtime, function, &args, 0).await?;
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn text_callback_request(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    property_name: &str,
    text: String,
) -> BuiltinResult<SessionSourceEvaluation> {
    let label: Arc<str> = Arc::from(format!(
        "<timer:{}:{}>",
        string_property(handle, "Name")?,
        property_name
    ));
    SessionSourceEvaluation::prepare(runtime, Arc::from(text), label, 0)
}

fn callback_base_args(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    event_type: &str,
    error: Option<&Arc<RuntimeError>>,
) -> Vec<Value> {
    vec![
        Value::HandleObject(handle.clone()),
        timer_event(runtime, event_type, error),
    ]
}

fn timer_event(
    runtime: &RuntimeContext,
    event_type: &str,
    error: Option<&Arc<RuntimeError>>,
) -> Value {
    let mut data = StructValue::new();
    data.insert(
        "time",
        Value::Num(
            runtime
                .service_ports()
                .timing()
                .wall_time_since_epoch()
                .as_secs_f64(),
        ),
    );
    if let Some(error) = error {
        data.insert(
            "Error",
            Value::MException(crate::runtime_error::exception_from_error(error.clone())),
        );
    }
    let mut event = StructValue::new();
    event.insert("Type", Value::String(event_type.to_string()));
    event.insert("Data", Value::Struct(data));
    Value::Struct(event)
}

pub(super) fn callback_is_empty(value: &Value) -> bool {
    match value {
        Value::String(text) => text.trim().is_empty(),
        Value::CharArray(chars) => chars.data.iter().collect::<String>().trim().is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        _ => false,
    }
}
