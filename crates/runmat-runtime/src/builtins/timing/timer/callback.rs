//! Timer callback admission and event delivery through shared executor ports.

use super::*;
use crate::context::SessionSourceEvaluation;
use runmat_async::RuntimeErrorOwner;
use runmat_value::StructValue;
use std::sync::Arc;

/// An executable callback slot. Property lookup and event presentation derive
/// from the same identity, so callers cannot pair different callback names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TimerCallback {
    Timer,
    Start,
    Stop,
    Error,
}

impl TimerCallback {
    pub(super) const ALL: [Self; 4] = [Self::Timer, Self::Start, Self::Stop, Self::Error];

    pub(super) const fn property(self) -> TimerProperty {
        match self {
            Self::Timer => TimerProperty::TimerFcn,
            Self::Start => TimerProperty::StartFcn,
            Self::Stop => TimerProperty::StopFcn,
            Self::Error => TimerProperty::ErrorFcn,
        }
    }
}

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
    callback: TimerCallback,
) -> BuiltinResult<()> {
    run_callback_with_error(runtime, handle, callback, None).await
}

pub(super) async fn run_error_callback(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    error: RuntimeErrorOwner,
) -> BuiltinResult<()> {
    run_callback_with_error(runtime, handle, TimerCallback::Error, Some(error)).await
}

async fn run_callback_with_error(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    slot: TimerCallback,
    error: Option<RuntimeErrorOwner>,
) -> BuiltinResult<()> {
    let callback = property(handle, slot.property().name())?;
    if callback_is_empty(&callback) {
        return Ok(());
    }
    match callback {
        Value::String(text) if !text.trim().is_empty() => {
            text_callback_request(runtime, handle, slot, text)?
                .evaluate(runtime.clone())
                .await?;
        }
        Value::CharArray(chars) if !chars.data.is_empty() => {
            let text: String = chars.data.iter().collect();
            if !text.trim().is_empty() {
                text_callback_request(runtime, handle, slot, text)?
                    .evaluate(runtime.clone())
                    .await?;
            }
        }
        Value::Cell(cell) => {
            let Some(function) = cell.data.first().cloned() else {
                return Ok(());
            };
            let mut args = callback_base_args(runtime, handle, slot, error.as_ref());
            args.extend(cell.data.iter().skip(1).cloned());
            crate::call_feval_async_with_outputs_in_context(runtime, function, &args, 0).await?;
        }
        function @ (Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)) => {
            let args = callback_base_args(runtime, handle, slot, error.as_ref());
            crate::call_feval_async_with_outputs_in_context(runtime, function, &args, 0).await?;
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn text_callback_request(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    callback: TimerCallback,
    text: String,
) -> BuiltinResult<SessionSourceEvaluation> {
    let label: Arc<str> = Arc::from(format!(
        "<timer:{}:{}>",
        string_property(handle, "Name")?,
        callback.property().name()
    ));
    SessionSourceEvaluation::prepare(runtime, Arc::from(text), label, 0)
}

fn callback_base_args(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    callback: TimerCallback,
    error: Option<&RuntimeErrorOwner>,
) -> Vec<Value> {
    vec![
        Value::HandleObject(handle.clone()),
        timer_event(runtime, callback, error),
    ]
}

fn timer_event(
    runtime: &RuntimeContext,
    callback: TimerCallback,
    error: Option<&RuntimeErrorOwner>,
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
    event.insert(
        "Type",
        Value::String(callback.property().name().to_string()),
    );
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn callback_slots_share_property_and_event_identity() {
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
        let expected = [
            TimerProperty::TimerFcn,
            TimerProperty::StartFcn,
            TimerProperty::StopFcn,
            TimerProperty::ErrorFcn,
        ];
        for (slot, property) in TimerCallback::ALL.into_iter().zip(expected) {
            assert_eq!(slot.property(), property);
            let Value::Struct(event) = timer_event(&runtime, slot, None) else {
                panic!("timer event must be a structure");
            };
            assert!(
                matches!(event.fields.get("Type"), Some(Value::String(name)) if name == property.name())
            );
            let Some(Value::Struct(data)) = event.fields.get("Data") else {
                panic!("timer event must contain Data");
            };
            assert!(!data.fields.contains_key("Error"));
        }
    }
}
