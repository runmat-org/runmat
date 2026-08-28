use runmat_execution::{ExecutionHandleSnapshot, ExecutionHandleState, PoolSnapshot};
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::RuntimeError;

/// Resolve a property on an execution handle.
///
/// `None` means the value is not an execution handle. Handle properties are
/// queried from the owning service so callers never observe a duplicated or
/// guessed lifecycle state.
pub fn load_member(
    context: &RuntimeContext,
    value: &Value,
    property: &str,
) -> Option<Result<Value, RuntimeError>> {
    if let Some(handle) = super::future::execution_value(value) {
        if !std::ptr::eq(handle, value) {
            return load_member(context, handle, property);
        }
    }
    match value {
        Value::Future(handle) => Some(
            context
                .execution()
                .inspect_future(handle)
                .map_err(|error| service_error("future", error))
                .and_then(|snapshot| {
                    execution_property("future", handle.id.to_string(), snapshot, property)
                }),
        ),
        Value::Task(handle) => Some(
            context
                .execution()
                .inspect_task(handle)
                .map_err(|error| service_error("future", error))
                .and_then(|snapshot| {
                    execution_property("future", handle.id.to_string(), snapshot, property)
                }),
        ),
        Value::Pool(handle) => Some(
            context
                .execution()
                .inspect_pool(handle)
                .map_err(|error| service_error("pool", error))
                .and_then(|snapshot| pool_property(snapshot, property)),
        ),
        Value::Job(handle) => Some(match property.to_ascii_lowercase().as_str() {
            "id" => Ok(Value::String(handle.id.to_string())),
            "runid" => Ok(Value::String(handle.run_id.to_string())),
            "numoutputarguments" => Ok(Value::Num(f64::from(handle.outputs.requested_outputs))),
            _ => Err(unknown_property("job", property)),
        }),
        _ => None,
    }
}

fn execution_property(
    kind: &str,
    id: String,
    snapshot: ExecutionHandleSnapshot,
    property: &str,
) -> Result<Value, RuntimeError> {
    match property.to_ascii_lowercase().as_str() {
        "id" => Ok(Value::String(id)),
        "state" => Ok(Value::String(state_name(snapshot.state).into())),
        "numoutputarguments" => Ok(Value::Num(f64::from(snapshot.outputs.requested_outputs))),
        "read" => Ok(Value::Bool(snapshot.read)),
        _ => Err(unknown_property(kind, property)),
    }
}

fn pool_property(snapshot: PoolSnapshot, property: &str) -> Result<Value, RuntimeError> {
    match property.to_ascii_lowercase().as_str() {
        "id" => Ok(Value::String(snapshot.handle.id.to_string())),
        "numworkers" => Ok(Value::Num(f64::from(snapshot.workers))),
        "connected" => Ok(Value::Bool(matches!(
            snapshot.state,
            runmat_execution::PoolState::Ready | runmat_execution::PoolState::Resizing
        ))),
        "state" => Ok(Value::String(
            format!("{:?}", snapshot.state).to_ascii_lowercase(),
        )),
        "backend" => Ok(Value::String(
            format!("{:?}", snapshot.backend).to_ascii_lowercase(),
        )),
        _ => Err(unknown_property("pool", property)),
    }
}

fn state_name(state: ExecutionHandleState) -> &'static str {
    match state {
        ExecutionHandleState::Deferred | ExecutionHandleState::Queued => "queued",
        ExecutionHandleState::Running => "running",
        ExecutionHandleState::Finished => "finished",
        ExecutionHandleState::Failed => "failed",
        ExecutionHandleState::Cancelled => "cancelled",
    }
}

fn unknown_property(kind: &str, property: &str) -> RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:parallel:UnknownProperty",
        format!("parallel {kind} has no property '{property}'"),
    )
}

fn service_error(kind: &str, error: crate::execution::ExecutionServiceError) -> RuntimeError {
    crate::build_runtime_error(format!("parallel {kind}: {error}"))
        .with_identifier("RunMat:parallel:ExecutionService")
        .build()
}
