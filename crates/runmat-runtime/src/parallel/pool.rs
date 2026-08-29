use runmat_execution::{PoolBackend, PoolRequest};
use runmat_value::{Tensor, Value};

use crate::context::RuntimeContext;
use crate::RuntimeError;

const IDENT_PARPOOL_INVALID_INPUT: &str = "RunMat:parpool:InvalidInput";
const IDENT_GCP_INVALID_INPUT: &str = "RunMat:gcp:InvalidInput";

pub fn ensure(context: &RuntimeContext, arguments: &[Value]) -> Result<Value, RuntimeError> {
    let request = parse_pool_request(arguments)?;
    let previous = context
        .execution()
        .current_pool()
        .map_err(|error| execution_error("parpool", error))?;
    let snapshot = context
        .execution()
        .ensure_pool(request)
        .map_err(|error| execution_error("parpool", error))?;
    if let Some(previous) = previous.filter(|previous| previous.handle != snapshot.handle) {
        crate::parallel::lease::retire_pool(context, &previous.handle)?;
    }
    Ok(Value::Pool(snapshot.handle))
}

pub fn current(context: &RuntimeContext, arguments: &[Value]) -> Result<Value, RuntimeError> {
    let create = match arguments {
        [] => true,
        [value] if text(value).as_deref() == Some("nocreate") => false,
        _ => {
            return Err(crate::runtime_error::semantic_error(
                IDENT_GCP_INVALID_INPUT,
                "gcp accepts no arguments or the option 'nocreate'",
            ))
        }
    };
    let snapshot = if create {
        Some(
            context
                .execution()
                .ensure_pool(PoolRequest::automatic())
                .map_err(|error| execution_error("gcp", error))?,
        )
    } else {
        context
            .execution()
            .current_pool()
            .map_err(|error| execution_error("gcp", error))?
    };
    Ok(snapshot
        .map(|snapshot| Value::Pool(snapshot.handle))
        .unwrap_or_else(empty_double))
}

pub fn close(
    context: &RuntimeContext,
    pool: &runmat_execution::PoolHandle,
) -> Result<(), RuntimeError> {
    context
        .execution()
        .close_pool(pool)
        .map_err(|error| execution_error("delete", error))?;
    crate::parallel::lease::retire_pool(context, pool)
}

fn parse_pool_request(arguments: &[Value]) -> Result<PoolRequest, RuntimeError> {
    match arguments {
        [] => Ok(PoolRequest::automatic()),
        [value] => {
            if let Some(workers) = worker_count(value) {
                return Ok(PoolRequest {
                    backend: None,
                    workers: Some(workers),
                });
            }
            let backend = text(value).and_then(|name| backend(&name)).ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    IDENT_PARPOOL_INVALID_INPUT,
                    "parpool expects a positive worker count or a supported pool kind",
                )
            })?;
            Ok(PoolRequest {
                backend,
                workers: None,
            })
        }
        [kind, workers] => {
            let kind = text(kind).ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    IDENT_PARPOOL_INVALID_INPUT,
                    "parpool pool kind must be text",
                )
            })?;
            let workers = worker_count(workers).ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    IDENT_PARPOOL_INVALID_INPUT,
                    "parpool worker count must be a positive integer scalar",
                )
            })?;
            let backend = backend(&kind).ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    IDENT_PARPOOL_INVALID_INPUT,
                    format!("parpool does not support pool kind '{kind}' in this host"),
                )
            })?;
            Ok(PoolRequest {
                backend,
                workers: Some(workers),
            })
        }
        _ => Err(crate::runtime_error::semantic_error(
            IDENT_PARPOOL_INVALID_INPUT,
            "parpool accepts at most a pool kind and worker count",
        )),
    }
}

fn backend(name: &str) -> Option<Option<PoolBackend>> {
    match name.trim().to_ascii_lowercase().as_str() {
        "local" | "automatic" => Some(None),
        "processes" | "localprocesses" | "local-processes" => {
            Some(Some(PoolBackend::LocalProcesses))
        }
        "browser" | "browserworkers" | "browser-workers" => Some(Some(PoolBackend::BrowserWorkers)),
        "remote" => Some(Some(PoolBackend::Remote)),
        _ => None,
    }
}

fn worker_count(value: &Value) -> Option<u32> {
    match value {
        Value::Num(value)
            if value.is_finite()
                && *value >= 1.0
                && value.fract() == 0.0
                && *value <= u32::MAX as f64 =>
        {
            Some(*value as u32)
        }
        Value::Int(value) => value
            .try_to_u64()
            .and_then(|value| (value > 0).then(|| u32::try_from(value).ok()).flatten()),
        _ => None,
    }
}

fn text(value: &Value) -> Option<String> {
    String::try_from(value)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
}

fn empty_double() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn execution_error(
    operation: &str,
    error: crate::execution::ExecutionServiceError,
) -> RuntimeError {
    let identifier = match operation {
        "parpool" => "RunMat:parpool:ExecutionService",
        "gcp" => "RunMat:gcp:ExecutionService",
        "delete" => "RunMat:delete:ExecutionService",
        _ => "RunMat:parallel:ExecutionService",
    };
    crate::build_runtime_error(format!("{operation}: {error}"))
        .with_builtin(operation)
        .with_identifier(identifier)
        .build()
}
