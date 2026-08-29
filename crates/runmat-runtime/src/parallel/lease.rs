use runmat_execution::{CompositeHandle, DistributedValueHandle, PoolHandle};

use crate::context::RuntimeContext;
use crate::RuntimeError;

pub fn validate_distributed(
    context: &RuntimeContext,
    handle: &DistributedValueHandle,
) -> Result<(), RuntimeError> {
    validate_pool(context, &handle.pool, "distributed value")
}

pub fn validate_composite(
    context: &RuntimeContext,
    handle: &CompositeHandle,
) -> Result<(), RuntimeError> {
    validate_pool(context, &handle.gang.pool, "Composite")
}

pub fn retire_pool(context: &RuntimeContext, pool: &PoolHandle) -> Result<(), RuntimeError> {
    if let Some(distributed) = context.service_ports().distributed() {
        distributed.retire_pool(pool.clone())?;
    }
    Ok(())
}

fn validate_pool(
    context: &RuntimeContext,
    pool: &PoolHandle,
    value_kind: &str,
) -> Result<(), RuntimeError> {
    if let Some(collective) = context.service_ports().collective() {
        let admitted = &collective.context().gang.pool;
        return validate_exact_pool(admitted, pool, value_kind);
    }
    let active = context
        .execution()
        .current_pool()
        .map_err(|error| lease_error(value_kind, error))?
        .ok_or_else(|| lease_error(value_kind, "its pool is closed"))?;
    validate_exact_pool(&active.handle, pool, value_kind)
}

fn validate_exact_pool(
    active: &PoolHandle,
    expected: &PoolHandle,
    value_kind: &str,
) -> Result<(), RuntimeError> {
    if active == expected {
        return Ok(());
    }
    let reason = if active.scope_id != expected.scope_id {
        "it belongs to another execution scope"
    } else if active.id != expected.id {
        "it belongs to another pool"
    } else {
        "its pool generation has been retired"
    };
    Err(lease_error(value_kind, reason))
}

fn lease_error(value_kind: &str, reason: impl std::fmt::Display) -> RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:parallel:StaleDistributedLease",
        format!("{value_kind} is unavailable because {reason}"),
    )
}
