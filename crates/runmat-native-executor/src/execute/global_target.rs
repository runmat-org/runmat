//! Typed access to one session-global destination.
//!
//! Native locals and workspace names are presentation identities. Session
//! storage is selected only by `GlobalBindingTarget`, including legacy slots
//! whose number need not match the native local slot.

use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::session::{PreparedSessionGlobalWrite, SessionVariableAuthority};
use runmat_runtime::workspace::GlobalBindingTarget;
use runmat_value::Value;

pub(super) fn alias_admission_error(
    context: &RuntimeContext,
    cause: runmat_runtime::context::host_allocation::HostAllocationError,
) -> runmat_runtime::RuntimeError {
    crate::error::allocation_error(
        Some(context),
        format!("native global-alias storage could not be admitted: {cause}"),
        cause,
    )
}

pub(super) fn current_value(
    authority: &SessionVariableAuthority,
    target: &GlobalBindingTarget,
) -> Result<Option<Value>, Box<runmat_runtime::RuntimeError>> {
    match target {
        GlobalBindingTarget::Named(name) => authority.global_value(name.as_str()).map_err(Box::new),
        GlobalBindingTarget::LegacySlot(slot) => authority.global_slot(*slot).map_err(Box::new),
    }
}

pub(super) fn prepare_store(
    authority: &SessionVariableAuthority,
    context: &RuntimeContext,
    target: &GlobalBindingTarget,
    value: Value,
) -> Result<PreparedSessionGlobalWrite, Box<runmat_runtime::RuntimeError>> {
    match target {
        GlobalBindingTarget::Named(name) => authority
            .prepare_store_global_with_context(context, name.as_str(), value)
            .map_err(Box::new),
        GlobalBindingTarget::LegacySlot(slot) => authority
            .prepare_store_global_slot_with_context(context, *slot, value)
            .map_err(Box::new),
    }
}

#[cfg(test)]
pub(super) fn store_legacy(
    target: &GlobalBindingTarget,
    value: Value,
) -> Result<(), Box<runmat_runtime::RuntimeError>> {
    match target {
        GlobalBindingTarget::Named(name) => {
            runmat_runtime::workspace::session::store_global_named(name.as_str(), value)
                .map_err(Box::new)
        }
        GlobalBindingTarget::LegacySlot(slot) => {
            runmat_runtime::workspace::session::store_global_slot(*slot, value).map_err(Box::new)
        }
    }
}
