//! Sole migration boundary for ambient runtime APIs.
//!
//! R09 routes existing scoped modules through the active [`RuntimeContext`]
//! here. New semantic code must receive `&RuntimeContext` explicitly. The
//! checked RM-1064 inventory assigns complete removal of this bridge to R29.

use super::{scope::active_runtime_context, RuntimeContext, RuntimeContextLocalId};

pub fn active() -> Option<RuntimeContext> {
    active_runtime_context()
}

/// Read the poll-local workspace owner without retaining or copying its context.
pub fn active_local_identity() -> Option<RuntimeContextLocalId> {
    read_active(|context| context.map(RuntimeContext::local_identity))
}

pub(crate) fn error_namespace() -> Option<String> {
    active_runtime_context().map(|context| context.error_namespace())
}

/// Read-only migration guard without cloning the complete active invocation.
/// Readers must not enter a scope or construct selected work while borrowed.
pub(crate) fn read_active<T>(read: impl FnOnce(Option<&RuntimeContext>) -> T) -> T {
    super::scope::with_active_runtime_context(read)
}
