//! Sole migration boundary for ambient runtime APIs.
//!
//! R09 routes existing scoped modules through the active [`RuntimeContext`]
//! here. New semantic code must receive `&RuntimeContext` explicitly. The
//! checked RM-1064 inventory assigns complete removal of this bridge to R29.

use super::{scope::active_runtime_context, RuntimeContext};

pub fn active() -> Option<RuntimeContext> {
    active_runtime_context()
}
