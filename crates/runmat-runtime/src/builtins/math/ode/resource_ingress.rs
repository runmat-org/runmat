//! ODE's explicit finite-memory frontier, before options, callbacks or solver storage.
use crate::{context::RuntimeContext, BuiltinResult};
pub(super) fn require_unbounded(
    name: &'static str,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    crate::context::host_allocation::require_unbounded_workspace(runtime,name,"finite host-memory execution is unsupported; ODE workspace and history have no complete resource bound")
}
#[cfg(test)]
#[path = "resource_ingress/tests.rs"]
mod tests;
