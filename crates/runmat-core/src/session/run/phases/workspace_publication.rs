//! Terminal owner of one Core invocation's already-admitted projection.
//!
//! VM stores stage source-ordered mutations and compiler-binding patches; they
//! do not necessarily commit them to Base before suspension. Ordinary result
//! publication flushes that delta. Abandonment must reach the same canonical
//! boundary after its selected child has been destroyed, not replay registers.

use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::BaseWorkspaceProjectionService;
use runmat_runtime::RuntimeError;
use std::rc::Rc;

pub(super) struct PendingWorkspacePublication {
    // Existing owner leases only: no new shared control or value mirror.
    projection: Rc<BaseWorkspaceProjectionService>,
    runtime: RuntimeContext,
}

impl PendingWorkspacePublication {
    pub(super) fn new(
        projection: &Rc<BaseWorkspaceProjectionService>,
        runtime: &RuntimeContext,
    ) -> Self {
        Self {
            projection: Rc::clone(projection),
            runtime: runtime.clone(),
        }
    }

    fn publish_pending(&self) -> Result<(), RuntimeError> {
        if self.projection.has_pending_changes()? {
            // Context-bound projection capacity and Core's coalesced binding
            // patch were admitted before each mutation. This existing path
            // uses their backing and synchronous final checkpoints; it does
            // not request new capacity or reset cancellation. Displaced
            // owners retire after the canonical/observer borrows end.
            self.projection.publish_with_context(&self.runtime)?;
        }
        Ok(())
    }
}

impl Drop for PendingWorkspacePublication {
    fn drop(&mut self) {
        if let Err(error) = self.publish_pending() {
            // Drop cannot return a diagnostic. A revoked/closed/conflicting
            // owner must still refuse its stale delta, never force a commit
            // or retry against a fresh authority. Retire it normally and keep
            // the full cause observable; the original Base stays canonical.
            tracing::error!(
                error = ?error,
                "abandoned Core workspace publication refused; pending delta retired without replay"
            );
        }
    }
}
