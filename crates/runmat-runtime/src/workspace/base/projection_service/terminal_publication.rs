//! Terminal publication of one invocation's authentic staged projection.
//!
//! Executors must destroy their selected child before this owner. It neither
//! mirrors register values nor replays language code. All mutations still use
//! the projection's original prospective admission and publication protocol.

use super::BaseWorkspaceProjectionService;
use crate::context::RuntimeContext;
use crate::RuntimeError;
use std::rc::Rc;

pub struct WorkspaceProjectionPublication {
    projection: Rc<BaseWorkspaceProjectionService>,
    runtime: RuntimeContext,
    active: bool,
}

impl BaseWorkspaceProjectionService {
    /// Retain the selected projection's terminal obligation before execution.
    /// Only existing owner leases are retained; no capacity or grant is minted.
    pub fn terminal_publication(
        self: &Rc<Self>,
        runtime: &RuntimeContext,
    ) -> Result<WorkspaceProjectionPublication, RuntimeError> {
        self.ensure_mutation_context(runtime)?;
        Ok(WorkspaceProjectionPublication {
            projection: Rc::clone(self),
            runtime: runtime.clone(),
            active: true,
        })
    }
}

impl WorkspaceProjectionPublication {
    /// Publish the admitted prefix through its original canonical boundary.
    /// A refusal is terminal: Drop must not retry stale or revoked writes.
    pub fn publish_pending(&mut self) -> Result<(), RuntimeError> {
        if !self.active {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:TerminalPublicationClosed",
                "workspace terminal publication was already refused",
            ));
        }
        let result = self.publish();
        if result.is_err() {
            self.active = false;
        }
        result
    }

    fn publish(&self) -> Result<(), RuntimeError> {
        if self.projection.has_pending_changes()? {
            self.projection.publish_with_context(&self.runtime)?;
        }
        Ok(())
    }
}

impl Drop for WorkspaceProjectionPublication {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            if let Err(error) = self.publish() {
                // Preserve the original Base on authority invalidation. Drop
                // cannot return this cause, force a commit, or queue a replay.
                tracing::error!(error = ?error,
                    "abandoned workspace publication refused; pending delta retired without replay");
            }
        }
    }
}
