use serde::{Deserialize, Serialize};

use crate::identity::{AttemptId, ExecutionScopeId, PoolId, TaskId, WorkerId};
use crate::PoolBackend;

/// Exact scheduler identity of one executing program invocation.
///
/// This is invocation metadata, not part of the immutable program artifact.
/// Hosts attach it only after placement so worker-side introspection observes
/// the same task, pool, and worker identities used by scheduling and fencing.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionAssignment {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub task_id: TaskId,
    pub attempt_id: AttemptId,
    pub worker_id: WorkerId,
    pub backend: PoolBackend,
    /// Exact worker-local resources leased for this attempt. Kernel placement
    /// may use only devices named by this fenced assignment.
    pub resources: crate::resource::ResourceAssignment,
}

impl ProgramExecutionAssignment {
    pub fn validate(&self) -> Result<(), crate::ContractError> {
        self.resources.validate()?;
        for lease in &self.resources.accelerator_leases {
            lease.validate_for_attempt(self.attempt_id)?;
        }
        Ok(())
    }
}
