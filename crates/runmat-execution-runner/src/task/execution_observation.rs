//! Provider observations are distinct from a driver terminal decision and from
//! complete artifact/resource accounting. Unknown consumption remains unknown.
use runmat_execution::identity::{AttemptId, WorkerId};
use serde::{Deserialize, Serialize};

use super::AttemptRequest;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "termination", rename_all = "snake_case", deny_unknown_fields)]
pub enum AttemptExecutionTermination {
    NotStarted,
    ProcessExited { code: Option<i32>, success: bool },
    Unconfirmed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptExecutionObservation {
    pub attempt_id: AttemptId,
    pub worker_id: WorkerId,
    /// Observed provider invocation interval; neither sampled CPU nor RSS.
    pub wall_millis: u64,
    pub termination: AttemptExecutionTermination,
    /// Framing and progress are included. These are local IPC, not network egress.
    pub local_ipc_sent_bytes: Option<u64>,
    pub local_ipc_received_bytes: Option<u64>,
}

impl AttemptExecutionObservation {
    pub fn validate_against(&self, request: &AttemptRequest) -> Result<(), String> {
        if self.attempt_id != request.id
            || self.worker_id != request.worker_id
            || self.wall_millis == 0
        {
            return Err("execution observation differs from attempt identity or interval".into());
        }
        match self.termination {
            AttemptExecutionTermination::NotStarted
                if self.local_ipc_sent_bytes != Some(0)
                    || self.local_ipc_received_bytes != Some(0) =>
            {
                Err("unstarted execution reports process transport".into())
            }
            AttemptExecutionTermination::ProcessExited { code, success }
                if success != (code == Some(0)) =>
            {
                Err("process exit code and success disagree".into())
            }
            _ => Ok(()),
        }
    }
}
