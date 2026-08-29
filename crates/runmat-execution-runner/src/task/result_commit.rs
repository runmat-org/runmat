use runmat_execution::identity::{AttemptId, ResultCommitId};
use serde::{Deserialize, Serialize};

use super::AttemptSuccess;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResultCommit {
    pub id: ResultCommitId,
    pub attempt_id: AttemptId,
    pub driver_fence: u64,
    pub result: AttemptSuccess,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommitDecision {
    Accepted(ResultCommit),
    Duplicate,
    StaleFence,
    StaleAttempt,
}

impl ResultCommit {
    pub fn from_success(attempt_id: AttemptId, driver_fence: u64, success: AttemptSuccess) -> Self {
        let fence = driver_fence.to_be_bytes();
        let id = ResultCommitId::derive(&[attempt_id.bytes(), &fence]);
        Self {
            id,
            attempt_id,
            driver_fence,
            result: success,
        }
    }
}
