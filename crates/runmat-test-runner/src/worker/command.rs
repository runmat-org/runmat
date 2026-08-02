use runmat_test::discovery::FrozenTestRunSnapshot;
use runmat_test::identity::{RunId, TestId};
use runmat_test::plan::TestPlan;

use crate::host::IsolationMode;

#[derive(Clone, Debug)]
pub struct RunSubmission {
    pub plan: TestPlan,
    pub snapshot: FrozenTestRunSnapshot,
}

impl RunSubmission {
    pub fn new(plan: TestPlan, snapshot: FrozenTestRunSnapshot) -> crate::RunnerResult<Self> {
        snapshot
            .validate()
            .map_err(|error| crate::RunnerError::InvalidConfiguration(error.to_string()))?;
        if plan.program_revision != snapshot.program_revision {
            return Err(crate::RunnerError::InvalidConfiguration(
                "test plan and frozen source snapshot revisions differ".into(),
            ));
        }
        Ok(Self { plan, snapshot })
    }
}

#[derive(Clone, Debug)]
pub struct SpawnRequest {
    pub submission: RunSubmission,
    pub isolation: IsolationMode,
}

#[derive(Clone, Debug)]
pub struct ExecutionRequest {
    pub test_id: TestId,
    pub attempt: u32,
    pub deadline_ms: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct CancelRequest {
    pub run_id: RunId,
    pub reason: String,
    pub grace_deadline_ms: u64,
}
