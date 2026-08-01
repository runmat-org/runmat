use serde::{Deserialize, Serialize};

use crate::identity::{RunId, TestId};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TestExecutionContext {
    pub run_id: RunId,
    pub test_id: TestId,
    pub attempt: u32,
    pub random_seed: u64,
}
