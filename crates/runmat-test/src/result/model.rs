use serde::{Deserialize, Serialize};

use crate::identity::{RunId, TestId};

use super::{AttemptResult, ResultState};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TestResult {
    pub test_id: TestId,
    pub state: ResultState,
    pub attempts: Vec<AttemptResult>,
    pub flaky: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub run_id: RunId,
    pub state: ResultState,
    pub tests: Vec<TestResult>,
}
