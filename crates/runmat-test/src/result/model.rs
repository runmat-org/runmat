use serde::{Deserialize, Serialize};

use crate::descriptor::TestDescriptor;
use crate::identity::{RunId, TestId};

use super::{AttemptResult, ResultState};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TestResult {
    pub test_id: TestId,
    /// Plan-authored presentation, separate from the stable opaque TestId.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report_identity: Option<TestReportIdentity>,
    pub state: ResultState,
    pub attempts: Vec<AttemptResult>,
    pub flaky: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TestReportIdentity {
    pub source_path: String,
    pub suite_name: String,
    pub display_name: String,
    pub case_name: String,
}

impl TestReportIdentity {
    pub fn from_plan(suite_name: &str, test: &TestDescriptor) -> Self {
        let case_name = test
            .display_name
            .strip_prefix(suite_name)
            .and_then(|tail| tail.strip_prefix('/'))
            .unwrap_or(&test.display_name)
            .to_owned();
        Self {
            source_path: test.procedure.source.relative_path.clone(),
            suite_name: suite_name.to_owned(),
            display_name: test.display_name.clone(),
            case_name,
        }
    }

    pub fn human_label(&self) -> String {
        format!("{}: {}", self.source_path, self.display_name)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub run_id: RunId,
    pub state: ResultState,
    pub tests: Vec<TestResult>,
}
