use serde::{Deserialize, Serialize};

use crate::identity::TestId;

use super::{Artifact, Diagnostic, ResultState};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptResult {
    pub test_id: TestId,
    pub attempt: u32,
    pub state: ResultState,
    #[serde(default)]
    pub diagnostics: Vec<Diagnostic>,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    #[serde(default)]
    pub output: String,
    pub abort_run: bool,
}
