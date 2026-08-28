use serde::{Deserialize, Serialize};

use crate::identity::{RunId, TestId};
use crate::lifecycle::{ExecutionPhase, QualificationKind};
use crate::result::{Artifact, AttemptResult, Diagnostic, RunResult};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TestEvent {
    pub sequence: u64,
    pub run_id: RunId,
    pub payload: TestEventPayload,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum TestEventPayload {
    RunStarted,
    TestStarted {
        test_id: TestId,
        attempt: u32,
    },
    PhaseStarted {
        test_id: TestId,
        attempt: u32,
        phase: ExecutionPhase,
        procedure: String,
    },
    PhaseFinished {
        test_id: TestId,
        attempt: u32,
        phase: ExecutionPhase,
        procedure: String,
    },
    Qualification {
        test_id: TestId,
        attempt: u32,
        kind: QualificationKind,
        diagnostic: Diagnostic,
    },
    Diagnostic {
        test_id: TestId,
        attempt: u32,
        diagnostic: Diagnostic,
    },
    Output {
        test_id: TestId,
        attempt: u32,
        text: String,
        truncated: bool,
    },
    Artifact {
        test_id: TestId,
        attempt: u32,
        artifact: Artifact,
    },
    Plugin {
        plugin: String,
        hook: String,
        status: PluginStatus,
        message: Option<String>,
    },
    TestFinished {
        result: AttemptResult,
    },
    RunFinished {
        result: RunResult,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginStatus {
    Completed,
    Failed,
}
