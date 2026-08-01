use serde::{Deserialize, Serialize};

use crate::event::TestEvent;
use crate::identity::{RunId, TestId};
use crate::plan::TestPlan;
use crate::result::AttemptResult;

use super::ProtocolHandshake;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum WorkerRequest {
    Handshake(ProtocolHandshake),
    InstallPlan { plan: TestPlan },
    Execute { test_id: TestId, attempt: u32 },
    Cancel { run_id: RunId, reason: String },
    Shutdown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum WorkerResponse {
    Handshake(ProtocolHandshake),
    Ready { run_id: RunId },
    Event { event: TestEvent },
    Completed { result: AttemptResult },
    Rejected { code: String, message: String },
    ShutdownComplete,
}
