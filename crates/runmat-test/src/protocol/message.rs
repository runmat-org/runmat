use serde::{Deserialize, Serialize};

use crate::discovery::FrozenTestRunSnapshot;
use crate::event::TestEvent;
use crate::identity::{RunId, TestId};
use crate::plan::TestPlan;
use crate::result::AttemptResult;

use super::ProtocolHandshake;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum WorkerRequest {
    Handshake(ProtocolHandshake),
    InstallPlan {
        plan: TestPlan,
        snapshot: FrozenTestRunSnapshot,
    },
    Execute {
        test_id: TestId,
        attempt: u32,
    },
    Cancel {
        run_id: RunId,
        reason: String,
    },
    Shutdown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum WorkerResponse {
    Handshake(ProtocolHandshake),
    Ready {
        run_id: RunId,
    },
    Event {
        event: TestEvent,
    },
    Completed {
        result: AttemptResult,
        coverage: Vec<crate::coverage::CoverageFragment>,
    },
    Rejected {
        code: String,
        message: String,
    },
    ShutdownComplete,
}
