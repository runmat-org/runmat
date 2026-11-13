use serde::{Deserialize, Serialize};

use crate::discovery::FrozenTestRunSnapshot;
use crate::event::TestEvent;
use crate::identity::{RunId, TestId};
use crate::plan::TestPlan;
use crate::result::AttemptResult;

use super::ProtocolHandshake;

/// Original physical execution authority, carried only over the owned worker pipe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerExecutionAdmission {
    pub assignment: runmat_execution::ProgramExecutionAssignment,
    pub resources: runmat_execution::resource::ResourceRequest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum WorkerRequest {
    Handshake(ProtocolHandshake),
    InstallPlan {
        plan: Box<TestPlan>,
        snapshot: Box<FrozenTestRunSnapshot>,
    },
    Execute {
        test_id: TestId,
        attempt: u32,
        #[serde(default)]
        admission: Option<WorkerExecutionAdmission>,
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
    BootstrapReady,
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
