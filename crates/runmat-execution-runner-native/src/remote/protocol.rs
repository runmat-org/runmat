use runmat_execution::identity::AttemptId;
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_runner::AttemptReport;
use serde::{Deserialize, Serialize};

use super::{RemoteAttempt, RemoteBundleReceipt};

pub const REMOTE_WORKER_PROTOCOL_V1: u16 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RemoteWorkerRequest {
    pub schema_version: u16,
    pub correlation_id: String,
    pub driver_fence: u64,
    pub command: RemoteWorkerCommand,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemoteWorkerCommand {
    InstallBundle {
        bundle_digest: Digest,
        bundle: Vec<u8>,
    },
    ActivateBundle {
        bundle_digest: Digest,
    },
    PutValue {
        reference: ValueRef,
        encoded: Vec<u8>,
    },
    Execute {
        attempt: Box<RemoteAttempt>,
    },
    Cancel {
        attempt_id: AttemptId,
    },
    Drain,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RemoteWorkerReply {
    pub schema_version: u16,
    pub correlation_id: String,
    pub outcome: RemoteWorkerOutcome,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemoteWorkerOutcome {
    BundleStored { receipt: RemoteBundleReceipt },
    ValueStored { receipt: super::RemoteValueReceipt },
    Attempt { report: AttemptReport },
    Acknowledged,
    Rejected { message: String },
}
