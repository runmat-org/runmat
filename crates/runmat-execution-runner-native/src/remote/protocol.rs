use runmat_execution::identity::AttemptId;
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_runner::AttemptReport;
use serde::{Deserialize, Serialize};

use runmat_execution_transport_native::transfer::ObjectChunk;

use super::{RemoteAttempt, RemoteBundleReceipt};

pub const REMOTE_WORKER_PROTOCOL_V2: u16 = 2;

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
    ProbeObject {
        reference: ValueRef,
    },
    PutObjectChunk {
        reference: ValueRef,
        chunk: ObjectChunk,
    },
    GetObjectChunk {
        reference: ValueRef,
        offset: u64,
        maximum_bytes: u32,
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
    ObjectPosition { receipt: super::RemoteObjectReceipt },
    ObjectChunk { chunk: ObjectChunk, complete: bool },
    Attempt { report: AttemptReport },
    Acknowledged,
    Rejected { message: String },
}
