use async_trait::async_trait;
use runmat_execution::resource::Capability;
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::{ProgramExecutionRequest, ProjectRevisionRecord};
use runmat_execution_runner::{AttemptReport, AttemptRequest, WorkerSpec};
use runmat_execution_transport_native::control::DriverArtifactDownload;
use std::collections::BTreeSet;

use crate::NativeExecutionResult;
use crate::ProgramProgress;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RemoteObjectReceipt {
    pub value_id: runmat_execution::identity::ValueId,
    pub next_offset: u64,
    pub complete: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RemoteAttempt {
    pub scheduling: AttemptRequest,
    pub program: RemoteAttemptProgram,
}

/// QUIC attempt payload. The Direct V7 variant serializes to its unchanged
/// request object; a Graph attempt carries only an exact staged Program ref.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum RemoteAttemptProgram {
    Direct(Box<ProgramExecutionRequest>),
    GraphStaged(RemoteStagedGraphReference),
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RemoteStagedGraphReference {
    StagedGraphV10 {
        program_digest: Digest,
        program_length: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RemoteBundleReceipt {
    pub bundle_digest: Digest,
    pub bundle_identity: Digest,
    pub project_revision: ProjectRevisionRecord,
    pub stored_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RemoteValueReceipt {
    pub value_id: runmat_execution::identity::ValueId,
    pub encoded_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteGraphProgramReceipt {
    pub program_digest: Digest,
    pub program_length: u64,
}

/// One allocation-scoped worker route owned by the authoritative driver.
///
/// Implementations may use pinned QUIC or the opaque relay, but must preserve
/// the same encrypted application protocol and fencing semantics.
#[async_trait]
pub trait RemoteWorkerChannel: Send + Sync {
    fn node_identity(&self) -> &str;

    fn worker(&self) -> &WorkerSpec;

    /// Fetches the worker description through the channel's encrypted protocol.
    ///
    /// The returned inventory is verified against the allocation metadata before
    /// the worker is admitted to the authoritative driver.
    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec>;

    /// Present only after exact encrypted ReadyV1 validation by this route.
    fn verified_software_capabilities(&self) -> Option<BTreeSet<Capability>> {
        None
    }

    async fn install_bundle(
        &self,
        bundle_digest: Digest,
        bundle: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt>;

    /// Transfer only an encrypted manifest and signed chunk descriptors;
    /// the worker streams each chunk into private staging and issues a receipt
    /// only after the final authenticated Program digest matches.
    async fn stage_graph_program(
        &self,
        _attempt_id: runmat_execution::identity::AttemptId,
        _encrypted_manifest: &[u8],
        _downloads: &[DriverArtifactDownload],
        _maximum_program_bytes: u64,
    ) -> NativeExecutionResult<RemoteGraphProgramReceipt> {
        Err(crate::NativeExecutionError::Protocol(
            "remote worker channel does not support staged Graph Programs".into(),
        ))
    }

    async fn activate_bundle(
        &self,
        bundle_digest: Digest,
    ) -> NativeExecutionResult<RemoteBundleReceipt>;

    async fn transfer_value(
        &self,
        reference: ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<RemoteValueReceipt>;

    async fn transfer_object(
        &self,
        _reference: ValueRef,
        _encoded: &[u8],
    ) -> NativeExecutionResult<RemoteObjectReceipt> {
        Err(crate::NativeExecutionError::Protocol(
            "remote worker channel does not support execution objects".into(),
        ))
    }

    async fn download_object(&self, _reference: ValueRef) -> NativeExecutionResult<Vec<u8>> {
        Err(crate::NativeExecutionError::Protocol(
            "remote worker channel does not support execution objects".into(),
        ))
    }

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport>;

    fn drain_progress(
        &self,
        _attempt_id: runmat_execution::identity::AttemptId,
    ) -> Vec<ProgramProgress> {
        Vec::new()
    }

    fn drain_collective_requests(
        &self,
        _attempt_id: runmat_execution::identity::AttemptId,
    ) -> Vec<runmat_execution::CollectiveRequest> {
        Vec::new()
    }

    async fn complete_collective(
        &self,
        _attempt_id: runmat_execution::identity::AttemptId,
        _request: &runmat_execution::CollectiveRequest,
        _result: crate::protocol::CollectiveProcessResult,
    ) -> NativeExecutionResult<()> {
        Err(crate::NativeExecutionError::Protocol(
            "remote worker channel does not support collectives".into(),
        ))
    }

    /// Deliver the scheduler's original cause; acknowledgement is not a
    /// physical-retirement certificate and cannot release its reservation.
    async fn cancel(
        &self,
        request: &AttemptRequest,
        reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()>;

    async fn drain(&self) -> NativeExecutionResult<()>;
}
