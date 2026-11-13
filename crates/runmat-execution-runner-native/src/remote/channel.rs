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

/// Issued only after an encrypted, session-bound worker-owner reply has been
/// checked against the exact scheduler request. Not a wire/deserializable type.
pub struct AuthenticatedPhysicalRetirement {
    request: AttemptRequest,
    usage: Option<super::protocol::RetiredAttemptUsage>,
}

impl AuthenticatedPhysicalRetirement {
    pub(crate) fn issue(request: AttemptRequest) -> Self {
        Self {
            request,
            usage: None,
        }
    }

    pub(super) fn issue_with_usage(
        request: AttemptRequest,
        usage: Option<super::protocol::RetiredAttemptUsage>,
    ) -> NativeExecutionResult<Self> {
        if let Some(usage) = usage {
            usage
                .accounting(&request)
                .map_err(crate::NativeExecutionError::Protocol)?;
        }
        Ok(Self { request, usage })
    }

    pub(super) fn resources(
        &self,
    ) -> Result<Option<runmat_execution_runner::task::AttemptResourceAccounting>, String> {
        self.usage
            .map(|usage| usage.accounting(&self.request))
            .transpose()
    }

    pub(crate) fn matches(&self, request: &AttemptRequest) -> bool {
        &self.request == request
    }
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

    /// Terminal evidence uses the scheduler's existing canonical report owner.
    async fn execute_observed(
        &self,
        attempt: RemoteAttempt,
        _console_budget: Option<runmat_execution::console::ConsoleBudget>,
    ) -> NativeExecutionResult<runmat_execution_runner::port::BackendReport> {
        let request = attempt.scheduling.clone();
        let report = self.execute(attempt).await?;
        Ok(runmat_execution_runner::port::BackendReport::for_request(
            &request, report,
        ))
    }

    /// A non-QUIC implementation has no authority to certify physical owner
    /// retirement. Control acknowledgements and ordinary reports are insufficient.
    async fn authenticated_physical_retirement(
        &self,
        _request: &AttemptRequest,
    ) -> NativeExecutionResult<Option<AuthenticatedPhysicalRetirement>> {
        Ok(None)
    }

    /// Consume the original worker's retained reservation after exact proof;
    /// the loss-recovery path calls this after scheduler acceptance. Ordinary
    /// terminals acknowledge before dispatching subsequent same-route work.
    async fn acknowledge_physical_retirement(
        &self,
        _request: &AttemptRequest,
    ) -> NativeExecutionResult<()> {
        Err(crate::NativeExecutionError::Protocol(
            "remote worker channel cannot acknowledge physical retirement".into(),
        ))
    }

    async fn prune_durable_checkpoint(
        &self,
        _sequence: u64,
        _attempts: &[CheckpointedAttempt],
    ) -> NativeExecutionResult<()> {
        Err(crate::NativeExecutionError::Protocol(
            "remote worker channel cannot reclaim a durable attempt journal".into(),
        ))
    }

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

    fn drain_fea_calls(
        &self,
        _attempt_id: runmat_execution::identity::AttemptId,
    ) -> Vec<crate::fea_proxy::Call> {
        Vec::new()
    }

    async fn complete_fea_call(
        &self,
        _attempt_id: runmat_execution::identity::AttemptId,
        _sequence: u64,
        _reply: crate::fea_proxy::Reply,
    ) -> NativeExecutionResult<()> {
        Err(crate::NativeExecutionError::Protocol(
            "remote worker channel does not support FEA parent control".into(),
        ))
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
/// Exact attempt identity carried only after authoritative durable checkpoint
/// publication. This is a storage-reclamation hint, never retirement proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CheckpointedAttempt {
    pub attempt_id: runmat_execution::identity::AttemptId,
    pub request_digest: Digest,
}
