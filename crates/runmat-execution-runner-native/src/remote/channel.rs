use async_trait::async_trait;
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_artifact::ProjectRevisionRecord;
use runmat_execution_runner::{AttemptReport, AttemptRequest, WorkerSpec};

use crate::NativeExecutionResult;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RemoteAttempt {
    pub scheduling: AttemptRequest,
    pub program: ProgramExecutionRequest,
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

/// One allocation-scoped worker route owned by the authoritative driver.
///
/// Implementations may use pinned QUIC or the opaque relay, but must preserve
/// the same encrypted application protocol and fencing semantics.
#[async_trait]
pub trait RemoteWorkerChannel: Send + Sync {
    fn node_identity(&self) -> &str;

    fn worker(&self) -> &WorkerSpec;

    async fn install_bundle(
        &self,
        bundle_digest: Digest,
        bundle: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt>;

    async fn activate_bundle(
        &self,
        bundle_digest: Digest,
    ) -> NativeExecutionResult<RemoteBundleReceipt>;

    async fn transfer_value(
        &self,
        reference: ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<RemoteValueReceipt>;

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport>;

    async fn cancel(&self, request: &AttemptRequest) -> NativeExecutionResult<()>;

    async fn drain(&self) -> NativeExecutionResult<()>;
}
