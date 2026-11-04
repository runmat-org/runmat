use std::sync::Arc;

use async_trait::async_trait;
use runmat_execution::Digest;
use runmat_execution_runner::{AttemptReport, AttemptRequest, WorkerSpec};
use runmat_execution_transport_native::overlay::WebSocketRelayConnection;

use super::route::RelayFrameRoute;
use super::{
    QuicRemoteWorkerChannel, RemoteAttempt, RemoteBundleReceipt, RemoteObjectReceipt,
    RemoteValueReceipt, RemoteWorkerChannel, RemoteWorkerChannelConfig,
};
use crate::NativeExecutionResult;

/// Allocation-scoped worker channel over the Server's opaque WebSocket relay.
///
/// The application protocol and encryption are shared with the direct QUIC
/// channel; the relay only changes the frame transport.
pub struct RelayRemoteWorkerChannel {
    protocol: Arc<QuicRemoteWorkerChannel>,
}

impl RelayRemoteWorkerChannel {
    pub async fn connect(
        url: &str,
        headers: &[(String, String)],
        config: RemoteWorkerChannelConfig,
    ) -> NativeExecutionResult<Arc<Self>> {
        let connection = WebSocketRelayConnection::connect(url, headers, config.limits)
            .await
            .map_err(|error| crate::NativeExecutionError::Protocol(error.to_string()))?;
        let protocol = QuicRemoteWorkerChannel::connect_route(
            config,
            Arc::new(RelayFrameRoute::new(connection)),
        )
        .await?;
        Ok(Arc::new(Self { protocol }))
    }
}

#[async_trait]
impl RemoteWorkerChannel for RelayRemoteWorkerChannel {
    fn node_identity(&self) -> &str {
        self.protocol.node_identity()
    }

    fn worker(&self) -> &WorkerSpec {
        self.protocol.worker()
    }

    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec> {
        self.protocol.describe_worker().await
    }

    fn verified_software_capabilities(
        &self,
    ) -> Option<std::collections::BTreeSet<runmat_execution::resource::Capability>> {
        self.protocol.verified_software_capabilities()
    }

    async fn install_bundle(
        &self,
        bundle_digest: Digest,
        bundle: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        self.protocol.install_bundle(bundle_digest, bundle).await
    }

    async fn stage_graph_program(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
        encrypted_manifest: &[u8],
        downloads: &[runmat_execution_transport_native::control::DriverArtifactDownload],
        maximum_program_bytes: u64,
    ) -> NativeExecutionResult<super::RemoteGraphProgramReceipt> {
        self.protocol
            .stage_graph_program(
                attempt_id,
                encrypted_manifest,
                downloads,
                maximum_program_bytes,
            )
            .await
    }

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        self.protocol.execute(attempt).await
    }

    async fn authenticated_physical_retirement(
        &self,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<Option<super::AuthenticatedPhysicalRetirement>> {
        self.protocol
            .authenticated_physical_retirement(request)
            .await
    }

    async fn acknowledge_physical_retirement(
        &self,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<()> {
        self.protocol.acknowledge_physical_retirement(request).await
    }

    async fn prune_durable_checkpoint(
        &self,
        sequence: u64,
        attempts: &[super::CheckpointedAttempt],
    ) -> NativeExecutionResult<()> {
        self.protocol
            .prune_durable_checkpoint(sequence, attempts)
            .await
    }

    fn drain_progress(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
    ) -> Vec<crate::ProgramProgress> {
        self.protocol.drain_progress(attempt_id)
    }

    fn drain_collective_requests(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
    ) -> Vec<runmat_execution::CollectiveRequest> {
        self.protocol.drain_collective_requests(attempt_id)
    }

    fn drain_fea_calls(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
    ) -> Vec<crate::fea_proxy::Call> {
        self.protocol.drain_fea_calls(attempt_id)
    }

    async fn complete_fea_call(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
        sequence: u64,
        reply: crate::fea_proxy::Reply,
    ) -> NativeExecutionResult<()> {
        self.protocol
            .complete_fea_call(attempt_id, sequence, reply)
            .await
    }

    async fn complete_collective(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
        request: &runmat_execution::CollectiveRequest,
        result: crate::protocol::CollectiveProcessResult,
    ) -> NativeExecutionResult<()> {
        self.protocol
            .complete_collective(attempt_id, request, result)
            .await
    }

    async fn activate_bundle(
        &self,
        bundle_digest: Digest,
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        self.protocol.activate_bundle(bundle_digest).await
    }

    async fn transfer_value(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<RemoteValueReceipt> {
        self.protocol.transfer_value(reference, encoded).await
    }

    async fn transfer_object(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<RemoteObjectReceipt> {
        self.protocol.transfer_object(reference, encoded).await
    }

    async fn download_object(
        &self,
        reference: runmat_execution::value::ValueRef,
    ) -> NativeExecutionResult<Vec<u8>> {
        self.protocol.download_object(reference).await
    }

    async fn cancel(
        &self,
        request: &AttemptRequest,
        reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()> {
        self.protocol.cancel(request, reason).await
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        self.protocol.drain().await
    }
}
