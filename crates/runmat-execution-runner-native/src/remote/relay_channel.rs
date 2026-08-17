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

    async fn install_bundle(
        &self,
        bundle_digest: Digest,
        bundle: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        self.protocol.install_bundle(bundle_digest, bundle).await
    }

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        self.protocol.execute(attempt).await
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

    async fn cancel(&self, request: &AttemptRequest) -> NativeExecutionResult<()> {
        self.protocol.cancel(request).await
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        self.protocol.drain().await
    }
}
