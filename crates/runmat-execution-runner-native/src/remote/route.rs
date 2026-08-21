use std::sync::Arc;

use async_trait::async_trait;
use runmat_execution_transport_native::frame::WireFrame;
use runmat_execution_transport_native::overlay::{
    QuicOverlayConnection, WebSocketRelayConnection, WebSocketRelayDuplex,
};

use crate::{NativeExecutionError, NativeExecutionResult};

#[async_trait]
pub(crate) trait RemoteFrameRoute: Send + Sync {
    async fn send(&self, frame: WireFrame) -> NativeExecutionResult<()>;
    async fn receive(&self) -> NativeExecutionResult<WireFrame>;
}

pub(crate) struct QuicFrameRoute(pub Arc<QuicOverlayConnection>);

#[async_trait]
impl RemoteFrameRoute for QuicFrameRoute {
    async fn send(&self, frame: WireFrame) -> NativeExecutionResult<()> {
        self.0.send(&frame).await.map_err(protocol)
    }

    async fn receive(&self) -> NativeExecutionResult<WireFrame> {
        self.0.receive().await.map_err(protocol)
    }
}

pub(crate) struct RelayFrameRoute(WebSocketRelayDuplex);

impl RelayFrameRoute {
    pub(crate) fn new(connection: WebSocketRelayConnection) -> Self {
        Self(connection.into_duplex())
    }
}

#[async_trait]
impl RemoteFrameRoute for RelayFrameRoute {
    async fn send(&self, frame: WireFrame) -> NativeExecutionResult<()> {
        self.0.send(frame).await.map_err(protocol)
    }

    async fn receive(&self) -> NativeExecutionResult<WireFrame> {
        self.0.receive().await.map_err(protocol)
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
