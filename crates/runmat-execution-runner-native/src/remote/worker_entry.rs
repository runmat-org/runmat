use std::sync::Arc;

use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_runner::WorkerSpec;
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{QuicOverlayListener, WebSocketRelayConnection};
use runmat_meshing_execution::MeshingStageKernel;

use super::route::{QuicFrameRoute, RelayFrameRoute};
use super::worker_server::{run_worker_loop, WorkerLoopContext};
use crate::{NativeExecutionError, NativeExecutionResult};

pub struct RemoteWorkerQuicRequest {
    pub listener: QuicOverlayListener,
    pub run_identity: String,
    pub worker: WorkerSpec,
    pub driver_fence: u64,
    pub session_id: [u8; 16],
    pub run_key: RunKeyMaterial,
    pub limits: FrameLimits,
}

pub async fn run_remote_worker_quic(request: RemoteWorkerQuicRequest) -> NativeExecutionResult<()> {
    run_remote_worker_quic_inner(request, None).await
}

pub struct RemoteMeshingWorkerQuicRequest {
    pub worker: RemoteWorkerQuicRequest,
    pub kernel: Arc<dyn MeshingStageKernel>,
    pub meshing_limits: crate::NativeMeshingHostLimits,
}

pub async fn run_remote_meshing_worker_quic(
    request: RemoteMeshingWorkerQuicRequest,
) -> NativeExecutionResult<()> {
    request.meshing_limits.validate()?;
    let meshing_host =
        super::worker_execution::RemoteMeshingHost::new(request.kernel, request.meshing_limits);
    run_remote_worker_quic_inner(request.worker, Some(meshing_host)).await
}

async fn run_remote_worker_quic_inner(
    request: RemoteWorkerQuicRequest,
    meshing_host: Option<super::worker_execution::RemoteMeshingHost>,
) -> NativeExecutionResult<()> {
    let RemoteWorkerQuicRequest {
        listener,
        run_identity,
        worker,
        driver_fence,
        session_id,
        run_key,
        limits,
    } = request;
    let connection = Arc::new(listener.accept().await.map_err(protocol)?);
    tokio::task::LocalSet::new()
        .run_until(run_worker_loop(
            Arc::new(QuicFrameRoute(connection)),
            WorkerLoopContext {
                run_identity,
                worker,
                driver_fence,
                session_id,
                run_key,
                limits,
                bundle_cache: None,
                meshing_host,
            },
        ))
        .await
}

pub struct RemoteWorkerRelayRequest<'a> {
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub run_identity: String,
    pub worker: WorkerSpec,
    pub driver_fence: u64,
    pub session_id: [u8; 16],
    pub run_key: RunKeyMaterial,
    pub limits: FrameLimits,
}

pub async fn run_remote_worker_relay(
    request: RemoteWorkerRelayRequest<'_>,
) -> NativeExecutionResult<()> {
    run_remote_worker_relay_cached(request, None).await
}

pub(crate) async fn run_remote_worker_relay_cached(
    request: RemoteWorkerRelayRequest<'_>,
    bundle_cache: Option<std::path::PathBuf>,
) -> NativeExecutionResult<()> {
    let RemoteWorkerRelayRequest {
        url,
        headers,
        run_identity,
        worker,
        driver_fence,
        session_id,
        run_key,
        limits,
    } = request;
    let connection = WebSocketRelayConnection::connect(url, headers, limits)
        .await
        .map_err(protocol)?;
    tokio::task::LocalSet::new()
        .run_until(run_worker_loop(
            Arc::new(RelayFrameRoute::new(connection)),
            WorkerLoopContext {
                run_identity,
                worker,
                driver_fence,
                session_id,
                run_key,
                limits,
                bundle_cache,
                meshing_host: None,
            },
        ))
        .await
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
