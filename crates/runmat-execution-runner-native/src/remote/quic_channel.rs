use std::collections::HashMap;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use runmat_execution::Digest;
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_runner::{AttemptReport, AttemptRequest, WorkerSpec};
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameKind, FrameLimits};
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayConnection};
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use uuid::Uuid;

use super::protocol::{
    RemoteWorkerCommand, RemoteWorkerOutcome, RemoteWorkerReply, RemoteWorkerRequest,
    REMOTE_WORKER_PROTOCOL_V1,
};
use super::route::{QuicFrameRoute, RemoteFrameRoute};
use super::{RemoteAttempt, RemoteBundleReceipt, RemoteValueReceipt, RemoteWorkerChannel};
use crate::{NativeExecutionError, NativeExecutionResult};

const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);

pub struct QuicRemoteWorkerChannel {
    node_identity: String,
    worker: WorkerSpec,
    driver_fence: u64,
    limits: FrameLimits,
    route: Arc<dyn RemoteFrameRoute>,
    sender: AsyncMutex<EncryptedFrameSession>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<RemoteWorkerReply>>>>,
}

pub struct RemoteWorkerChannelConfig {
    pub run_identity: String,
    pub node_identity: String,
    pub worker: WorkerSpec,
    pub driver_fence: u64,
    pub session_id: [u8; 16],
    pub run_key: RunKeyMaterial,
    pub limits: FrameLimits,
}

impl QuicRemoteWorkerChannel {
    pub async fn connect(
        config: RemoteWorkerChannelConfig,
        endpoint: &PinnedQuicEndpoint,
    ) -> NativeExecutionResult<Arc<Self>> {
        let bind = SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0);
        let connection = Arc::new(
            QuicOverlayConnection::connect(bind, endpoint, config.limits)
                .await
                .map_err(protocol)?,
        );
        Self::connect_route(config, Arc::new(QuicFrameRoute(connection))).await
    }

    pub(crate) async fn connect_route(
        config: RemoteWorkerChannelConfig,
        route: Arc<dyn RemoteFrameRoute>,
    ) -> NativeExecutionResult<Arc<Self>> {
        let RemoteWorkerChannelConfig {
            run_identity,
            node_identity,
            worker,
            driver_fence,
            session_id,
            run_key,
            limits,
        } = config;
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let channel = Arc::new(Self {
            node_identity,
            worker,
            driver_fence,
            limits,
            route: Arc::clone(&route),
            sender: AsyncMutex::new(
                EncryptedFrameSession::new(
                    run_identity.clone(),
                    session_id,
                    "driver-to-worker",
                    1,
                    run_key.clone(),
                )
                .map_err(protocol)?,
            ),
            pending: Arc::clone(&pending),
        });
        let mut receiver =
            EncryptedFrameSession::new(run_identity, session_id, "worker-to-driver", 1, run_key)
                .map_err(protocol)?;
        tokio::spawn(async move {
            loop {
                let frame = match route.receive().await {
                    Ok(frame) => frame,
                    Err(_) => break,
                };
                let plaintext = match receiver.open(&frame, limits) {
                    Ok(plaintext) => plaintext,
                    Err(_) => break,
                };
                let reply: RemoteWorkerReply =
                    match serde_json::from_slice::<RemoteWorkerReply>(&plaintext) {
                        Ok(reply) if reply.schema_version == REMOTE_WORKER_PROTOCOL_V1 => reply,
                        _ => break,
                    };
                if let Some(sender) = pending
                    .lock()
                    .expect("remote reply registry poisoned")
                    .remove(&reply.correlation_id)
                {
                    let _ = sender.send(reply);
                }
            }
            pending
                .lock()
                .expect("remote reply registry poisoned")
                .clear();
        });
        Ok(channel)
    }

    async fn request(
        &self,
        command: RemoteWorkerCommand,
    ) -> NativeExecutionResult<RemoteWorkerOutcome> {
        let correlation_id = Uuid::new_v4().to_string();
        let request = RemoteWorkerRequest {
            schema_version: REMOTE_WORKER_PROTOCOL_V1,
            correlation_id: correlation_id.clone(),
            driver_fence: self.driver_fence,
            command,
        };
        let (sender, receiver) = oneshot::channel();
        self.pending
            .lock()
            .expect("remote reply registry poisoned")
            .insert(correlation_id.clone(), sender);
        let plaintext = serde_json::to_vec(&request).map_err(protocol)?;
        let frame = self
            .sender
            .lock()
            .await
            .seal(FrameKind::Control, &plaintext, self.limits)
            .map_err(protocol)?;
        if let Err(error) = self.route.send(frame).await {
            self.pending
                .lock()
                .expect("remote reply registry poisoned")
                .remove(&correlation_id);
            return Err(protocol(error));
        }
        let reply = tokio::time::timeout(RESPONSE_TIMEOUT, receiver)
            .await
            .map_err(|_| protocol("remote worker response timed out"))?
            .map_err(|_| protocol("remote worker route closed"))?;
        match reply.outcome {
            RemoteWorkerOutcome::Rejected { message } => {
                Err(NativeExecutionError::Protocol(message))
            }
            outcome => Ok(outcome),
        }
    }
}

#[async_trait]
impl RemoteWorkerChannel for QuicRemoteWorkerChannel {
    fn node_identity(&self) -> &str {
        &self.node_identity
    }

    fn worker(&self) -> &WorkerSpec {
        &self.worker
    }

    async fn install_bundle(
        &self,
        bundle_digest: Digest,
        bundle: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        match self
            .request(RemoteWorkerCommand::InstallBundle {
                bundle_digest,
                bundle: bundle.to_vec(),
            })
            .await?
        {
            RemoteWorkerOutcome::BundleStored { receipt } => Ok(receipt),
            _ => Err(protocol("remote worker returned the wrong bundle reply")),
        }
    }

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        match self
            .request(RemoteWorkerCommand::Execute {
                attempt: Box::new(attempt),
            })
            .await?
        {
            RemoteWorkerOutcome::Attempt { report } => Ok(report),
            _ => Err(protocol("remote worker returned the wrong attempt reply")),
        }
    }

    async fn activate_bundle(
        &self,
        bundle_digest: Digest,
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        match self
            .request(RemoteWorkerCommand::ActivateBundle { bundle_digest })
            .await?
        {
            RemoteWorkerOutcome::BundleStored { receipt } => Ok(receipt),
            _ => Err(protocol("remote worker returned the wrong bundle reply")),
        }
    }

    async fn transfer_value(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<RemoteValueReceipt> {
        match self
            .request(RemoteWorkerCommand::PutValue {
                reference,
                encoded: encoded.to_vec(),
            })
            .await?
        {
            RemoteWorkerOutcome::ValueStored { receipt } => Ok(receipt),
            _ => Err(protocol("remote worker returned the wrong value reply")),
        }
    }

    async fn cancel(&self, request: &AttemptRequest) -> NativeExecutionResult<()> {
        self.ack(RemoteWorkerCommand::Cancel {
            attempt_id: request.id,
        })
        .await
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        self.ack(RemoteWorkerCommand::Drain).await
    }
}

impl QuicRemoteWorkerChannel {
    async fn ack(&self, command: RemoteWorkerCommand) -> NativeExecutionResult<()> {
        match self.request(command).await? {
            RemoteWorkerOutcome::Acknowledged => Ok(()),
            _ => Err(protocol("remote worker returned the wrong acknowledgement")),
        }
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
