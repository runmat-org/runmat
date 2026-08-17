use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use runmat_execution::Digest;
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_runner::{AttemptReport, AttemptRequest, WorkerSpec};
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameKind, FrameLimits};
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayConnection};
use runmat_execution_transport_native::transfer::ObjectChunk;
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use uuid::Uuid;

use super::protocol::{
    RemoteWorkerCommand, RemoteWorkerOutcome, RemoteWorkerReply, RemoteWorkerRequest,
    REMOTE_WORKER_PROTOCOL_V3,
};
use super::route::{QuicFrameRoute, RemoteFrameRoute};
use super::{
    RemoteAttempt, RemoteBundleReceipt, RemoteObjectReceipt, RemoteValueReceipt,
    RemoteWorkerChannel,
};
use crate::{NativeExecutionError, NativeExecutionResult};

const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_OBJECT_CHUNK_BYTES: usize = 256 * 1024;
const MAX_BUFFERED_PROGRESS: usize = 256;
struct PendingReply {
    frame_kind: FrameKind,
    attempt_id: Option<runmat_execution::identity::AttemptId>,
    sender: oneshot::Sender<RemoteWorkerReply>,
}

pub struct QuicRemoteWorkerChannel {
    node_identity: String,
    worker: WorkerSpec,
    driver_fence: u64,
    limits: FrameLimits,
    route: Arc<dyn RemoteFrameRoute>,
    sender: AsyncMutex<EncryptedFrameSession>,
    pending: Arc<Mutex<HashMap<String, PendingReply>>>,
    progress: Arc<
        Mutex<HashMap<runmat_execution::identity::AttemptId, VecDeque<crate::ProgramProgress>>>,
    >,
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
        let progress = Arc::new(Mutex::new(HashMap::<_, VecDeque<_>>::new()));
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
            progress: Arc::clone(&progress),
        });
        let mut receiver =
            EncryptedFrameSession::new(run_identity, session_id, "worker-to-driver", 1, run_key)
                .map_err(protocol)?;
        tokio::spawn(async move {
            let mut progress_sequences = HashMap::new();
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
                        Ok(reply) if reply.schema_version == REMOTE_WORKER_PROTOCOL_V3 => reply,
                        _ => break,
                    };
                if let RemoteWorkerOutcome::Progress {
                    attempt_id,
                    progress: event,
                } = &reply.outcome
                {
                    if event.validate().is_err()
                        || progress_sequences
                            .insert(*attempt_id, event.sequence)
                            .is_some_and(|previous| event.sequence <= previous)
                    {
                        break;
                    }
                    let mut queues = progress.lock().expect("remote progress registry poisoned");
                    let queue = queues.entry(*attempt_id).or_default();
                    if queue.len() == MAX_BUFFERED_PROGRESS {
                        queue.pop_front();
                    }
                    queue.push_back(event.clone());
                    continue;
                }
                if let Some(pending) = pending
                    .lock()
                    .expect("remote reply registry poisoned")
                    .remove(&reply.correlation_id)
                {
                    if frame.kind != pending.frame_kind {
                        break;
                    }
                    if let Some(attempt_id) = pending.attempt_id {
                        progress_sequences.remove(&attempt_id);
                    }
                    let _ = pending.sender.send(reply);
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
        kind: FrameKind,
        command: RemoteWorkerCommand,
    ) -> NativeExecutionResult<RemoteWorkerOutcome> {
        let correlation_id = Uuid::new_v4().to_string();
        let attempt_id = match &command {
            RemoteWorkerCommand::Execute { attempt } => Some(attempt.scheduling.id),
            _ => None,
        };
        let request = RemoteWorkerRequest {
            schema_version: REMOTE_WORKER_PROTOCOL_V3,
            correlation_id: correlation_id.clone(),
            driver_fence: self.driver_fence,
            command,
        };
        let (sender, receiver) = oneshot::channel();
        self.pending
            .lock()
            .expect("remote reply registry poisoned")
            .insert(
                correlation_id.clone(),
                PendingReply {
                    frame_kind: kind,
                    attempt_id,
                    sender,
                },
            );
        let plaintext = serde_json::to_vec(&request).map_err(protocol)?;
        let frame = self
            .sender
            .lock()
            .await
            .seal(kind, &plaintext, self.limits)
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
            .request(
                FrameKind::Control,
                RemoteWorkerCommand::InstallBundle {
                    bundle_digest,
                    bundle: bundle.to_vec(),
                },
            )
            .await?
        {
            RemoteWorkerOutcome::BundleStored { receipt } => Ok(receipt),
            _ => Err(protocol("remote worker returned the wrong bundle reply")),
        }
    }

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        match self
            .request(
                FrameKind::Control,
                RemoteWorkerCommand::Execute {
                    attempt: Box::new(attempt),
                },
            )
            .await?
        {
            RemoteWorkerOutcome::Attempt { report } => Ok(report),
            _ => Err(protocol("remote worker returned the wrong attempt reply")),
        }
    }

    fn drain_progress(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
    ) -> Vec<crate::ProgramProgress> {
        self.progress
            .lock()
            .expect("remote progress registry poisoned")
            .remove(&attempt_id)
            .map(VecDeque::into_iter)
            .map(Iterator::collect)
            .unwrap_or_default()
    }

    async fn activate_bundle(
        &self,
        bundle_digest: Digest,
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        match self
            .request(
                FrameKind::Control,
                RemoteWorkerCommand::ActivateBundle { bundle_digest },
            )
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
            .request(
                FrameKind::Control,
                RemoteWorkerCommand::PutValue {
                    reference,
                    encoded: encoded.to_vec(),
                },
            )
            .await?
        {
            RemoteWorkerOutcome::ValueStored { receipt } => Ok(receipt),
            _ => Err(protocol("remote worker returned the wrong value reply")),
        }
    }

    async fn transfer_object(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<RemoteObjectReceipt> {
        validate_object(&reference, encoded)?;
        let mut receipt = match self
            .request(
                FrameKind::Artifact,
                RemoteWorkerCommand::ProbeObject {
                    reference: reference.clone(),
                },
            )
            .await?
        {
            RemoteWorkerOutcome::ObjectPosition { receipt } => receipt,
            _ => {
                return Err(protocol(
                    "remote worker returned the wrong object probe reply",
                ))
            }
        };
        if receipt.value_id != reference.id || receipt.next_offset > reference.encoded_length {
            return Err(protocol(
                "remote worker returned an invalid object position",
            ));
        }
        while !receipt.complete {
            let offset = usize::try_from(receipt.next_offset)
                .map_err(|_| protocol("remote object offset does not fit this host"))?;
            let end = offset
                .saturating_add(MAX_OBJECT_CHUNK_BYTES)
                .min(encoded.len());
            if end == offset {
                return Err(protocol("remote worker object transfer made no progress"));
            }
            receipt = match self
                .request(
                    FrameKind::Artifact,
                    RemoteWorkerCommand::PutObjectChunk {
                        reference: reference.clone(),
                        chunk: ObjectChunk {
                            offset: receipt.next_offset,
                            bytes: encoded[offset..end].to_vec(),
                        },
                    },
                )
                .await?
            {
                RemoteWorkerOutcome::ObjectPosition { receipt } => receipt,
                _ => {
                    return Err(protocol(
                        "remote worker returned the wrong object chunk reply",
                    ))
                }
            };
            if receipt.value_id != reference.id || receipt.next_offset != end as u64 {
                return Err(protocol(
                    "remote worker acknowledged a different object chunk",
                ));
            }
        }
        Ok(receipt)
    }

    async fn download_object(
        &self,
        reference: runmat_execution::value::ValueRef,
    ) -> NativeExecutionResult<Vec<u8>> {
        let capacity = usize::try_from(reference.encoded_length)
            .map_err(|_| protocol("remote object length does not fit this host"))?;
        let mut encoded = Vec::with_capacity(capacity);
        loop {
            let (chunk, complete) = match self
                .request(
                    FrameKind::Artifact,
                    RemoteWorkerCommand::GetObjectChunk {
                        reference: reference.clone(),
                        offset: encoded.len() as u64,
                        maximum_bytes: MAX_OBJECT_CHUNK_BYTES as u32,
                    },
                )
                .await?
            {
                RemoteWorkerOutcome::ObjectChunk { chunk, complete } => (chunk, complete),
                _ => {
                    return Err(protocol(
                        "remote worker returned the wrong object download reply",
                    ))
                }
            };
            if chunk.offset != encoded.len() as u64
                || chunk.bytes.is_empty() && !complete
                || chunk.bytes.len() > MAX_OBJECT_CHUNK_BYTES
            {
                return Err(protocol(
                    "remote worker returned a non-contiguous object chunk",
                ));
            }
            encoded.extend_from_slice(&chunk.bytes);
            if encoded.len() > capacity {
                return Err(protocol(
                    "remote worker exceeded the declared object length",
                ));
            }
            if complete {
                validate_object(&reference, &encoded)?;
                return Ok(encoded);
            }
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
        match self.request(FrameKind::Control, command).await? {
            RemoteWorkerOutcome::Acknowledged => Ok(()),
            _ => Err(protocol("remote worker returned the wrong acknowledgement")),
        }
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

fn validate_object(
    reference: &runmat_execution::value::ValueRef,
    encoded: &[u8],
) -> NativeExecutionResult<()> {
    use runmat_execution::value::{ValueLimits, ValuePayload};

    ValuePayload::Object(Box::new(reference.clone()))
        .validate(ValueLimits::default())
        .map_err(protocol)?;
    if encoded.len() as u64 != reference.encoded_length
        || runmat_execution::Digest::sha256(encoded) != reference.logical_digest
    {
        return Err(protocol("remote object bytes differ from their reference"));
    }
    Ok(())
}
