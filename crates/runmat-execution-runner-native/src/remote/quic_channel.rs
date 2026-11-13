use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use runmat_execution::resource::Capability;
use runmat_execution::Digest;
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_runner::{AttemptReport, AttemptRequest, WorkerSpec};
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameKind, FrameLimits};
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayConnection};
use runmat_execution_transport_native::transfer::ObjectChunk;
use tokio::sync::{oneshot, watch, Mutex as AsyncMutex};
use uuid::Uuid;

use super::protocol::{
    RemoteWorkerCommand, RemoteWorkerOutcome, RemoteWorkerReply, RemoteWorkerRequest,
    REMOTE_WORKER_PROTOCOL_VERSION,
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
mod receive;
mod replies;
use replies::{PendingReply, ReplyRegistry};

pub struct QuicRemoteWorkerChannel {
    run_identity: String,
    session_id: [u8; 16],
    node_identity: String,
    worker: WorkerSpec,
    driver_fence: u64,
    limits: FrameLimits,
    route: Arc<dyn RemoteFrameRoute>,
    sender: AsyncMutex<EncryptedFrameSession>,
    pending: Arc<Mutex<ReplyRegistry>>,
    progress: Arc<
        Mutex<HashMap<runmat_execution::identity::AttemptId, VecDeque<crate::ProgramProgress>>>,
    >,
    collectives: Arc<
        Mutex<
            HashMap<
                runmat_execution::identity::AttemptId,
                VecDeque<runmat_execution::CollectiveRequest>,
            >,
        >,
    >,
    fea_calls: Arc<
        Mutex<HashMap<runmat_execution::identity::AttemptId, VecDeque<crate::fea_proxy::Call>>>,
    >,
    verified_ready: Arc<Mutex<Option<std::collections::BTreeSet<Capability>>>>,
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
            mut worker,
            driver_fence,
            session_id,
            run_key,
            limits,
        } = config;
        // The allocation supplies capacity and host placement, not software
        // support. Only the encrypted reply from the running binary may add it.
        worker
            .resources
            .capabilities
            .remove(&Capability::GraphSelectedValuesV11);
        worker
            .resources
            .capabilities
            .remove(&Capability::GraphCallableArgumentsV12);
        worker
            .resources
            .capabilities
            .remove(&Capability::GraphProjectRuntimePolicyV13);
        worker
            .resources
            .capabilities
            .remove(&Capability::GraphInvocationControlsV14);
        let expected_worker = worker.id;
        let expected_pool = worker.pool_id;
        let expected_run = run_identity.clone();
        let pending = Arc::new(Mutex::new(ReplyRegistry::default()));
        let progress = Arc::new(Mutex::new(HashMap::<_, VecDeque<_>>::new()));
        let collectives = Arc::new(Mutex::new(HashMap::<_, VecDeque<_>>::new()));
        let fea_calls = Arc::new(Mutex::new(HashMap::<_, VecDeque<_>>::new()));
        let verified_ready = Arc::new(Mutex::new(None));
        let ready_state = Arc::clone(&verified_ready);
        let channel = Arc::new(Self {
            run_identity: run_identity.clone(),
            session_id,
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
            collectives: Arc::clone(&collectives),
            fea_calls: Arc::clone(&fea_calls),
            verified_ready,
        });
        let mut receiver =
            EncryptedFrameSession::new(run_identity, session_id, "worker-to-driver", 1, run_key)
                .map_err(protocol)?;
        tokio::spawn(async move {
            let ready = receive::ReadyExpectation {
                run_identity: expected_run,
                session_id,
                driver_fence,
                worker_id: expected_worker,
                pool_id: expected_pool,
                capabilities: ready_state,
            };
            let registries = receive::Registries {
                pending: &pending,
                progress: &progress,
                collectives: &collectives,
                fea_calls: &fea_calls,
            };
            if let Err(error) =
                receive::run(route.as_ref(), &mut receiver, limits, &ready, registries).await
            {
                pending
                    .lock()
                    .expect("remote reply registry poisoned")
                    .fail(error);
            }
            collectives
                .lock()
                .expect("remote collective registry poisoned")
                .clear();
            fea_calls
                .lock()
                .expect("remote FEA request registry poisoned")
                .clear();
        });
        // Register V10 immediately. The existing pool reconciliation loop
        // observes authenticated ReadyV1 later without delaying V10 startup.
        Ok(channel)
    }

    async fn request(
        &self,
        kind: FrameKind,
        command: RemoteWorkerCommand,
    ) -> NativeExecutionResult<RemoteWorkerOutcome> {
        let correlation_id = Uuid::new_v4().to_string();
        let attempt_id = match &command {
            RemoteWorkerCommand::Execute { attempt, .. } => Some(attempt.scheduling.id),
            _ => None,
        };
        let request = RemoteWorkerRequest {
            schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
            correlation_id: correlation_id.clone(),
            driver_fence: self.driver_fence,
            command,
        };
        let plaintext = serde_json::to_vec(&request).map_err(protocol)?;
        let frame = self
            .sender
            .lock()
            .await
            .seal(kind, &plaintext, self.limits)
            .map_err(protocol)?;
        let (sender, receiver) = oneshot::channel();
        // Activity is a coalesced timestamp, not an unbounded queue of notifications.
        let (activity, mut activity_receiver) = watch::channel(tokio::time::Instant::now());
        self.pending
            .lock()
            .expect("remote reply registry poisoned")
            .register(
                correlation_id.clone(),
                PendingReply {
                    frame_kind: kind,
                    attempt_id,
                    sender,
                    activity,
                },
            )
            .map_err(|source| NativeExecutionError::RemoteChannel { source })?;
        if let Err(error) = self.route.send(frame).await {
            let source = self
                .pending
                .lock()
                .expect("remote reply registry poisoned")
                .fail(error);
            return Err(NativeExecutionError::RemoteChannel { source });
        }
        tokio::pin!(receiver);
        let mut activity_open = true;
        let mut inactivity_deadline = *activity_receiver.borrow_and_update() + RESPONSE_TIMEOUT;
        let reply = loop {
            tokio::select! {
                reply = &mut receiver => {
                    break reply.map_err(|_| protocol("remote worker reply dispatcher stopped without a terminal cause"))?
                        .map_err(|source| NativeExecutionError::RemoteChannel { source })?;
                }
                activity = activity_receiver.changed(), if activity_open => {
                    if activity.is_ok() {
                        inactivity_deadline = *activity_receiver.borrow_and_update() + RESPONSE_TIMEOUT;
                    } else {
                        activity_open = false;
                    }
                }
                _ = tokio::time::sleep_until(inactivity_deadline) => {
                    let mut pending = self.pending.lock().expect("remote reply registry poisoned");
                    let refreshed = *activity_receiver.borrow_and_update() + RESPONSE_TIMEOUT;
                    if refreshed > tokio::time::Instant::now() {
                        inactivity_deadline = refreshed;
                        continue;
                    }
                    pending.remove(&correlation_id);
                    return Err(protocol("remote worker response timed out after inactivity"));
                }
            }
        };
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

    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec> {
        match self
            .request(FrameKind::Control, RemoteWorkerCommand::Describe)
            .await?
        {
            RemoteWorkerOutcome::WorkerDescription { worker } => Ok(worker),
            _ => Err(protocol(
                "remote worker returned the wrong description reply",
            )),
        }
    }

    fn verified_software_capabilities(&self) -> Option<std::collections::BTreeSet<Capability>> {
        self.verified_ready
            .lock()
            .expect("worker readiness poisoned")
            .clone()
    }

    async fn authenticated_physical_retirement(
        &self,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<Option<super::AuthenticatedPhysicalRetirement>> {
        if request.worker_id != self.worker.id
            || request.task.pool_id != self.worker.pool_id
            || request.driver_fence != self.driver_fence
        {
            return Err(protocol(
                "retirement query differs from authenticated worker route",
            ));
        }
        let expected_digest = Digest::sha256(&serde_json::to_vec(request).map_err(protocol)?);
        match self
            .request(
                FrameKind::Control,
                RemoteWorkerCommand::QueryPhysicalRetirement {
                    attempt: Box::new(request.clone()),
                },
            )
            .await?
        {
            RemoteWorkerOutcome::PhysicalRetirement {
                run_identity,
                session_id,
                driver_fence,
                worker_id,
                pool_id,
                attempt_id,
                request_digest,
                usage,
            } if run_identity == self.run_identity
                && session_id == self.session_id
                && driver_fence == self.driver_fence
                && worker_id == self.worker.id
                && pool_id == self.worker.pool_id
                && attempt_id == request.id
                && request_digest == expected_digest =>
            {
                Ok(Some(
                    super::AuthenticatedPhysicalRetirement::issue_with_usage(
                        request.clone(),
                        usage,
                    )?,
                ))
            }
            RemoteWorkerOutcome::RetirementUnavailable => Ok(None),
            _ => Err(protocol(
                "remote retirement evidence differs from its exact authenticated owner",
            )),
        }
    }

    async fn acknowledge_physical_retirement(
        &self,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<()> {
        let command = RemoteWorkerCommand::AcknowledgePhysicalRetirement {
            attempt: Box::new(request.clone()),
        };
        match self.ack(command.clone()).await {
            Ok(()) => Ok(()),
            // The first ACK may have been applied while its reply was lost.
            // The worker's exact bounded replay record makes this safe.
            Err(_) => self.ack(command).await,
        }
    }

    async fn prune_durable_checkpoint(
        &self,
        sequence: u64,
        attempts: &[super::CheckpointedAttempt],
    ) -> NativeExecutionResult<()> {
        if sequence == 0 || attempts.len() > runmat_execution::security::MAX_RETIREMENT_ATTEMPTS {
            return Err(protocol(
                "durable checkpoint prune exceeds worker journal bound",
            ));
        }
        let command = RemoteWorkerCommand::PruneDurableCheckpoint {
            sequence,
            attempts: attempts.to_vec(),
        };
        match self.ack(command.clone()).await {
            Ok(()) => Ok(()),
            Err(_) => self.ack(command).await,
        }
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

    async fn stage_graph_program(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
        encrypted_manifest: &[u8],
        downloads: &[runmat_execution_transport_native::control::DriverArtifactDownload],
        maximum_program_bytes: u64,
    ) -> NativeExecutionResult<super::RemoteGraphProgramReceipt> {
        let (program_digest, program_length, chunk_count) = match self
            .request(
                FrameKind::Control,
                RemoteWorkerCommand::BeginGraphProgram {
                    attempt_id,
                    encrypted_manifest: encrypted_manifest.to_vec(),
                    maximum_program_bytes,
                },
            )
            .await?
        {
            RemoteWorkerOutcome::GraphProgramPrepared {
                program_digest,
                program_length,
                chunk_count,
            } => (program_digest, program_length, chunk_count),
            RemoteWorkerOutcome::GraphProgramStaged { receipt } => return Ok(receipt),
            _ => {
                return Err(protocol(
                    "remote worker returned the wrong Graph preparation reply",
                ))
            }
        };
        if downloads.len() != chunk_count as usize {
            return Err(protocol(
                "Graph Program signed chunk count differs from worker manifest",
            ));
        }
        for (ordinal, download) in downloads.iter().enumerate() {
            if download.kind != "program-chunk" {
                return Err(protocol(
                    "Graph Program transfer contains a nonchunk descriptor",
                ));
            }
            let ordinal = u32::try_from(ordinal)
                .map_err(|_| protocol("Graph Program has too many chunks"))?;
            match self
                .request(
                    FrameKind::Control,
                    RemoteWorkerCommand::StageGraphProgramChunk {
                        attempt_id,
                        program_digest,
                        ordinal,
                        download: download.clone(),
                    },
                )
                .await?
            {
                RemoteWorkerOutcome::Acknowledged => {}
                _ => {
                    return Err(protocol(
                        "remote worker returned the wrong Graph chunk reply",
                    ))
                }
            }
        }
        match self
            .request(
                FrameKind::Control,
                RemoteWorkerCommand::FinishGraphProgram {
                    attempt_id,
                    program_digest,
                },
            )
            .await?
        {
            RemoteWorkerOutcome::GraphProgramStaged { receipt }
                if receipt.program_digest == program_digest
                    && receipt.program_length == program_length =>
            {
                Ok(receipt)
            }
            _ => Err(protocol(
                "remote worker returned the wrong Graph Program receipt",
            )),
        }
    }

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        Ok(self.execute_observed(attempt, None).await?.report)
    }

    async fn execute_observed(
        &self,
        attempt: RemoteAttempt,
        console_budget: Option<runmat_execution::console::ConsoleBudget>,
    ) -> NativeExecutionResult<runmat_execution_runner::port::BackendReport> {
        if attempt.scheduling.task.console.is_some() != console_budget.is_some() {
            return Err(protocol(
                "remote console invocation omits its bounded subtree grant",
            ));
        }
        let deadline = tokio::time::Instant::now() + RESPONSE_TIMEOUT;
        loop {
            match self
                .request(
                    FrameKind::Control,
                    RemoteWorkerCommand::Execute {
                        attempt: Box::new(attempt.clone()),
                        console_budget,
                    },
                )
                .await?
            {
                RemoteWorkerOutcome::Attempt {
                    report,
                    console,
                    console_scopes,
                    delegated_console_root,
                    console_budget: returned_budget,
                } => {
                    if returned_budget != console_budget {
                        return Err(protocol(
                            "worker changed its original console subtree grant",
                        ));
                    }
                    if let Some(console) = &console {
                        console.validate().map_err(protocol)?;
                        if Some(console.limits()) != attempt.scheduling.task.console {
                            return Err(protocol(
                                "remote console differs from its admitted invocation controls",
                            ));
                        }
                    }
                    runmat_execution::console::validate_scopes(&console_scopes)
                        .map_err(protocol)?;
                    if delegated_console_root.is_some()
                        && !matches!(
                            attempt.program,
                            super::channel::RemoteAttemptProgram::GraphStaged(_)
                        )
                    {
                        return Err(protocol("Direct worker cannot delegate its console root"));
                    }
                    if delegated_console_root.is_some()
                        && matches!(&report, AttemptReport::Succeeded { result } if !matches!(result, runmat_execution_runner::AttemptSuccess::Graph { .. }))
                    {
                        return Err(protocol(
                            "delegated console root requires the typed Graph result",
                        ));
                    }
                    runmat_execution::console::validate_delegated_root(
                        console.as_deref(),
                        &console_scopes,
                        delegated_console_root,
                        matches!(report, AttemptReport::Succeeded { .. }),
                    )
                    .map_err(protocol)?;
                    runmat_execution::console::validate_source_controls(
                        console.as_deref(),
                        &console_scopes,
                        delegated_console_root,
                        attempt.scheduling.task.console,
                        attempt.scheduling.id,
                    )
                    .map_err(protocol)?;
                    if matches!(report, AttemptReport::Succeeded { .. })
                        && attempt.scheduling.task.console.is_some()
                        && delegated_console_root.is_none()
                        && console.as_ref().is_none_or(|capture| {
                            capture.completion()
                                != runmat_execution::console::ConsoleCompletion::Complete
                        })
                    {
                        return Err(protocol(
                            "remote success lacks complete requested console evidence",
                        ));
                    }
                    let mut observed = runmat_execution_runner::port::BackendReport::for_request(
                        &attempt.scheduling,
                        report,
                    );
                    observed.console = console;
                    observed.console_scopes = console_scopes;
                    observed.delegated_console_root = delegated_console_root;
                    return Ok(observed);
                }
                RemoteWorkerOutcome::AdmissionDeferred {
                    attempt_id,
                    worker_id,
                    driver_fence,
                } if attempt_id == attempt.scheduling.id
                    && worker_id == self.worker.id
                    && driver_fence == self.driver_fence =>
                {
                    if tokio::time::Instant::now() >= deadline {
                        return Err(protocol(
                            "durable attempt journal remained full after checkpoint backpressure",
                        ));
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                _ => return Err(protocol("remote worker returned the wrong attempt reply")),
            }
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

    fn drain_collective_requests(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
    ) -> Vec<runmat_execution::CollectiveRequest> {
        self.collectives
            .lock()
            .expect("remote collective registry poisoned")
            .remove(&attempt_id)
            .map(VecDeque::into_iter)
            .map(Iterator::collect)
            .unwrap_or_default()
    }

    fn drain_fea_calls(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
    ) -> Vec<crate::fea_proxy::Call> {
        self.fea_calls
            .lock()
            .expect("remote FEA request registry poisoned")
            .remove(&attempt_id)
            .map(VecDeque::into_iter)
            .map(Iterator::collect)
            .unwrap_or_default()
    }

    async fn complete_fea_call(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
        sequence: u64,
        reply: crate::fea_proxy::Reply,
    ) -> NativeExecutionResult<()> {
        self.ack(RemoteWorkerCommand::CompleteFeaAdmission {
            attempt_id,
            sequence,
            reply,
        })
        .await
    }

    async fn complete_collective(
        &self,
        attempt_id: runmat_execution::identity::AttemptId,
        request: &runmat_execution::CollectiveRequest,
        result: crate::protocol::CollectiveProcessResult,
    ) -> NativeExecutionResult<()> {
        self.ack(RemoteWorkerCommand::CompleteCollective {
            attempt_id,
            context: request.context.clone(),
            id: request.id,
            sequence: request.sequence,
            result,
        })
        .await
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

    async fn cancel(
        &self,
        request: &AttemptRequest,
        reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()> {
        self.ack(RemoteWorkerCommand::Cancel {
            attempt_id: request.id,
            reason,
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

#[cfg(test)]
mod tests;
