use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use runmat_execution::identity::AttemptId;
use runmat_execution::resource::ResourceRequest;
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_artifact::{
    ExecutableForm, ExecutionBundle, ProgramExecutionResponse, WorkerProgramRequest,
};
use runmat_execution_runner::scheduler::{fits, release, reserve, ResourceAllocation};
use runmat_execution_runner::{AttemptRequest, WorkerSpec};
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameKind, FrameLimits};
use tokio::sync::Mutex;

use super::protocol::{
    RemoteWorkerCommand, RemoteWorkerOutcome, RemoteWorkerReply, RemoteWorkerRequest, WorkerReady,
    REMOTE_WORKER_PROTOCOL_VERSION,
};
use super::route::RemoteFrameRoute;
use super::worker_attempt_journal::{WorkerAttemptJournal, WorkerAttemptJournalOwner};
use super::worker_protocol::{
    acknowledged, command_frame_kind, rejected, reply, reply_kind, reply_progress,
};
use super::{RemoteAttempt, RemoteAttemptProgram, RemoteStagedGraphReference};
use crate::{NativeExecutionError, NativeExecutionResult};

const MAX_PREEXECUTION_CANCELLATIONS: usize = 1 << 20;
const GRAPH_STAGE_IDLE_RETENTION: Duration = Duration::from_secs(60);
// Fixed protocol-control storage owned by the authenticated worker session:
// all slots are allocated at startup and the journal never grows beyond them.
// Pending receipts also retain their original per-attempt resource grant.
// Exact ACK drops that variable metadata and releases the grant once; compact
// ACK replay markers use only these preallocated session slots and may evict.
const MAX_RETIREMENT_RECEIPTS: usize = 256;

struct WorkerRetirementReceipt {
    attempt_id: AttemptId,
    request_digest: runmat_execution::Digest,
    usage: Option<super::protocol::RetiredAttemptUsage>,
    // None is an exact ACK replay record in the fixed worker-session journal.
    // The charged reservation is released exactly once when this is taken.
    claim: Option<(
        ResourceRequest,
        runmat_execution::resource::ResourceAssignment,
    )>,
}

struct WorkerState {
    bundle: Option<Arc<ExecutionBundle>>,
    materialized_project: Option<Arc<crate::materialized_project::MaterializedProject>>,
    bundle_digest: Option<runmat_execution::Digest>,
    attempts: HashMap<AttemptId, ActiveAttempt>,
    /// Session-owned, preallocated protocol-control journal. This is separate
    /// from `allocated`, which charges every pending physical reservation.
    retirement_receipts: VecDeque<WorkerRetirementReceipt>,
    /// Capacity reserved by active attempts, not sampled process utilization.
    allocated: ResourceAllocation,
    cancelled_attempts: HashMap<AttemptId, runmat_execution::CancellationReason>,
    values: HashMap<runmat_execution::identity::ValueId, runmat_execution::value::ValuePayload>,
    objects: super::object_transfer::RemoteObjectStore,
    draining: bool,
    bundle_cache: Option<std::path::PathBuf>,
    graph_staging: HashMap<AttemptId, GraphStageSlot>,
    graph_stage_download_in_flight:
        HashMap<AttemptId, runmat_execution::cancellation::CancellationSource>,
    graph_programs: HashMap<AttemptId, Arc<super::program_staging::StagedGraphProgram>>,
    // This state is created for exactly one authenticated run/key epoch in
    // run_worker_loop. A digest is only a lookup within that sealed scope;
    // attempt leases and idle expiry own the underlying private bytes.
    graph_cache: HashMap<runmat_execution::Digest, CachedGraphProgram>,
}

struct CachedGraphProgram {
    program: Arc<super::program_staging::StagedGraphProgram>,
    last_used: tokio::time::Instant,
}

fn retire_idle_graph_cache(
    state: &mut WorkerState,
    now: tokio::time::Instant,
    expired_only: bool,
) -> Vec<Arc<super::program_staging::StagedGraphProgram>> {
    let mut retired = Vec::new();
    let mut retained = HashMap::with_capacity(state.graph_cache.len());
    for (digest, cached) in std::mem::take(&mut state.graph_cache) {
        let idle = Arc::strong_count(&cached.program) == 1;
        let expired = now.duration_since(cached.last_used) >= GRAPH_STAGE_IDLE_RETENTION;
        if idle && (!expired_only || expired) {
            retired.push(cached.program);
        } else {
            retained.insert(digest, cached);
        }
    }
    state.graph_cache = retained;
    retired
}

fn graph_stage_scratch_bytes(
    state: &WorkerState,
    incoming_program: u64,
    incoming_manifest: u64,
) -> Option<u64> {
    let cached = state.graph_cache.values().try_fold(0_u64, |total, entry| {
        total.checked_add(entry.program.length())
    })?;
    let partial = state
        .graph_staging
        .values()
        .try_fold(cached, |total, slot| {
            total
                .checked_add(slot.program_length)?
                .checked_add(slot.manifest_bytes)
        })?;
    // The downloader holds the encoded CBOR object while decoding allocates
    // ciphertext and opening allocates plaintext. All three can coexist
    // before the bounded chunk is appended to the partial stage.
    let chunk_buffers =
        runmat_execution_artifact::encryption::MAX_PROGRAM_CHUNK_ENCODED_OBJECT_BYTES
            .checked_add(
                runmat_execution_artifact::encryption::MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES + 16,
            )?
            .checked_add(
                runmat_execution_artifact::encryption::MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES,
            )?;
    partial
        .checked_add(incoming_program)?
        .checked_add(incoming_manifest)?
        .checked_add(chunk_buffers as u64)
}

struct ActiveAttempt {
    task: tokio::task::JoinHandle<()>,
    reply: Arc<super::attempt_reply::AttemptReply>,
    resources: ResourceRequest,
    assignment: runmat_execution::resource::ResourceAssignment,
    cancellation: Arc<super::worker_execution::AttemptCancellation>,
    collective: Option<Arc<super::collective::RemoteCollectiveChannel>>,
    fea_control: Arc<super::fea_control::WorkerControl>,
    cooperative: bool,
    finished: tokio::sync::watch::Receiver<bool>,
    request_digest: runmat_execution::Digest,
}

impl WorkerState {
    fn finish_attempt(
        &mut self,
        attempt_id: AttemptId,
        verified: bool,
        usage: Option<super::protocol::RetiredAttemptUsage>,
    ) -> Option<ActiveAttempt> {
        let active = self.attempts.remove(&attempt_id)?;
        if verified {
            if self.retirement_receipts.len() == MAX_RETIREMENT_RECEIPTS {
                if let Some(acknowledged) = self
                    .retirement_receipts
                    .iter()
                    .position(|receipt| receipt.claim.is_none())
                {
                    self.retirement_receipts.remove(acknowledged);
                }
            }
            if self.retirement_receipts.len() < MAX_RETIREMENT_RECEIPTS {
                self.retirement_receipts.push_back(WorkerRetirementReceipt {
                    attempt_id,
                    request_digest: active.request_digest,
                    usage,
                    claim: Some((active.resources.clone(), active.assignment.clone())),
                });
            } else {
                // No uncharged metadata or false absence proof: this worker
                // quarantines the still-charged allocation and stops admission.
                self.draining = true;
            }
        }
        // Every completed owner stays charged until its exact receipt ACK.
        // An unverified owner gets no receipt and remains quarantined.
        Some(active)
    }
}

struct GraphStageSlot {
    writer: Arc<Mutex<super::program_staging::GraphProgramStageWriter>>,
    program_digest: runmat_execution::Digest,
    program_length: u64,
    manifest_bytes: u64,
}

/// The wire carries a small staged reference; only the exact private
/// completed-stage owner can turn it into a callable request.
struct ResolvedRemoteAttempt {
    scheduling: AttemptRequest,
    program: WorkerProgramRequest,
    graph_wire_bytes: Option<usize>,
}

pub(super) struct WorkerLoopContext {
    pub(super) run_identity: String,
    pub(super) worker: WorkerSpec,
    pub(super) driver_fence: u64,
    pub(super) session_id: [u8; 16],
    pub(super) run_key: RunKeyMaterial,
    pub(super) limits: FrameLimits,
    pub(super) bundle_cache: Option<std::path::PathBuf>,
    pub(super) meshing_host: Option<super::worker_execution::RemoteMeshingHost>,
    /// Set only by an allocation-bound NodeAgent bootstrap from the driver.
    pub(super) ready_v1: bool,
    pub(super) journal: Option<WorkerAttemptJournalOwner>,
}

pub(super) async fn run_worker_loop(
    connection: Arc<dyn RemoteFrameRoute>,
    context: WorkerLoopContext,
) -> NativeExecutionResult<()> {
    let WorkerLoopContext {
        run_identity,
        worker,
        driver_fence,
        session_id,
        run_key,
        limits,
        bundle_cache,
        meshing_host,
        ready_v1,
        journal,
    } = context;
    let attempt_journal = journal
        .map(|owner| WorkerAttemptJournal::new(owner, worker.id, run_key.clone()))
        .transpose()?
        .map(|journal| Arc::new(std::sync::Mutex::new(journal)));
    let mut receiver = EncryptedFrameSession::new(
        run_identity.clone(),
        session_id,
        "driver-to-worker",
        1,
        run_key.clone(),
    )
    .map_err(protocol)?;
    let graph_run_key = run_key.clone();
    let graph_downloader =
        runmat_execution_transport_native::control::SignedArtifactDownloader::new();
    let sender = Arc::new(Mutex::new(
        EncryptedFrameSession::new(
            run_identity.clone(),
            session_id,
            "worker-to-driver",
            1,
            run_key,
        )
        .map_err(protocol)?,
    ));
    if ready_v1 {
        reply(
            connection.as_ref(),
            &sender,
            limits,
            RemoteWorkerReply {
                schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                correlation_id: "worker-ready-v1".into(),
                outcome: RemoteWorkerOutcome::WorkerReady {
                    ready: WorkerReady {
                        run_identity: run_identity.clone(),
                        session_id,
                        driver_fence,
                        worker_id: worker.id,
                        pool_id: worker.pool_id,
                        capabilities: super::worker_protocol::software_capabilities(),
                    },
                },
            },
        )
        .await?;
    }
    let state = Arc::new(Mutex::new(WorkerState {
        bundle: None,
        materialized_project: None,
        bundle_digest: None,
        attempts: HashMap::new(),
        retirement_receipts: VecDeque::with_capacity(MAX_RETIREMENT_RECEIPTS),
        allocated: ResourceAllocation::default(),
        cancelled_attempts: HashMap::new(),
        values: HashMap::new(),
        objects: super::object_transfer::RemoteObjectStore::new()?,
        draining: false,
        bundle_cache,
        graph_staging: HashMap::new(),
        graph_stage_download_in_flight: HashMap::new(),
        graph_programs: HashMap::new(),
        graph_cache: HashMap::new(),
    }));
    let drain_complete = Arc::new(tokio::sync::Notify::new());
    let mut graph_cache_reaper = tokio::time::interval(Duration::from_secs(30));
    let mut graph_stage_tasks = tokio::task::JoinSet::new();
    let result: NativeExecutionResult<()> = async {
    // A QUIC receive can already own a partially read stream. Keep that
    // future across housekeeping wakeups instead of discarding its bytes.
    let mut incoming_frame = connection.receive();
    loop {
        let notified = drain_complete.notified();
        let is_drained = {
            let state = state.lock().await;
            state.draining && state.attempts.is_empty()
        };
        if is_drained {
            return Ok(());
        }
        let frame = tokio::select! {
            frame = &mut incoming_frame => {
                let frame = frame?;
                incoming_frame = connection.receive();
                frame
            },
            _ = notified => continue,
            _ = graph_cache_reaper.tick() => {
                let retired = {
                    let mut state = state.lock().await;
                    retire_idle_graph_cache(&mut state, tokio::time::Instant::now(), true)
                };
                drop(retired);
                continue;
            }
            joined = graph_stage_tasks.join_next(), if !graph_stage_tasks.is_empty() => {
                // A completed signed download no longer needs a task slot;
                // keep the JoinSet bounded by concurrent staging, not by
                // the historical chunk count of this worker session.
                if joined.is_some_and(|result| result.is_err()) {
                    return Err(protocol("Graph Program staging task failed"));
                }
                continue;
            }
        };
        if !matches!(frame.kind, FrameKind::Control | FrameKind::Artifact) {
            return Err(protocol(
                "remote worker received an unsupported command frame",
            ));
        }
        let plaintext = receiver.open(&frame, limits).map_err(protocol)?;
        super::protocol_schema::admit_remote_worker_bytes(&plaintext).map_err(protocol)?;
        let request: RemoteWorkerRequest = serde_json::from_slice(&plaintext).map_err(protocol)?;
        if command_frame_kind(&request.command) != frame.kind {
            return Err(protocol(
                "remote worker command used the wrong encrypted frame kind",
            ));
        }
        if request.validate_schema().is_err() || request.driver_fence != driver_fence {
            reply_kind(
                connection.as_ref(),
                &sender,
                limits,
                frame.kind,
                rejected(
                    request.correlation_id,
                    "stale or unsupported driver authority",
                ),
            )
            .await?;
            continue;
        }
        match request.command {
            RemoteWorkerCommand::Describe => {
                reply(
                    connection.as_ref(),
                    &sender,
                    limits,
                    RemoteWorkerReply {
                        schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                        correlation_id: request.correlation_id,
                        outcome: RemoteWorkerOutcome::WorkerDescription {
                            worker: worker.clone(),
                        },
                    },
                )
                .await?;
            }
            RemoteWorkerCommand::InstallBundle {
                bundle_digest,
                bundle,
            } => {
                let cache = state.lock().await.bundle_cache.clone();
                let outcome =
                    match super::worker_bundle::install(cache.as_deref(), bundle_digest, &bundle) {
                        Ok(installed) => {
                            let receipt = installed.receipt.clone();
                            let mut state = state.lock().await;
                            state.bundle = Some(installed.bundle);
                            state.materialized_project = installed.materialized_project;
                            state.bundle_digest = Some(installed.digest);
                            RemoteWorkerOutcome::BundleStored { receipt }
                        }
                        Err(message) => RemoteWorkerOutcome::Rejected { message },
                    };
                reply(
                    connection.as_ref(),
                    &sender,
                    limits,
                    RemoteWorkerReply {
                        schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                        correlation_id: request.correlation_id,
                        outcome,
                    },
                )
                .await?;
            }
            RemoteWorkerCommand::ActivateBundle { bundle_digest } => {
                let cache = state.lock().await.bundle_cache.clone();
                let outcome = match cache
                    .as_deref()
                    .ok_or_else(|| "worker has no node bundle cache".to_string())
                    .and_then(|cache| super::worker_bundle::activate(cache, bundle_digest))
                {
                    Ok(installed) => {
                        let receipt = installed.receipt.clone();
                        let mut state = state.lock().await;
                        state.bundle = Some(installed.bundle);
                        state.materialized_project = installed.materialized_project;
                        state.bundle_digest = Some(installed.digest);
                        RemoteWorkerOutcome::BundleStored { receipt }
                    }
                    Err(message) => RemoteWorkerOutcome::Rejected { message },
                };
                reply(
                    connection.as_ref(),
                    &sender,
                    limits,
                    RemoteWorkerReply {
                        schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                        correlation_id: request.correlation_id,
                        outcome,
                    },
                )
                .await?;
            }
            RemoteWorkerCommand::BeginGraphProgram {
                attempt_id,
                encrypted_manifest,
                maximum_program_bytes,
            } => {
                let unavailable = {
                    let state = state.lock().await;
                    // Decode/open briefly holds the encoded manifest,
                    // ciphertext and plaintext together. Refuse that peak
                    // before allocating decoded copies or a private stage.
                    let manifest_peak = (encrypted_manifest.len() as u64).checked_mul(3);
                    state.draining
                        || state.cancelled_attempts.contains_key(&attempt_id)
                        || state.graph_staging.contains_key(&attempt_id)
                        || manifest_peak.is_none_or(|peak| {
                            graph_stage_scratch_bytes(&state, 0, peak)
                                .is_none_or(|bytes| bytes > worker.resources.scratch_bytes)
                        })
                };
                let outcome = if unavailable {
                    RemoteWorkerOutcome::Rejected {
                        message: "Graph attempt cannot begin private staging".into(),
                    }
                } else {
                    match super::worker_graph_stage::begin(
                        graph_run_key.clone(),
                        &run_identity,
                        &encrypted_manifest,
                        maximum_program_bytes,
                    )
                    .await
                    {
                        Ok(writer) => {
                            let manifest = writer.manifest();
                            let program_digest = manifest.program_digest;
                            let program_length = manifest.program_length;
                            let manifest_bytes = encrypted_manifest.len() as u64;
                            let chunk_count = manifest.chunks.len() as u32;
                            let mut state = state.lock().await;
                            let mut retired = retire_idle_graph_cache(
                                &mut state,
                                tokio::time::Instant::now(),
                                true,
                            );
                            let outcome = if state.draining
                                || state.cancelled_attempts.contains_key(&attempt_id)
                                || state.graph_staging.contains_key(&attempt_id)
                            {
                                RemoteWorkerOutcome::Rejected {
                                    message: "Graph attempt cannot begin private staging".into(),
                                }
                            } else if let Some(existing) = state.graph_programs.get(&attempt_id) {
                                if existing.digest() == program_digest
                                    && existing.length() == program_length
                                {
                                    RemoteWorkerOutcome::GraphProgramStaged {
                                        receipt: super::RemoteGraphProgramReceipt {
                                            program_digest,
                                            program_length,
                                        },
                                    }
                                } else {
                                    RemoteWorkerOutcome::Rejected {
                                        message: "staged Graph digest has a different length"
                                            .into(),
                                    }
                                }
                            } else if let Some(cached) = state.graph_cache.get_mut(&program_digest)
                            {
                                if cached.program.length() != program_length {
                                    RemoteWorkerOutcome::Rejected {
                                        message: "staged Graph digest has a different length"
                                            .into(),
                                    }
                                } else {
                                    cached.last_used = tokio::time::Instant::now();
                                    let program = Arc::clone(&cached.program);
                                    state.graph_programs.insert(attempt_id, program);
                                    RemoteWorkerOutcome::GraphProgramStaged {
                                        receipt: super::RemoteGraphProgramReceipt {
                                            program_digest,
                                            program_length,
                                        },
                                    }
                                }
                            } else {
                                if graph_stage_scratch_bytes(
                                    &state,
                                    program_length,
                                    manifest_bytes,
                                )
                                    .is_none_or(|bytes| bytes > worker.resources.scratch_bytes)
                                {
                                    retired.extend(retire_idle_graph_cache(
                                        &mut state,
                                        tokio::time::Instant::now(),
                                        false,
                                    ));
                                }
                                if graph_stage_scratch_bytes(
                                    &state,
                                    program_length,
                                    manifest_bytes,
                                )
                                    .is_none_or(|bytes| bytes > worker.resources.scratch_bytes)
                                {
                                    RemoteWorkerOutcome::Rejected {
                                        message:
                                            "Graph Program staging exceeds worker scratch allocation"
                                                .into(),
                                    }
                                } else {
                                    state.graph_staging.insert(
                                        attempt_id,
                                        GraphStageSlot {
                                            writer: Arc::new(Mutex::new(writer)),
                                            program_digest,
                                            program_length,
                                            manifest_bytes,
                                        },
                                    );
                                    RemoteWorkerOutcome::GraphProgramPrepared {
                                        program_digest,
                                        program_length,
                                        chunk_count,
                                    }
                                }
                            };
                            drop(state);
                            drop(retired);
                            outcome
                        }
                        Err(error) => RemoteWorkerOutcome::Rejected {
                            message: error.to_string(),
                        },
                    }
                };
                reply(
                    connection.as_ref(),
                    &sender,
                    limits,
                    RemoteWorkerReply {
                        schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                        correlation_id: request.correlation_id,
                        outcome,
                    },
                )
                .await?;
            }
            RemoteWorkerCommand::StageGraphProgramChunk {
                attempt_id,
                program_digest,
                ordinal,
                download,
            } => {
                let writer = {
                    let mut state = state.lock().await;
                    if state.draining
                        || state.cancelled_attempts.contains_key(&attempt_id)
                        || state
                            .graph_stage_download_in_flight
                            .contains_key(&attempt_id)
                    {
                        None
                    } else {
                        let writer = state
                            .graph_staging
                            .get(&attempt_id)
                            .filter(|slot| slot.program_digest == program_digest)
                            .map(|slot| Arc::clone(&slot.writer));
                        if writer.is_some() {
                            state.graph_stage_download_in_flight.insert(
                                attempt_id,
                                runmat_execution::cancellation::CancellationSource::new(),
                            );
                        }
                        writer
                    }
                };
                let Some(writer) = writer else {
                    reply(
                        connection.as_ref(),
                        &sender,
                        limits,
                        rejected(
                            request.correlation_id,
                            "worker has no available Graph Program staging",
                        ),
                    )
                    .await?;
                    continue;
                };
                let stage_state = Arc::clone(&state);
                let stage_connection = Arc::clone(&connection);
                let stage_sender = Arc::clone(&sender);
                let stage_downloader = graph_downloader.clone();
                let stage_cancellation = {
                    let state = state.lock().await;
                    state
                        .graph_stage_download_in_flight
                        .get(&attempt_id)
                        .expect("accepted stage has cancellation owner")
                        .token()
                };
                graph_stage_tasks.spawn_local(async move {
                    let result = tokio::select! {
                        result = async {
                            let mut writer_guard = writer.lock().await;
                            super::worker_graph_stage::download_next(
                                &mut writer_guard,
                                &stage_downloader,
                                program_digest,
                                ordinal,
                                &download,
                            ).await
                        } => result,
                        _ = stage_cancellation.cancelled() => {
                            Err(protocol("Graph Program chunk staging was cancelled"))
                        }
                    };
                    let retired = {
                        let mut state = stage_state.lock().await;
                        state.graph_stage_download_in_flight.remove(&attempt_id);
                        if result.is_err()
                            && state
                                .graph_staging
                                .get(&attempt_id)
                                .is_some_and(|slot| Arc::ptr_eq(&slot.writer, &writer))
                        {
                            state.graph_staging.remove(&attempt_id)
                        } else {
                            None
                        }
                    };
                    drop(writer);
                    drop(retired);
                    let outcome = match result {
                        Ok(()) => RemoteWorkerOutcome::Acknowledged,
                        Err(error) => RemoteWorkerOutcome::Rejected {
                            message: error.to_string(),
                        },
                    };
                    let _ = reply(
                        stage_connection.as_ref(),
                        &stage_sender,
                        limits,
                        RemoteWorkerReply {
                            schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                            correlation_id: request.correlation_id,
                            outcome,
                        },
                    )
                    .await;
                });
            }
            RemoteWorkerCommand::FinishGraphProgram {
                attempt_id,
                program_digest,
            } => {
                let slot = {
                    let mut state = state.lock().await;
                    if state.draining
                        || state.cancelled_attempts.contains_key(&attempt_id)
                        || state
                            .graph_stage_download_in_flight
                            .contains_key(&attempt_id)
                    {
                        None
                    } else {
                        state.graph_staging.remove(&attempt_id)
                    }
                };
                let outcome = match slot {
                    Some(slot)
                        if slot.program_digest == program_digest
                            && Arc::strong_count(&slot.writer) == 1 =>
                    {
                        let Some(writer) = Arc::into_inner(slot.writer) else {
                            unreachable!("only the worker owns finished Graph staging")
                        };
                        let writer = writer.into_inner();
                        match super::worker_graph_stage::finish(writer, program_digest).await {
                            Ok(program) => {
                                let receipt = super::RemoteGraphProgramReceipt {
                                    program_digest: program.digest(),
                                    program_length: program.length(),
                                };
                                let incoming = Arc::new(program);
                                let mut state = state.lock().await;
                                if state.draining || state.cancelled_attempts.contains_key(&attempt_id)
                                {
                                    RemoteWorkerOutcome::Rejected {
                                        message:
                                            "Graph attempt was cancelled before staging finished"
                                                .into(),
                                    }
                                } else {
                                    let selected = if let Some(cached) =
                                        state.graph_cache.get_mut(&program_digest)
                                    {
                                        if cached.program.length() != incoming.length() {
                                            None
                                        } else {
                                            cached.last_used = tokio::time::Instant::now();
                                            Some(Arc::clone(&cached.program))
                                        }
                                    } else {
                                        state.graph_cache.insert(
                                            program_digest,
                                            CachedGraphProgram {
                                                program: Arc::clone(&incoming),
                                                last_used: tokio::time::Instant::now(),
                                            },
                                        );
                                        Some(Arc::clone(&incoming))
                                    };
                                    if let Some(selected) = selected {
                                        let retired =
                                            state.graph_programs.insert(attempt_id, selected);
                                        drop(state);
                                        drop(retired);
                                        RemoteWorkerOutcome::GraphProgramStaged { receipt }
                                    } else {
                                        drop(state);
                                        RemoteWorkerOutcome::Rejected {
                                            message: "staged Graph digest has a different length"
                                                .into(),
                                        }
                                    }
                                }
                            }
                            Err(error) => RemoteWorkerOutcome::Rejected {
                                message: error.to_string(),
                            },
                        }
                    }
                    Some(slot) => {
                        let mut state = state.lock().await;
                        if slot.program_digest == program_digest
                            && !state.cancelled_attempts.contains_key(&attempt_id)
                        {
                            state.graph_staging.insert(attempt_id, slot);
                        }
                        RemoteWorkerOutcome::Rejected {
                            message: "Graph Program staging identity or ownership changed".into(),
                        }
                    }
                    None => RemoteWorkerOutcome::Rejected {
                        message: "worker has no completed Graph Program staging".into(),
                    },
                };
                reply(
                    connection.as_ref(),
                    &sender,
                    limits,
                    RemoteWorkerReply {
                        schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                        correlation_id: request.correlation_id,
                        outcome,
                    },
                )
                .await?;
            }
            RemoteWorkerCommand::PutValue { reference, encoded } => {
                let outcome = match super::value_transfer::decode_value(
                    &reference,
                    &encoded,
                    &run_identity,
                ) {
                    Ok(value) => {
                        let mut state = state.lock().await;
                        match state.values.get(&reference.id) {
                            Some(existing) if existing != &value => RemoteWorkerOutcome::Rejected {
                                message: "remote value id was reused for different content".into(),
                            },
                            _ => {
                                state.values.insert(reference.id, value);
                                RemoteWorkerOutcome::ValueStored {
                                    receipt: super::RemoteValueReceipt {
                                        value_id: reference.id,
                                        encoded_bytes: encoded.len() as u64,
                                    },
                                }
                            }
                        }
                    }
                    Err(error) => RemoteWorkerOutcome::Rejected {
                        message: error.to_string(),
                    },
                };
                reply(
                    connection.as_ref(),
                    &sender,
                    limits,
                    RemoteWorkerReply {
                        schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                        correlation_id: request.correlation_id,
                        outcome,
                    },
                )
                .await?;
            }
            RemoteWorkerCommand::Execute { attempt, console_budget } => {
                let valid_console_budget = match (attempt.scheduling.task.console, console_budget) {
                    (None, None) => true,
                    (Some(controls), Some(budget)) => budget.validate().is_ok()
                        && budget.max_records <= u64::from(controls.max_records)
                        && budget.max_text_bytes <= controls.max_text_bytes
                        && budget.max_encoded_bytes <= (limits.maximum_payload_bytes as u64).saturating_sub(64 * 1024)
                        && budget.minimum_memory_bytes().is_ok_and(|bytes| bytes <= attempt.scheduling.task.resources.memory_bytes),
                    _ => false,
                };
                if !valid_console_budget {
                    reply(connection.as_ref(), &sender, limits,
                        rejected(request.correlation_id, "console subtree grant differs from its authenticated invocation or terminal frame")).await?;
                    continue;
                }
                if attempt_journal.as_ref().is_some_and(|journal| {
                    journal
                        .lock()
                        .expect("worker attempt journal poisoned")
                        .is_full()
                }) {
                    reply(
                        connection.as_ref(),
                        &sender,
                        limits,
                        RemoteWorkerReply {
                            schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                            correlation_id: request.correlation_id,
                            outcome: RemoteWorkerOutcome::AdmissionDeferred {
                                attempt_id: attempt.scheduling.id,
                                worker_id: worker.id,
                                driver_fence,
                            },
                        },
                    )
                    .await?;
                    continue;
                }
                let attempt = match resolve_attempt(&state, &worker, driver_fence, *attempt).await {
                    Ok(attempt) => attempt,
                    Err(message) => {
                        reply(
                            connection.as_ref(),
                            &sender,
                            limits,
                            rejected(request.correlation_id, message),
                        )
                        .await?;
                        continue;
                    }
                };
                let rejection =
                    validate_and_reserve_attempt(&state, &worker, driver_fence, &attempt).await;
                if let Some(message) = rejection {
                    reply(
                        connection.as_ref(),
                        &sender,
                        limits,
                        rejected(request.correlation_id, message),
                    )
                    .await?;
                    continue;
                }
                let attempt_id = attempt.scheduling.id;
                let scheduling = attempt.scheduling.clone();
                let cancellation =
                    Arc::new(super::worker_execution::AttemptCancellation::default());
                let (fea_control, mut fea_calls) = super::fea_control::WorkerControl::new(
                    runmat_execution::ProgramExecutionAssignment {
                        scope_id: scheduling.scope_id,
                        pool_id: scheduling.task.pool_id,
                        task_id: scheduling.task_id,
                        attempt_id: scheduling.id,
                        worker_id: scheduling.worker_id,
                        backend: runmat_execution::PoolBackend::Remote,
                        resources: scheduling.resource_assignment.clone(),
                    },
                    u64::from(scheduling.task.resources.cpu_millicores),
                    cancellation.token(),
                );
                let fea_control_for_replies = Arc::clone(&fea_control);
                if let Some(journal) = &attempt_journal {
                    let begun = {
                        let mut journal = journal.lock().expect("worker attempt journal poisoned");
                        journal.begin(&scheduling)
                    };
                    if let Err(error) = begun {
                        let mut state = state.lock().await;
                        release(
                            &mut state.allocated,
                            &scheduling.task.resources,
                            &scheduling.resource_assignment,
                        );
                        state.draining = true;
                        drop(state);
                        reply(
                            connection.as_ref(),
                            &sender,
                            limits,
                            rejected(request.correlation_id, error.to_string()),
                        )
                        .await?;
                        continue;
                    }
                }
                let request_digest = runmat_execution::Digest::sha256(
                    &serde_json::to_vec(&scheduling).map_err(protocol)?,
                );
                let reserved_resources = scheduling.task.resources.clone();
                let reserved_assignment = scheduling.resource_assignment.clone();
                let graph_run_identity = run_identity.clone();
                let graph_worker_resources = worker.resources.clone();
                let graph_worker_host = worker.host.clone();
                let correlation_id = request.correlation_id;
                let attempt_reply = super::attempt_reply::AttemptReply::spawn(
                    Arc::clone(&connection),
                    Arc::clone(&sender),
                    limits,
                    correlation_id.clone(),
                    console_budget,
                );
                let attempt_reply_for_task = Arc::clone(&attempt_reply);
                let connection = Arc::clone(&connection);
                let sender = Arc::clone(&sender);
                let collective = match &attempt.program {
                    WorkerProgramRequest::Direct(program) => match &program.context {
                        runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
                            Some(super::collective::RemoteCollectiveChannel::new(
                                attempt_id,
                                correlation_id.clone(),
                                task.clone(),
                                Arc::clone(&connection),
                                Arc::clone(&sender),
                                limits,
                            ))
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let collective_for_task = collective.as_ref().map(Arc::clone);
                let fea_control_for_task = Arc::clone(&fea_control);
                let state_for_task = Arc::clone(&state);
                let (materialized_project, bundle, objects) = {
                    let state = state.lock().await;
                    (
                        state.materialized_project.as_ref().cloned(),
                        state.bundle.as_ref().cloned(),
                        state.objects.clone(),
                    )
                };
                // Direct invocations run in the installed bundle's frozen
                // portable environment; Graph attempts carry their own.
                let environment = bundle
                    .as_ref()
                    .map(|bundle| bundle.manifest.portable_environment.clone())
                    .unwrap_or_default();
                // Workload artifacts read object inputs from the worker store
                // and cooperate with cancellation instead of being aborted.
                let externalized = matches!(
                    &attempt.program,
                    WorkerProgramRequest::Direct(program)
                        if matches!(
                            program.artifact.form,
                            ExecutableForm::MeshingWorkload
                                | ExecutableForm::AnalysisWorkload
                                | ExecutableForm::GeometryWorkload
                        )
                );
                let graph_attempt = matches!(&attempt.program, WorkerProgramRequest::Graph(_));
                // Execution now owns an OS blocking thread even for plain language work.
                // Never abort only its async wrapper and then claim the thread retired.
                let cooperative = true;
                let cancellation_for_task = Arc::clone(&cancellation);
                let cancellation_after_task = Arc::clone(&cancellation);
                let (finished_sender, finished) = tokio::sync::watch::channel(false);
                let meshing_host = meshing_host.clone();
                let drain_complete_for_task = Arc::clone(&drain_complete);
                let attempt_journal_for_task = attempt_journal.as_ref().map(Arc::clone);
                let (start_sender, start_receiver) = tokio::sync::oneshot::channel();
                let task = tokio::task::spawn_local(async move {
                    if start_receiver.await.is_err() {
                        return;
                    }
                    let started = Instant::now();
                    let original_clock = super::fea_admission::AllocationClock::begin();
                    let program = attempt.program;
                    let materialized = match &program {
                        WorkerProgramRequest::Graph(_) => Ok(Vec::new()),
                        WorkerProgramRequest::Direct(direct) if externalized => {
                            Ok(direct.arguments.clone())
                        }
                        WorkerProgramRequest::Direct(direct) => {
                            let state = state_for_task.lock().await;
                            direct
                                .arguments
                                .iter()
                                .map(|value| {
                                    super::value_transfer::materialize(value, &state.values)
                                })
                                .collect::<NativeExecutionResult<Vec<_>>>()
                        }
                    };
                    let (progress_sender, mut progress_receiver) = tokio::sync::mpsc::channel(256);
                    let scheduling_for_execution = scheduling.clone();
                    // A numerical FEA permit may synchronously wait for its parent grant.
                    // Keep the encrypted-session LocalSet available to relay that control
                    // while this owned attempt runs on its own bounded blocking thread.
                    let execution = tokio::task::spawn_blocking(move || {
                        let runtime = match tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                        {
                            Ok(runtime) => runtime,
                            Err(error) => {
                                return super::worker_execution::WorkerExecutionOutcome::RetirementUnverified(
                                    format!("worker execution runtime unavailable: {error}"),
                                )
                            }
                        };
                        tokio::task::LocalSet::new().block_on(&runtime, async move {
                        match materialized {
                            Ok(arguments) => {
                                let program = match program {
                                    WorkerProgramRequest::Direct(mut direct) => {
                                        direct.arguments = arguments;
                                        WorkerProgramRequest::Direct(direct)
                                    }
                                    WorkerProgramRequest::Graph(graph) => {
                                        WorkerProgramRequest::Graph(graph)
                                    }
                                };
                                let collective = collective_for_task.map(|channel| {
                                    std::rc::Rc::new(
                                        super::collective::RemoteCollectiveService::new(channel),
                                    )
                                        as std::rc::Rc<
                                            dyn runmat_runtime::context::RuntimeCollectiveService,
                                        >
                                });
                                let fea_admission = match super::fea_admission::for_verified_attempt(
                                    &graph_run_identity, &scheduling_for_execution, original_clock, cancellation_for_task.token(),
                                    fea_control_for_task,
                                ) {
                                    Ok(admission) => admission,
                                    Err(error) => return super::worker_execution::WorkerExecutionOutcome::Response(
                                        runmat_execution_artifact::ProgramExecutionResponse::Failure { message: error.to_string() }),
                                };
                                fea_admission.run_future(super::worker_execution::execute(
                                    super::worker_execution::WorkerExecutionInput {
                                        attempt_id,
                                        program,
                                        graph_attempt: graph_attempt.then_some(
                                            super::worker_execution::GraphAttemptContext {
                                                run_identity: graph_run_identity,
                                                resources: graph_worker_resources,
                                                host: graph_worker_host,
                                            },
                                        ),
                                        project: materialized_project,
                                        bundle,
                                        environment,
                                        resources: scheduling_for_execution.task.resources.clone(),
                                        console_budget,
                                    },
                                    super::worker_execution::WorkerExecutionPorts {
                                        collective,
                                        meshing_host,
                                        objects,
                                        cancellation: cancellation_for_task,
                                        progress_sender,
                                    },
                                ))
                                .await
                            }
                            Err(error) => super::worker_execution::WorkerExecutionOutcome::Response(
                                ProgramExecutionResponse::Failure {
                                    message: error.to_string(),
                                },
                            ),
                        }
                        })
                    });
                    tokio::pin!(execution);
                    let response = loop {
                        tokio::select! {
                            Some(call) = fea_calls.recv() => {
                                let bounded = runmat_canonical_codec::encode_json_bounded(
                                    &call,
                                    crate::fea_proxy::MAX_FRAME_BYTES,
                                );
                                if bounded.is_err() {
                                    let _ = fea_control_for_replies.complete(
                                        call.sequence,
                                        crate::fea_proxy::Reply::Denied {
                                            denied: runmat_runtime::analysis::fea_run_admission::FeaRunDenied::LicenseInvalid,
                                        },
                                    );
                                } else {
                                    let sequence = call.sequence;
                                    if reply(
                                        connection.as_ref(),
                                        &sender,
                                        limits,
                                        RemoteWorkerReply {
                                            schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                                            correlation_id: correlation_id.clone(),
                                            outcome: RemoteWorkerOutcome::FeaAdmissionCall {
                                                attempt_id,
                                                call,
                                            },
                                        },
                                    ).await.is_err() {
                                        let _ = fea_control_for_replies.complete(
                                            sequence,
                                            crate::fea_proxy::Reply::Denied {
                                                denied: runmat_runtime::analysis::fea_run_admission::FeaRunDenied::ServerUnreachable,
                                            },
                                        );
                                    }
                                }
                            }
                            Some(progress) = progress_receiver.recv() => {
                                let _ = reply_progress(
                                    connection.as_ref(),
                                    &sender,
                                    limits,
                                    &correlation_id,
                                    attempt_id,
                                    progress,
                                ).await;
                            }
                            response = &mut execution => {
                                while let Ok(progress) = progress_receiver.try_recv() {
                                    let _ = reply_progress(
                                        connection.as_ref(),
                                        &sender,
                                        limits,
                                        &correlation_id,
                                        attempt_id,
                                        progress,
                                    ).await;
                                }
                                break response.unwrap_or_else(|error| {
                                    super::worker_execution::WorkerExecutionOutcome::RetirementUnverified(
                                        format!("worker execution thread stopped: {error}"),
                                    )
                                });
                            }
                        }
                    };
                    // Cancellation enters the selected runtime token and
                    // completes cooperative/adapter retirement before terminal
                    // publication. Never publish a late successful result after
                    // the scheduler has cancelled the attempt.
                    let mut console = None;
                    let (mut report, mut retirement_unverified) = match response {
                        super::worker_execution::WorkerExecutionOutcome::RetirementUnverified(message) => (
                            runmat_execution_runner::AttemptReport::RetirementUnverified { message },
                            true,
                        ),
                        super::worker_execution::WorkerExecutionOutcome::Response(response) => {
                            let report = if cancellation_after_task.was_cancelled() {
                                runmat_execution_runner::AttemptReport::Cancelled
                            } else {
                                super::worker_execution::report(
                                    &scheduling,
                                    response,
                                    u64::try_from(started.elapsed().as_millis())
                                        .unwrap_or(u64::MAX)
                                        .max(1),
                                )
                            };
                            (report, false)
                        }
                        super::worker_execution::WorkerExecutionOutcome::CapturedResponse(response, capture) => {
                            console = capture;
                            let report = if cancellation_after_task.was_cancelled() {
                                runmat_execution_runner::AttemptReport::Cancelled
                            } else {
                                super::worker_execution::report(&scheduling, response,
                                    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX).max(1))
                            };
                            (report, false)
                        }
                    };
                    if !retirement_unverified {
                        if let Some(journal) = &attempt_journal_for_task {
                            if let Err(error) = journal
                                .lock()
                                .expect("worker attempt journal poisoned")
                                .complete(attempt_id, &report)
                            {
                                report = runmat_execution_runner::AttemptReport::RetirementUnverified {
                                    message: format!("attempt result inventory was not durably committed: {error}"),
                                };
                                retirement_unverified = true;
                            }
                        }
                    }
                    let retired_usage = match &report {
                        runmat_execution_runner::AttemptReport::Succeeded { result } => result
                            .resources()
                            .map(super::protocol::RetiredAttemptUsage::from_accounting),
                        _ => None,
                    };
                    if retirement_unverified {
                        // Stop admission first, but keep the active attempt
                        // present until its fatal report has had a bounded
                        // encrypted send. Otherwise the drain loop may close
                        // the connection before the reply task is polled.
                        state_for_task.lock().await.draining = true;
                        attempt_reply_for_task.complete_before_drain_with_console(report, console).await;
                    } else {
                        attempt_reply_for_task.complete_with_console(report, console);
                    }
                    let mut state = state_for_task.lock().await;
                    let _ = state.finish_attempt(attempt_id, !retirement_unverified, retired_usage);
                    finished_sender.send_replace(true);
                    let retired_program = state.graph_programs.remove(&attempt_id);
                    if state.draining && state.attempts.is_empty() {
                        drain_complete_for_task.notify_waiters();
                    }
                    drop(state);
                    drop(retired_program);
                });
                state.lock().await.attempts.insert(
                    attempt_id,
                    ActiveAttempt {
                        task,
                        reply: attempt_reply,
                        resources: reserved_resources,
                        assignment: reserved_assignment,
                        cancellation,
                        collective,
                        fea_control,
                        cooperative,
                        finished,
                        request_digest,
                    },
                );
                let _ = start_sender.send(());
            }
            RemoteWorkerCommand::CompleteCollective {
                attempt_id,
                context,
                id,
                sequence,
                result,
            } => {
                let collective = state
                    .lock()
                    .await
                    .attempts
                    .get(&attempt_id)
                    .and_then(|active| active.collective.as_ref().map(Arc::clone));
                let completion = collective
                    .ok_or_else(|| {
                        "remote collective completion targets an unknown or non-SPMD attempt"
                            .to_string()
                    })
                    .and_then(|collective| collective.complete(&context, id, sequence, result));
                let response = match completion {
                    Ok(()) => acknowledged(request.correlation_id),
                    Err(message) => rejected(request.correlation_id, message),
                };
                reply(connection.as_ref(), &sender, limits, response).await?;
            }
            RemoteWorkerCommand::CompleteFeaAdmission {
                attempt_id,
                sequence,
                reply: parent_reply,
            } => {
                let control = state
                    .lock()
                    .await
                    .attempts
                    .get(&attempt_id)
                    .map(|active| Arc::clone(&active.fea_control));
                let completion = control
                    .ok_or("FEA completion targets an unknown attempt")
                    .and_then(|control| control.complete(sequence, parent_reply));
                let response = match completion {
                    Ok(()) => acknowledged(request.correlation_id),
                    Err(message) => rejected(request.correlation_id, message),
                };
                reply(connection.as_ref(), &sender, limits, response).await?;
            }
            RemoteWorkerCommand::QueryPhysicalRetirement { attempt } => {
                let digest = runmat_execution::Digest::sha256(
                    &serde_json::to_vec(attempt.as_ref()).map_err(protocol)?,
                );
                let exact_scope = attempt.driver_fence == driver_fence
                    && attempt.worker_id == worker.id
                    && attempt.task.pool_id == worker.pool_id;
                let retired = if exact_scope {
                    state.lock().await.retirement_receipts.iter().find(|receipt| {
                        receipt.attempt_id == attempt.id
                            && receipt.request_digest == digest
                            && receipt.claim.is_some()
                    }).map(|receipt| receipt.usage)
                } else { None };
                let outcome = if let Some(usage) = retired {
                    RemoteWorkerOutcome::PhysicalRetirement {
                        run_identity: run_identity.clone(),
                        session_id,
                        driver_fence,
                        worker_id: worker.id,
                        pool_id: worker.pool_id,
                        attempt_id: attempt.id,
                        request_digest: digest,
                        usage,
                    }
                } else {
                    RemoteWorkerOutcome::RetirementUnavailable
                };
                reply(connection.as_ref(), &sender, limits, RemoteWorkerReply {
                    schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                    correlation_id: request.correlation_id,
                    outcome,
                }).await?;
            }
            RemoteWorkerCommand::AcknowledgePhysicalRetirement { attempt } => {
                let digest = runmat_execution::Digest::sha256(
                    &serde_json::to_vec(attempt.as_ref()).map_err(protocol)?,
                );
                let exact_scope = attempt.driver_fence == driver_fence
                    && attempt.worker_id == worker.id
                    && attempt.task.pool_id == worker.pool_id;
                let acknowledged_exact = if exact_scope {
                    let mut state = state.lock().await;
                    if let Some(index) = state.retirement_receipts.iter().position(|receipt| {
                        receipt.attempt_id == attempt.id && receipt.request_digest == digest
                    }) {
                        // Physical ACK releases runtime capacity once, but
                        // not the fixed 256-entry worker-allocation journal:
                        // an older durable driver checkpoint can still need
                        // this original attempt's measured interval.
                        let claim = state.retirement_receipts.get_mut(index)
                            .expect("matched worker retirement receipt").claim.take();
                        if let Some((resources, assignment)) = claim {
                            release(&mut state.allocated, &resources, &assignment);
                        }
                        true
                    } else { false }
                } else { false };
                let response = if acknowledged_exact {
                    acknowledged(request.correlation_id)
                } else {
                    rejected(request.correlation_id, "retirement acknowledgement has no exact outstanding owner")
                };
                reply(connection.as_ref(), &sender, limits, response).await?;
            }
            RemoteWorkerCommand::PruneDurableCheckpoint { sequence, attempts } => {
                let outcome = match &attempt_journal {
                    Some(journal) => journal
                        .lock()
                        .expect("worker attempt journal poisoned")
                        .prune_after_checkpoint(sequence, &attempts),
                    None => Err(protocol("worker has no allocation-bound attempt journal")),
                };
                let response = match outcome {
                    Ok(()) => acknowledged(request.correlation_id),
                    Err(error) => rejected(request.correlation_id, error.to_string()),
                };
                reply(connection.as_ref(), &sender, limits, response).await?;
            }
            RemoteWorkerCommand::Cancel { attempt_id, reason } => {
                let mut state = state.lock().await;
                let mut cancelled_reply = None;
                let mut completion = None;
                let mut aborted_task = None;
                // A Cancel may overtake Execute while the driver streams a
                // staged Graph. Retain that exact attempt's cancellation so
                // a later Execute cannot start foreign effects.
                if !state.cancelled_attempts.contains_key(&attempt_id)
                    && state.cancelled_attempts.len() >= MAX_PREEXECUTION_CANCELLATIONS
                {
                    state.draining = true;
                } else {
                    state.cancelled_attempts.entry(attempt_id).or_insert(reason);
                }
                let reason = state.cancelled_attempts.get(&attempt_id).copied().unwrap_or(reason);
                let retired_stage = state.graph_staging.remove(&attempt_id);
                let retired_program = state.graph_programs.remove(&attempt_id);
                let stage_cancellation = state
                    .graph_stage_download_in_flight
                    .get(&attempt_id)
                    .cloned();
                if let Some(active) = state.attempts.get(&attempt_id) {
                    active.cancellation.cancel(reason);
                    if active.cooperative {
                        completion = Some(active.finished.clone());
                    } else if let Some(active) = state.finish_attempt(attempt_id, false, None) {
                        // A control acknowledgement does not certify the
                        // process-tree owner. Quarantine this reservation.
                        state.draining = true;
                        active.task.abort();
                        cancelled_reply = Some(active.reply);
                        aborted_task = Some(active.task);
                    }
                }
                if state.draining && state.attempts.is_empty() {
                    drain_complete.notify_waiters();
                }
                drop(state);
                if let Some(stage_cancellation) = stage_cancellation {
                    stage_cancellation.cancel(reason);
                }
                drop(retired_stage);
                drop(retired_program);
                if let Some(cancelled_reply) = cancelled_reply {
                    cancelled_reply.complete(runmat_execution_runner::AttemptReport::Cancelled);
                }
                // Retirement may itself be waiting for a parent FEA grant on
                // this encrypted session. Keep its control receiver running,
                // but do not acknowledge Cancel until the attempt has stopped.
                if completion.is_some() || aborted_task.is_some() {
                    let connection = Arc::clone(&connection);
                    let sender = Arc::clone(&sender);
                    tokio::task::spawn_local(async move {
                        if let Some(mut completion) = completion {
                            if !*completion.borrow() {
                                let _ = completion.changed().await;
                            }
                        }
                        if let Some(aborted_task) = aborted_task {
                            let _ = aborted_task.await;
                        }
                        let _ = reply(
                            connection.as_ref(),
                            &sender,
                            limits,
                            acknowledged(request.correlation_id),
                        )
                        .await;
                    });
                } else {
                    reply(
                        connection.as_ref(),
                        &sender,
                        limits,
                        acknowledged(request.correlation_id),
                    )
                    .await?;
                }
            }
            RemoteWorkerCommand::Drain => {
                let mut state = state.lock().await;
                state.draining = true;
                let drained = state.attempts.is_empty();
                let stage_cancellations = state
                    .graph_stage_download_in_flight
                    .values()
                    .cloned()
                    .collect::<Vec<_>>();
                let retired_staging = std::mem::take(&mut state.graph_staging);
                let retired_cache =
                    retire_idle_graph_cache(&mut state, tokio::time::Instant::now(), false);
                drop(state);
                for stage_cancellation in stage_cancellations {
                    stage_cancellation.cancel(runmat_execution::CancellationReason::Shutdown);
                }
                drop(retired_staging);
                drop(retired_cache);
                reply(
                    connection.as_ref(),
                    &sender,
                    limits,
                    acknowledged(request.correlation_id),
                )
                .await?;
                if drained {
                    // Let QUIC deliver the terminal acknowledgement before the
                    // listener endpoint is dropped and closes the connection.
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    return Ok(());
                }
            }
            command => {
                let outcome = state.lock().await.objects.handle(command, &run_identity);
                reply_kind(
                    connection.as_ref(),
                    &sender,
                    limits,
                    FrameKind::Artifact,
                    RemoteWorkerReply {
                        schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                        correlation_id: request.correlation_id,
                        outcome,
                    },
                )
                .await?;
            }
        }
    }
    }
    .await;
    // The worker loop owns every signed HTTP stage task. On Drain, QUIC
    // disconnect, or any protocol error, cancellation and task retirement
    // complete before its private program roots can be released.
    let (stage_cancellations, retired_staging, retired_programs, retired_cache, active_attempts) = {
        let mut state = state.lock().await;
        let stage_cancellations = state
            .graph_stage_download_in_flight
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let retired_staging = std::mem::take(&mut state.graph_staging);
        let retired_programs = std::mem::take(&mut state.graph_programs);
        let retired_cache = std::mem::take(&mut state.graph_cache);
        let active_attempts = std::mem::take(&mut state.attempts);
        (
            stage_cancellations,
            retired_staging,
            retired_programs,
            retired_cache,
            active_attempts,
        )
    };
    // Route loss stops dispatch, but is not a retirement certificate. Retain
    // ownership until every in-process attempt task (including any joined
    // blocking kernel) has actually completed. The node's process-tree owner
    // is still required for the externally trusted certificate.
    for active in active_attempts.values() {
        active
            .cancellation
            .cancel(runmat_execution::CancellationReason::Shutdown);
    }
    for (_, active) in active_attempts {
        let _ = active.task.await;
    }
    for cancellation in stage_cancellations {
        cancellation.cancel(runmat_execution::CancellationReason::Shutdown);
    }
    graph_stage_tasks.abort_all();
    while graph_stage_tasks.join_next().await.is_some() {}
    drop(retired_staging);
    drop(retired_programs);
    drop(retired_cache);
    result
}

async fn resolve_attempt(
    state: &Mutex<WorkerState>,
    worker: &WorkerSpec,
    driver_fence: u64,
    attempt: RemoteAttempt,
) -> Result<ResolvedRemoteAttempt, String> {
    let RemoteAttempt {
        scheduling,
        program,
    } = attempt;
    scheduling.validate().map_err(|error| error.to_string())?;
    let staged = {
        let state = state.lock().await;
        if state.draining
            || scheduling.driver_fence != driver_fence
            || scheduling.worker_id != worker.id
            || scheduling.task.pool_id != worker.pool_id
        {
            return Err("attempt is outside live worker authority".into());
        }
        if state.cancelled_attempts.contains_key(&scheduling.id) {
            return Err("attempt was cancelled before execution".into());
        }
        match &program {
            RemoteAttemptProgram::GraphStaged(RemoteStagedGraphReference::StagedGraphV10 {
                program_digest,
                program_length,
            }) => {
                let staged = state
                    .graph_programs
                    .get(&scheduling.id)
                    .cloned()
                    .ok_or_else(|| "Graph Program is not staged by this worker".to_string())?;
                if staged.digest() != *program_digest || staged.length() != *program_length {
                    return Err("staged Graph Program identity changed".into());
                }
                Some(staged)
            }
            RemoteAttemptProgram::Direct(_) => None,
        }
    };
    let (program, graph_wire_bytes) = match (program, staged) {
        (RemoteAttemptProgram::Direct(direct), None) => {
            (WorkerProgramRequest::Direct(direct), None)
        }
        (RemoteAttemptProgram::GraphStaged(_), Some(staged)) => {
            let wire_bytes = usize::try_from(staged.length())
                .map_err(|_| "staged Graph Program exceeds process ABI".to_string())?;
            let mut graph = tokio::task::spawn_blocking(move || {
                let object: runmat_execution_artifact::RemoteProgramObject =
                    crate::local_store::read_verified_json_file(
                        staged.private_path(),
                        &staged.digest(),
                        staged.length(),
                        runmat_execution::resource::MAX_GRAPH_PROGRAM_WIRE_BYTES,
                    )
                    .map_err(|error| error.to_string())?;
                match object {
                    runmat_execution_artifact::RemoteProgramObject::Graph(graph) => {
                        Ok::<_, String>(*graph)
                    }
                    runmat_execution_artifact::RemoteProgramObject::Direct(_) => {
                        Err("staged Graph reference contains a Direct Program".into())
                    }
                }
            })
            .await
            .map_err(|error| error.to_string())??;
            if graph.assignment.is_some() {
                return Err("staged Graph Program already has an assignment".into());
            }
            graph.assignment = Some(runmat_execution::ProgramExecutionAssignment {
                scope_id: scheduling.scope_id,
                pool_id: scheduling.task.pool_id,
                task_id: scheduling.task_id,
                attempt_id: scheduling.id,
                worker_id: scheduling.worker_id,
                backend: runmat_execution::PoolBackend::Remote,
                resources: scheduling.resource_assignment.clone(),
            });
            (WorkerProgramRequest::graph(graph), Some(wire_bytes))
        }
        _ => return Err("remote attempt program form is inconsistent".into()),
    };
    Ok(ResolvedRemoteAttempt {
        scheduling,
        program,
        graph_wire_bytes,
    })
}

async fn validate_and_reserve_attempt(
    state: &Mutex<WorkerState>,
    worker: &WorkerSpec,
    driver_fence: u64,
    attempt: &ResolvedRemoteAttempt,
) -> Option<String> {
    let mut state = state.lock().await;
    if state.draining
        || attempt.scheduling.driver_fence != driver_fence
        || attempt.scheduling.worker_id != worker.id
        || attempt.scheduling.task.pool_id != worker.pool_id
    {
        return Some("attempt is outside live worker authority".into());
    }
    if state.attempts.contains_key(&attempt.scheduling.id) {
        return Some("attempt is already active on the worker".into());
    }
    if attempt
        .scheduling
        .task
        .validate_invocation_reservation()
        .is_err()
    {
        return Some("attempt invocation context exceeds its reservation".into());
    }
    match &attempt.program {
        WorkerProgramRequest::Direct(program) => {
            if let Err(message) = super::direct_admission::validate(
                program,
                super::direct_admission::Admission::Worker {
                    scheduling: &attempt.scheduling,
                    worker,
                },
            ) {
                return Some(message.into());
            }
            let Some(bundle) = state.bundle.as_ref() else {
                return Some("exact execution bundle is not installed".into());
            };
            if bundle.requires_materialization() != state.materialized_project.is_some()
                || bundle.manifest.portable_environment != program.environment.entries()
                || !bundle
                    .manifest
                    .recipes
                    .iter()
                    .any(|value| value == &program.recipe)
                || !bundle
                    .manifest
                    .artifacts
                    .iter()
                    .any(|value| value == &program.artifact)
            {
                return Some("attempt program is not authorized by the installed bundle".into());
            }
        }
        WorkerProgramRequest::Graph(graph) => {
            if let Err(message) = graph_admission::validate(
                &attempt.scheduling,
                graph,
                attempt.graph_wire_bytes,
                worker,
            ) {
                return Some(message.into());
            }
        }
    }
    reserve_attempt_capacity(
        &worker.resources,
        &mut state.allocated,
        &attempt.scheduling.task.resources,
        &attempt.scheduling.resource_assignment,
    )
    .err()
    .map(str::to_owned)
}

fn reserve_attempt_capacity(
    inventory: &runmat_execution::resource::ResourceInventory,
    allocated: &mut ResourceAllocation,
    request: &ResourceRequest,
    assignment: &runmat_execution::resource::ResourceAssignment,
) -> Result<(), &'static str> {
    request
        .validate()
        .map_err(|_| "attempt has an invalid resource request")?;
    if !fits(inventory, allocated, request) {
        return Err("attempt resources exceed the worker's available inventory");
    }
    // `reserve` can fail after a preceding field was updated; commit only a complete reservation.
    let mut next = allocated.clone();
    reserve(&mut next, request, assignment)
        .map_err(|_| "attempt resource reservation overflowed")?;
    *allocated = next;
    Ok(())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

mod graph_admission;

#[cfg(test)]
mod tests;
