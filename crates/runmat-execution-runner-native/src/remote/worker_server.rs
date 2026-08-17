use std::collections::HashMap;
use std::sync::Arc;

use runmat_execution::identity::AttemptId;
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_artifact::ExecutionBundle;
use runmat_execution_runner::{AttemptFailureKind, AttemptReport, AttemptSuccess, WorkerSpec};
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameKind, FrameLimits};
use runmat_execution_transport_native::overlay::{QuicOverlayListener, WebSocketRelayConnection};
use tokio::sync::Mutex;

use super::protocol::{
    RemoteWorkerCommand, RemoteWorkerOutcome, RemoteWorkerReply, RemoteWorkerRequest,
    REMOTE_WORKER_PROTOCOL_V2,
};
use super::route::{QuicFrameRoute, RelayFrameRoute, RemoteFrameRoute};
use super::RemoteAttempt;
use crate::{NativeExecutionError, NativeExecutionResult};

struct WorkerState {
    bundle: Option<ExecutionBundle>,
    materialized_project: Option<Arc<crate::materialized_project::MaterializedProject>>,
    bundle_digest: Option<runmat_execution::Digest>,
    attempts: HashMap<AttemptId, tokio::task::JoinHandle<()>>,
    values: HashMap<runmat_execution::identity::ValueId, runmat_execution::value::ValuePayload>,
    objects: super::object_transfer::RemoteObjectCache,
    draining: bool,
    bundle_cache: Option<std::path::PathBuf>,
}

struct WorkerLoopContext {
    run_identity: String,
    worker: WorkerSpec,
    driver_fence: u64,
    session_id: [u8; 16],
    run_key: RunKeyMaterial,
    limits: FrameLimits,
    bundle_cache: Option<std::path::PathBuf>,
}

pub async fn run_remote_worker_quic(
    listener: QuicOverlayListener,
    run_identity: impl Into<String>,
    worker: WorkerSpec,
    driver_fence: u64,
    session_id: [u8; 16],
    run_key: RunKeyMaterial,
    limits: FrameLimits,
) -> NativeExecutionResult<()> {
    let connection = Arc::new(listener.accept().await.map_err(protocol)?);
    tokio::task::LocalSet::new()
        .run_until(run_worker_loop(
            Arc::new(QuicFrameRoute(connection)),
            WorkerLoopContext {
                run_identity: run_identity.into(),
                worker,
                driver_fence,
                session_id,
                run_key,
                limits,
                bundle_cache: None,
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
            },
        ))
        .await
}

async fn run_worker_loop(
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
    } = context;
    let mut receiver = EncryptedFrameSession::new(
        run_identity.clone(),
        session_id,
        "driver-to-worker",
        1,
        run_key.clone(),
    )
    .map_err(protocol)?;
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
    let state = Arc::new(Mutex::new(WorkerState {
        bundle: None,
        materialized_project: None,
        bundle_digest: None,
        attempts: HashMap::new(),
        values: HashMap::new(),
        objects: super::object_transfer::RemoteObjectCache::default(),
        draining: false,
        bundle_cache,
    }));
    let drain_complete = Arc::new(tokio::sync::Notify::new());
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
            frame = connection.receive() => frame?,
            _ = notified => continue,
        };
        if !matches!(frame.kind, FrameKind::Control | FrameKind::Artifact) {
            return Err(protocol(
                "remote worker received an unsupported command frame",
            ));
        }
        let plaintext = receiver.open(&frame, limits).map_err(protocol)?;
        let request: RemoteWorkerRequest = serde_json::from_slice(&plaintext).map_err(protocol)?;
        if command_frame_kind(&request.command) != frame.kind {
            return Err(protocol(
                "remote worker command used the wrong encrypted frame kind",
            ));
        }
        if request.schema_version != REMOTE_WORKER_PROTOCOL_V2
            || request.driver_fence != driver_fence
        {
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
                        schema_version: REMOTE_WORKER_PROTOCOL_V2,
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
                        schema_version: REMOTE_WORKER_PROTOCOL_V2,
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
                        schema_version: REMOTE_WORKER_PROTOCOL_V2,
                        correlation_id: request.correlation_id,
                        outcome,
                    },
                )
                .await?;
            }
            RemoteWorkerCommand::Execute { attempt } => {
                let rejection = validate_attempt(&state, &worker, driver_fence, &attempt).await;
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
                let attempt = *attempt;
                let attempt_id = attempt.scheduling.id;
                let correlation_id = request.correlation_id;
                let connection = Arc::clone(&connection);
                let sender = Arc::clone(&sender);
                let state_for_task = Arc::clone(&state);
                let materialized_project =
                    state.lock().await.materialized_project.as_ref().cloned();
                let drain_complete_for_task = Arc::clone(&drain_complete);
                let task = tokio::task::spawn_local(async move {
                    let mut program = attempt.program;
                    let materialized = {
                        let state = state_for_task.lock().await;
                        program
                            .arguments
                            .iter()
                            .map(|value| super::value_transfer::materialize(value, &state.values))
                            .collect::<NativeExecutionResult<Vec<_>>>()
                    };
                    let response = match materialized {
                        Ok(arguments) => {
                            program.arguments = arguments;
                            crate::execute_host_program_request_with_project(
                                program,
                                materialized_project
                                    .as_deref()
                                    .map(crate::materialized_project::MaterializedProject::handoff),
                            )
                            .await
                        }
                        Err(error) => {
                            runmat_execution_artifact::ProgramExecutionResponse::Failure {
                                message: error.to_string(),
                            }
                        }
                    };
                    let report = match response {
                        runmat_execution_artifact::ProgramExecutionResponse::Success { value } => {
                            AttemptReport::Succeeded {
                                result: AttemptSuccess {
                                    outputs: vec![value],
                                    result_objects: Vec::new(),
                                },
                            }
                        }
                        runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
                            ..
                        } => AttemptReport::Failed {
                            kind: AttemptFailureKind::Rejected,
                            message: "remote externalized results require verified artifact transfer"
                                .into(),
                        },
                        runmat_execution_artifact::ProgramExecutionResponse::Failure {
                            message,
                        } => AttemptReport::Failed {
                            kind: AttemptFailureKind::Execution,
                            message,
                        },
                    };
                    let _ = reply(
                        connection.as_ref(),
                        &sender,
                        limits,
                        RemoteWorkerReply {
                            schema_version: REMOTE_WORKER_PROTOCOL_V2,
                            correlation_id,
                            outcome: RemoteWorkerOutcome::Attempt { report },
                        },
                    )
                    .await;
                    let mut state = state_for_task.lock().await;
                    state.attempts.remove(&attempt_id);
                    if state.draining && state.attempts.is_empty() {
                        drain_complete_for_task.notify_waiters();
                    }
                });
                state.lock().await.attempts.insert(attempt_id, task);
            }
            RemoteWorkerCommand::Cancel { attempt_id } => {
                if let Some(task) = state.lock().await.attempts.remove(&attempt_id) {
                    task.abort();
                }
                reply(
                    connection.as_ref(),
                    &sender,
                    limits,
                    acknowledged(request.correlation_id),
                )
                .await?;
            }
            RemoteWorkerCommand::Drain => {
                let mut state = state.lock().await;
                state.draining = true;
                let drained = state.attempts.is_empty();
                drop(state);
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
                let outcome = super::object_transfer::handle_command(
                    &mut state.lock().await.objects,
                    command,
                    &run_identity,
                );
                reply_kind(
                    connection.as_ref(),
                    &sender,
                    limits,
                    FrameKind::Artifact,
                    RemoteWorkerReply {
                        schema_version: REMOTE_WORKER_PROTOCOL_V2,
                        correlation_id: request.correlation_id,
                        outcome,
                    },
                )
                .await?;
            }
        }
    }
}

async fn validate_attempt(
    state: &Mutex<WorkerState>,
    worker: &WorkerSpec,
    driver_fence: u64,
    attempt: &RemoteAttempt,
) -> Option<String> {
    let state = state.lock().await;
    if state.draining
        || attempt.scheduling.driver_fence != driver_fence
        || attempt.scheduling.worker_id != worker.id
    {
        return Some("attempt is outside live worker authority".into());
    }
    let Some(bundle) = state.bundle.as_ref() else {
        return Some("exact execution bundle is not installed".into());
    };
    if bundle.requires_source_project() != state.materialized_project.is_some()
        || attempt.program.validate_for_portable_host().is_err()
        || !bundle
            .manifest
            .recipes
            .iter()
            .any(|value| value == &attempt.program.recipe)
        || !bundle
            .manifest
            .artifacts
            .iter()
            .any(|value| value == &attempt.program.artifact)
    {
        return Some("attempt program is not authorized by the installed bundle".into());
    }
    None
}

async fn reply(
    connection: &dyn RemoteFrameRoute,
    sender: &Mutex<EncryptedFrameSession>,
    limits: FrameLimits,
    reply: RemoteWorkerReply,
) -> NativeExecutionResult<()> {
    reply_kind(connection, sender, limits, FrameKind::Control, reply).await
}

async fn reply_kind(
    connection: &dyn RemoteFrameRoute,
    sender: &Mutex<EncryptedFrameSession>,
    limits: FrameLimits,
    kind: FrameKind,
    reply: RemoteWorkerReply,
) -> NativeExecutionResult<()> {
    let plaintext = serde_json::to_vec(&reply).map_err(protocol)?;
    let frame = sender
        .lock()
        .await
        .seal(kind, &plaintext, limits)
        .map_err(protocol)?;
    connection.send(frame).await
}

fn command_frame_kind(command: &RemoteWorkerCommand) -> FrameKind {
    match command {
        RemoteWorkerCommand::ProbeObject { .. }
        | RemoteWorkerCommand::PutObjectChunk { .. }
        | RemoteWorkerCommand::GetObjectChunk { .. } => FrameKind::Artifact,
        _ => FrameKind::Control,
    }
}

fn acknowledged(correlation_id: String) -> RemoteWorkerReply {
    RemoteWorkerReply {
        schema_version: REMOTE_WORKER_PROTOCOL_V2,
        correlation_id,
        outcome: RemoteWorkerOutcome::Acknowledged,
    }
}

fn rejected(correlation_id: String, message: impl Into<String>) -> RemoteWorkerReply {
    RemoteWorkerReply {
        schema_version: REMOTE_WORKER_PROTOCOL_V2,
        correlation_id,
        outcome: RemoteWorkerOutcome::Rejected {
            message: message.into(),
        },
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
