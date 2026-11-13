use runmat_core::RunMatSession;
use runmat_test::discovery::FrozenTestRunSnapshot;
use runmat_test::plan::TestPlan;
use runmat_test::protocol::{
    negotiate, ProtocolHandshake, ProtocolLimits, WorkerCapability, WorkerRequest, WorkerResponse,
};

use crate::transport::decode_bootstrap;
use crate::{NativeRunnerError, NativeRunnerResult};
use runmat_execution_runner_native::fea_proxy;
use runmat_process_host::ipc::private_control::ControlFrameWriter;
use runmat_process_host::ipc::{FrameLimits, WorkerParentLink};
use runmat_runtime::analysis::fea_run_admission::*;
use std::sync::Arc;

struct InstalledRun {
    plan: TestPlan,
    snapshot: FrozenTestRunSnapshot,
    session: RunMatSession,
}

/// Serve native Core tests over their private control channels. Standard
/// streams carry only application output, captured by the process owner.
///
/// Native composition roots call this before starting their ordinary host
/// runtime when launched with their private worker marker. The coordinator
/// remains transport-agnostic, while CLI and Desktop reuse this exact Core
/// execution adapter instead of implementing independent child loops.
pub async fn run_core_worker_stdio() -> NativeRunnerResult<()> {
    let control = runmat_process_host::ipc::private_control::connect_host_blocking()
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    let local = ProtocolHandshake::current(
        "runmat-native-worker",
        vec![
            WorkerCapability::StrongIsolation,
            WorkerCapability::CapturedOutput,
            WorkerCapability::Artifacts,
            WorkerCapability::Coverage,
        ],
    );
    let frame_limits = FrameLimits {
        max_message_bytes: local.limits.max_message_bytes,
    };
    let replies = Arc::new(fea_proxy::Replies::default());
    let observer_replies = replies.clone();
    let active_cancel = Arc::new(std::sync::Mutex::new(
        None::<(
            runmat_test::identity::RunId,
            runmat_execution::cancellation::CancellationSource,
        )>,
    ));
    let observer_cancel = active_cancel.clone();
    let observer_limits = local.limits;
    let (parent_link, mut frames) = WorkerParentLink::start_bounded_with_observer(
        control.reader,
        frame_limits,
        Some(Arc::new(move |bytes| {
            if observer_replies.observe(bytes) {
                return true;
            }
            #[derive(serde::Deserialize)]
            #[serde(tag = "type", rename_all = "snake_case")]
            enum CancelProbe {
                Cancel {},
            }
            if matches!(
                serde_json::from_slice::<CancelProbe>(bytes),
                Ok(CancelProbe::Cancel {})
            ) {
                if let Ok(WorkerRequest::Cancel { run_id, .. }) =
                    runmat_test::protocol::decode_request(bytes, observer_limits)
                {
                    if let Some((active, cancellation)) =
                        &*observer_cancel.lock().unwrap_or_else(|e| e.into_inner())
                    {
                        if *active == run_id {
                            cancellation.cancel(runmat_execution::CancellationReason::User);
                        }
                    }
                }
            }
            false
        })),
    )
    .map_err(|error| protocol_error("failed to monitor coordinator", error))?;
    replies.activate_reader();
    let mut output = ControlFrameWriter::new(control.writer, frame_limits)
        .map_err(|error| protocol_error("failed to open coordinator writer", error))?;
    let first = frames
        .recv()
        .await
        .ok_or_else(|| NativeRunnerError::Protocol("coordinator closed before handshake".into()))?
        .map_err(|error| protocol_error("failed to read coordinator handshake", error))?;
    let first = runmat_test::protocol::decode_request(&first, local.limits)
        .map_err(|error| protocol_error("invalid coordinator handshake", error))?;
    let remote = match first {
        WorkerRequest::Handshake(handshake) => handshake,
        request => {
            return Err(NativeRunnerError::Protocol(format!(
                "expected handshake, received {request:?}"
            )));
        }
    };
    let limits = negotiate(&local, &remote)
        .map_err(|error| protocol_error("worker handshake is incompatible", error))?;
    write_response(&output, &WorkerResponse::Handshake(local.clone()), limits).await?;
    let bootstrap = frames
        .recv()
        .await
        .ok_or_else(|| NativeRunnerError::Protocol("coordinator closed before bootstrap".into()))?
        .map_err(|error| protocol_error("failed to read native worker bootstrap", error))?;
    if bootstrap.len() > limits.max_message_bytes as usize {
        return Err(NativeRunnerError::Protocol(
            "bootstrap exceeds negotiated limit".into(),
        ));
    }
    let bootstrap = decode_bootstrap(&bootstrap)?;
    runmat_accelerate::initialize_acceleration_provider_with(
        &(&bootstrap.execution.accelerate).into(),
    );
    write_response(&output, &WorkerResponse::BootstrapReady, limits).await?;
    let mut input = super::control_requests::ControlRequests::from_frames(frames, limits);

    let mut installed: Option<InstalledRun> = None;
    loop {
        let request = input.next().await?;
        match request {
            WorkerRequest::Handshake(_) => {
                reject(
                    &mut output,
                    limits,
                    "duplicate_handshake",
                    "worker handshake is already complete",
                )
                .await?;
            }
            WorkerRequest::InstallPlan { plan, snapshot } => {
                if plan.program_revision != snapshot.program_revision {
                    reject(
                        &mut output,
                        limits,
                        "revision_mismatch",
                        "plan and frozen source revisions differ",
                    )
                    .await?;
                    continue;
                }
                if let Some(mut previous) = installed.take() {
                    previous
                        .session
                        .shutdown_foreign_runtime()
                        .await
                        .map_err(|error| {
                            protocol_error("failed to shut down the previous worker session", error)
                        })?;
                }
                let mut session =
                    RunMatSession::with_options(bootstrap.execution.enable_jit, false).map_err(
                        |error| protocol_error("failed to initialize worker session", error),
                    )?;
                session
                    .configure_test_snapshot(&snapshot)
                    .map_err(|error| protocol_error("invalid frozen test environment", error))?;
                session
                    .set_foreign_config(&bootstrap.execution.foreign)
                    .map_err(|error| protocol_error("invalid worker foreign policy", error))?;
                if let Some(project) = bootstrap.project.clone() {
                    let project = issue_local_project_paths(project)?;
                    session
                        .install_project_handoff(project)
                        .await
                        .map_err(|error| {
                            protocol_error("failed to install frozen worker project", error)
                        })?;
                }
                let run_id = plan.run_id.clone();
                installed = Some(InstalledRun {
                    plan: *plan,
                    snapshot: *snapshot,
                    session,
                });
                write_response(&output, &WorkerResponse::Ready { run_id }, limits).await?;
            }
            WorkerRequest::Execute {
                test_id,
                attempt,
                admission,
            } => {
                let Some(run) = installed.as_mut() else {
                    reject(
                        &mut output,
                        limits,
                        "plan_not_installed",
                        "install a plan before executing tests",
                    )
                    .await?;
                    continue;
                };
                let provider: Arc<dyn FeaRunAdmission> = if let Some(admission) = admission {
                    admission
                        .assignment
                        .validate()
                        .map_err(|error| protocol_error("invalid test assignment", error))?;
                    admission
                        .assignment
                        .resources
                        .validate_for_request(&admission.resources)
                        .map_err(|error| protocol_error("invalid test resources", error))?;
                    admission
                        .resources
                        .validate()
                        .map_err(|error| protocol_error("invalid test resource request", error))?;
                    Arc::new(fea_proxy::WorkerAdmission(fea_proxy::Client::new(
                        admission.assignment,
                        u64::from(admission.resources.cpu_millicores),
                        output.clone(),
                        replies.clone(),
                    )))
                } else {
                    Arc::new(UnadmittedTestWorker)
                };
                let admission = FeaRunAdmissionContext::new(provider);
                let cancellation = runmat_execution::cancellation::CancellationSource::new();
                *active_cancel.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some((run.plan.run_id.clone(), cancellation.clone()));
                let execution = admission.run_future(run.session.execute_planned_test(
                    &run.snapshot,
                    &run.plan,
                    &test_id,
                    attempt,
                    cancellation.token(),
                ));
                tokio::pin!(execution);
                let result = loop {
                    tokio::select! {
                        result = &mut execution => break result,
                        control = input.next() => {
                            match control? {
                                WorkerRequest::Cancel { run_id, .. }
                                    if run_id == run.plan.run_id =>
                                {
                                    cancellation.cancel(runmat_execution::CancellationReason::User);
                                }
                                WorkerRequest::Cancel { .. } => {
                                    reject(
                                        &mut output,
                                        limits,
                                        "wrong_run",
                                        "cancellation targeted a different run",
                                    )
                                    .await?;
                                }
                                _ => {
                                    reject(
                                        &mut output,
                                        limits,
                                        "worker_busy",
                                        "worker accepts only cancellation while a test is active",
                                    )
                                    .await?;
                                }
                            }
                        }
                    }
                };
                active_cancel
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
                match result {
                    Ok(attempt) => {
                        for event in attempt.events {
                            write_response(&output, &WorkerResponse::Event { event }, limits)
                                .await?;
                        }
                        write_response(
                            &output,
                            &WorkerResponse::Completed {
                                result: attempt.result,
                                coverage: attempt.coverage,
                            },
                            limits,
                        )
                        .await?;
                    }
                    Err(error) => {
                        reject(&mut output, limits, "execution_failed", &error.to_string()).await?;
                    }
                }
            }
            WorkerRequest::Cancel { .. } => {
                reject(
                    &mut output,
                    limits,
                    "no_active_test",
                    "worker has no active test to cancel",
                )
                .await?;
            }
            WorkerRequest::Shutdown => {
                if let Some(mut run) = installed.take() {
                    run.session
                        .shutdown_foreign_runtime()
                        .await
                        .map_err(|error| {
                            protocol_error("failed to shut down the worker session", error)
                        })?;
                }
                parent_link.disarm();
                write_response(&output, &WorkerResponse::ShutdownComplete, limits).await?;
                return Ok(());
            }
        }
    }
}

/// The bootstrap identifies a project revision, not a portable access root.
/// Issue the native worker's own local paths before Core derives attribution
/// for frozen test files. The selected source bytes still come from the run
/// snapshot; this step never rereads a test body.
fn issue_local_project_paths(
    mut handoff: runmat_package::FrozenProjectHandoff,
) -> NativeRunnerResult<runmat_package::FrozenProjectHandoff> {
    let root = std::fs::canonicalize(".")
        .map_err(|error| protocol_error("native worker cannot issue its project root", error))?;
    let original_root = handoff.project.workspace_root.clone();
    let manifest_relative = handoff
        .project
        .manifest_path
        .strip_prefix(&original_root)
        .map_err(|_| {
            NativeRunnerError::Protocol("native manifest is outside project root".into())
        })?;
    let manifest = std::fs::canonicalize(root.join(manifest_relative)).map_err(|error| {
        protocol_error("native worker cannot locate its project manifest", error)
    })?;
    if manifest.parent() != Some(root.as_path()) {
        return Err(NativeRunnerError::Protocol(
            "native worker manifest is outside its project root".into(),
        ));
    }
    for path in handoff.project.access_paths.values_mut() {
        let issued = path
            .strip_prefix(&original_root)
            .map_or_else(|_| path.clone(), |relative| root.join(relative));
        *path = std::fs::canonicalize(issued)
            .map_err(|error| protocol_error("native worker cannot locate project source", error))?;
    }
    handoff.project.workspace_root = root;
    handoff.project.manifest_path = manifest;
    handoff
        .validate()
        .map_err(|error| protocol_error("native worker project remap is invalid", error))?;
    Ok(handoff)
}

async fn reject(
    output: &mut ControlFrameWriter,
    limits: ProtocolLimits,
    code: &str,
    message: &str,
) -> NativeRunnerResult<()> {
    write_response(
        output,
        &WorkerResponse::Rejected {
            code: code.into(),
            message: message.into(),
        },
        limits,
    )
    .await
}

struct UnadmittedTestWorker;
impl FeaRunAdmission for UnadmittedTestWorker {
    fn admit(&self, _: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        Err(FeaRunDenied::LicenseInvalid)
    }
}

async fn write_response(
    output: &ControlFrameWriter,
    response: &WorkerResponse,
    limits: ProtocolLimits,
) -> NativeRunnerResult<()> {
    let payload = runmat_test::protocol::encode_response(response, limits)
        .map_err(|error| protocol_error("invalid worker response", error))?;
    output
        .write_owned(payload)
        .await
        .map_err(|error| protocol_error("failed to write worker response", error))
}

fn protocol_error(context: &str, error: impl std::fmt::Display) -> NativeRunnerError {
    NativeRunnerError::Protocol(format!("{context}: {error}"))
}
