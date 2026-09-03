use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use runmat_core::RunMatSession;
use runmat_test::discovery::FrozenTestRunSnapshot;
use runmat_test::plan::TestPlan;
use runmat_test::protocol::{
    negotiate, ProtocolHandshake, ProtocolLimits, WorkerCapability, WorkerRequest, WorkerResponse,
};

use crate::transport::{read_bootstrap, read_request, write_response};
use crate::{NativeRunnerError, NativeRunnerResult};

struct InstalledRun {
    plan: TestPlan,
    snapshot: FrozenTestRunSnapshot,
    session: RunMatSession,
}

/// Serve the native RunMat Core test-worker protocol over stdin/stdout.
///
/// Native composition roots call this before starting their ordinary host
/// runtime when launched with their private worker marker. The coordinator
/// remains transport-agnostic, while CLI and Desktop reuse this exact Core
/// execution adapter instead of implementing independent child loops.
pub async fn run_core_worker_stdio() -> NativeRunnerResult<()> {
    let (mut input, mut output) = runmat_process_host::ipc::stdio::endpoint()
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
    let first = read_request(&mut input, local.limits)
        .await
        .map_err(|error| protocol_error("failed to read coordinator handshake", error))?;
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
    write_response(
        &mut output,
        &WorkerResponse::Handshake(local.clone()),
        limits,
    )
    .await?;
    let bootstrap = read_bootstrap(&mut input, limits)
        .await
        .map_err(|error| protocol_error("failed to read native worker bootstrap", error))?;

    let mut installed: Option<InstalledRun> = None;
    loop {
        let request = read_request(&mut input, limits).await?;
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
                let enable_jit = std::env::var("RUNMAT_TEST_JIT").as_deref() != Ok("0");
                let mut session =
                    RunMatSession::with_options(enable_jit, false).map_err(|error| {
                        protocol_error("failed to initialize worker session", error)
                    })?;
                if let Some(project) = bootstrap.project.clone() {
                    session.install_project_handoff(project).map_err(|error| {
                        protocol_error("failed to install frozen worker project", error)
                    })?;
                }
                let run_id = plan.run_id.clone();
                installed = Some(InstalledRun {
                    plan: *plan,
                    snapshot: *snapshot,
                    session,
                });
                write_response(&mut output, &WorkerResponse::Ready { run_id }, limits).await?;
            }
            WorkerRequest::Execute { test_id, attempt } => {
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
                let cancellation = Arc::new(AtomicBool::new(false));
                let execution = run.session.execute_planned_test(
                    &run.snapshot,
                    &run.plan,
                    &test_id,
                    attempt,
                    cancellation.clone(),
                );
                tokio::pin!(execution);
                let result = loop {
                    tokio::select! {
                        result = &mut execution => break result,
                        control = read_request(&mut input, limits) => {
                            match control? {
                                WorkerRequest::Cancel { run_id, .. }
                                    if run_id == run.plan.run_id =>
                                {
                                    cancellation.store(true, Ordering::Relaxed);
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
                match result {
                    Ok(attempt) => {
                        for event in attempt.events {
                            write_response(&mut output, &WorkerResponse::Event { event }, limits)
                                .await?;
                        }
                        write_response(
                            &mut output,
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
                write_response(&mut output, &WorkerResponse::ShutdownComplete, limits).await?;
                return Ok(());
            }
        }
    }
}

async fn reject(
    output: &mut (impl tokio::io::AsyncWrite + Unpin),
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

fn protocol_error(context: &str, error: impl std::fmt::Display) -> NativeRunnerError {
    NativeRunnerError::Protocol(format!("{context}: {error}"))
}
