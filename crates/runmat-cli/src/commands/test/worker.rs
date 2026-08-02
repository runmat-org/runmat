use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use runmat_core::RunMatSession;
use runmat_test::discovery::FrozenTestRunSnapshot;
use runmat_test::plan::TestPlan;
use runmat_test::protocol::{
    negotiate, ProtocolHandshake, ProtocolLimits, WorkerCapability, WorkerRequest, WorkerResponse,
};
use runmat_test_runner_native::transport::{read_bootstrap, read_request, write_response};
use tokio::io::{stdin, stdout, BufReader, BufWriter};

struct InstalledRun {
    plan: TestPlan,
    snapshot: FrozenTestRunSnapshot,
    session: RunMatSession,
}

pub async fn run_stdio() -> Result<()> {
    let mut input = BufReader::new(stdin());
    let mut output = BufWriter::new(stdout());
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
        .context("failed to read coordinator handshake")?;
    let remote = match first {
        WorkerRequest::Handshake(handshake) => handshake,
        request => return Err(anyhow!("expected handshake, received {request:?}")),
    };
    let limits = negotiate(&local, &remote).context("worker handshake is incompatible")?;
    write_response(
        &mut output,
        &WorkerResponse::Handshake(local.clone()),
        limits,
    )
    .await?;
    let bootstrap = read_bootstrap(&mut input, limits)
        .await
        .context("failed to read native worker bootstrap")?;

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
                let mut session = RunMatSession::with_options(enable_jit, false)
                    .context("failed to initialize worker session")?;
                if let Some(project) = bootstrap.project.clone() {
                    session
                        .install_project_handoff(project)
                        .context("failed to install frozen worker project")?;
                }
                let run_id = plan.run_id.clone();
                installed = Some(InstalledRun {
                    plan,
                    snapshot,
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
) -> Result<()> {
    write_response(
        output,
        &WorkerResponse::Rejected {
            code: code.into(),
            message: message.into(),
        },
        limits,
    )
    .await
    .map_err(anyhow::Error::from)
}
