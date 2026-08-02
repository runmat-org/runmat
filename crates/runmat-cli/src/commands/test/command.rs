use std::collections::BTreeSet;

use anyhow::{Context, Result};
use runmat_config::project::{ProjectTestIsolation, ProjectTestReport};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_test::discovery::DiscoveryDiagnosticSeverity;
use runmat_test::result::TerminalDisposition;
use runmat_test_runner::artifact::persist_reports;
use runmat_test_runner::host::IsolationMode;
use runmat_test_runner::reporter::{
    HumanReporter, JsonReporter, JunitReporter, ReporterFanout, TapReporter,
};
use runmat_test_runner::schedule::RetryPolicy;
use runmat_test_runner::worker::{RunSubmission, WorkerBackend};
use runmat_test_runner::{CoordinatedRun, Coordinator, CoordinatorConfig};
use runmat_test_runner_native::artifact::FilesystemArtifactStore;
use runmat_test_runner_native::host::{NativeCancellation, NativeClock};
use runmat_test_runner_native::telemetry::NativeTelemetry;
use runmat_test_runner_native::{
    LocalBackend, LocalBackendConfig, ProcessBackend, ProcessBackendConfig,
};

use crate::cli::{Cli, TestArgs, TestIsolationArg, TestReportArg};
use crate::presentation;

use super::discovery::prepare;
use super::exit::TestCommandError;

pub async fn execute(args: TestArgs, cli: &Cli, _runtime: &RunMatRuntimeConfig) -> Result<()> {
    match execute_inner(args, cli).await {
        Ok(()) => Ok(()),
        Err(error) if error.downcast_ref::<TestCommandError>().is_some() => Err(error),
        Err(error) => {
            eprintln!("{}: {error:#}", presentation::stderr().error("Error"));
            Err(TestCommandError::new(2).into())
        }
    }
}

async fn execute_inner(args: TestArgs, cli: &Cli) -> Result<()> {
    let prepared = prepare(&args, cli).await?;
    let mut session = runmat_core::RunMatSession::with_options(!cli.no_jit, false)
        .context("failed to initialize test discovery session")?;
    if let Some(project) = prepared.project_handoff.clone() {
        session
            .install_project_handoff(project)
            .context("failed to install frozen project for test discovery")?;
    }
    let discovery = session
        .discover_tests(&prepared.snapshot)
        .context("test discovery failed")?;
    for diagnostic in &discovery.diagnostics {
        let source = diagnostic
            .source
            .as_ref()
            .map(|source| format!("{}: ", source.relative_path))
            .unwrap_or_default();
        match diagnostic.severity {
            DiscoveryDiagnosticSeverity::Error => eprintln!(
                "{}: {source}{} ({})",
                presentation::stderr().error("Error"),
                diagnostic.message,
                diagnostic.code
            ),
            DiscoveryDiagnosticSeverity::Warning => eprintln!(
                "{}: {source}{} ({})",
                presentation::stderr().warning("Warning"),
                diagnostic.message,
                diagnostic.code
            ),
            DiscoveryDiagnosticSeverity::Information => {
                eprintln!("Info: {source}{} ({})", diagnostic.message, diagnostic.code);
            }
        }
    }
    if discovery
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == DiscoveryDiagnosticSeverity::Error)
    {
        anyhow::bail!("test discovery produced errors");
    }
    let selected = discovery.select(&prepared.selector);
    if selected.suites.iter().all(|suite| suite.tests.is_empty()) {
        anyhow::bail!("no tests matched the requested selection");
    }
    let invocation = serde_json::to_string(&prepared.selector)
        .context("failed to encode deterministic test selection")?;
    let plan = selected
        .into_plan(invocation)
        .context("failed to materialize the selected test plan")?;
    if args.list {
        for test in plan.tests() {
            println!(
                "{}\t{}\t{}",
                test.id.as_str(),
                test.display_name,
                test.procedure.source.relative_path
            );
        }
        return Ok(());
    }

    let isolation = isolation(&args, prepared.test_config.isolation)?;
    let allowed_environment = prepared
        .test_config
        .environment_allowlist
        .iter()
        .chain(args.environment_allowlist.iter())
        .cloned()
        .collect::<BTreeSet<_>>();
    let cancellation = NativeCancellation::default();
    let signal_cancellation = cancellation.clone();
    let signal_task = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_cancellation.cancel("interrupted by user");
        }
    });
    let mut reporters = reporters(&args, &prepared.test_config.reports);
    let jobs = args.jobs.or(prepared.test_config.jobs).unwrap_or(1);
    let config = CoordinatorConfig {
        isolation,
        jobs,
        timeout_ms: args.timeout_ms.or(prepared.test_config.timeout_ms),
        cancellation_grace_ms: 1_000,
        retry: RetryPolicy { max_attempts: 1 },
        shard_index: args
            .shard_index
            .or(prepared.test_config.shard.map(|shard| shard.index)),
        shard_count: args
            .shard_count
            .or(prepared.test_config.shard.map(|shard| shard.count)),
    };
    let submission = RunSubmission::new(plan, prepared.snapshot)?;
    let coordinator = Coordinator::new(config)?;
    let run = match isolation {
        IsolationMode::Auto | IsolationMode::Process => {
            let mut backend_config = ProcessBackendConfig::same_binary(
                std::env::current_exe().context("failed to locate the runmat executable")?,
            );
            backend_config.project_handoff = prepared.project_handoff.clone();
            backend_config.environment.insert(
                "RUNMAT_TEST_JIT".into(),
                if cli.no_jit { "0" } else { "1" }.into(),
            );
            for name in allowed_environment {
                if let Ok(value) = std::env::var(&name) {
                    backend_config.environment.insert(name, value);
                }
            }
            let backend = ProcessBackend::new(backend_config)
                .map_err(|error| anyhow::anyhow!("failed to configure native workers: {error}"))?;
            run_coordinator(
                &coordinator,
                submission,
                &backend,
                &cancellation,
                &mut reporters,
            )
            .await?
        }
        IsolationMode::Session | IsolationMode::None => {
            let mut backend_config = LocalBackendConfig::new(isolation);
            backend_config.enable_jit = !cli.no_jit;
            backend_config.max_workers = jobs;
            backend_config.project_handoff = prepared.project_handoff.clone();
            let backend = LocalBackend::new(backend_config)
                .map_err(|error| anyhow::anyhow!("failed to configure native sessions: {error}"))?;
            run_coordinator(
                &coordinator,
                submission,
                &backend,
                &cancellation,
                &mut reporters,
            )
            .await?
        }
        IsolationMode::Worker => unreachable!("native worker isolation was rejected"),
    };
    signal_task.abort();

    for report in &run.reports {
        if report.name == "test-results.txt" {
            print!("{}", String::from_utf8_lossy(&report.bytes));
        }
    }
    let report_root = if args.report_dir.is_absolute() {
        args.report_dir.clone()
    } else {
        prepared.project_root.join(&args.report_dir)
    };
    let store = FilesystemArtifactStore::new(report_root);
    let manifest = persist_reports(&store, &run.result.run_id, &run.reports).await?;
    if run.result.state.is_success() {
        if run
            .reports
            .iter()
            .any(|report| report.name != "test-results.txt")
        {
            eprintln!("Reports: {} artifact(s)", manifest.artifacts.len());
        }
        return Ok(());
    }
    if run.result.state.disposition == TerminalDisposition::Cancelled {
        return Err(TestCommandError::new(130).into());
    }
    if run.infrastructure_failures > 0 {
        return Err(TestCommandError::new(2).into());
    }
    Err(TestCommandError::new(1).into())
}

async fn run_coordinator<B>(
    coordinator: &Coordinator,
    submission: RunSubmission,
    backend: &B,
    cancellation: &NativeCancellation,
    reporters: &mut ReporterFanout,
) -> Result<CoordinatedRun>
where
    B: WorkerBackend,
{
    coordinator
        .run(
            submission,
            backend,
            &NativeClock,
            cancellation,
            &NativeTelemetry,
            reporters,
        )
        .await
        .map_err(anyhow::Error::from)
}

fn isolation(args: &TestArgs, configured: Option<ProjectTestIsolation>) -> Result<IsolationMode> {
    let value = args
        .isolation
        .map(|value| match value {
            TestIsolationArg::Auto => IsolationMode::Auto,
            TestIsolationArg::Process => IsolationMode::Process,
            TestIsolationArg::Session => IsolationMode::Session,
            TestIsolationArg::None => IsolationMode::None,
        })
        .or_else(|| {
            configured.map(|value| match value {
                ProjectTestIsolation::Auto => IsolationMode::Auto,
                ProjectTestIsolation::Process => IsolationMode::Process,
                ProjectTestIsolation::Worker => IsolationMode::Worker,
                ProjectTestIsolation::Session => IsolationMode::Session,
                ProjectTestIsolation::None => IsolationMode::None,
            })
        })
        .unwrap_or(IsolationMode::Auto);
    if value == IsolationMode::Worker {
        anyhow::bail!("worker isolation is a browser capability; native hosts use process");
    }
    Ok(value)
}

fn reporters(args: &TestArgs, configured: &[ProjectTestReport]) -> ReporterFanout {
    let requested = if args.reports.is_empty() {
        if configured.is_empty() {
            vec![TestReportArg::Human]
        } else {
            configured
                .iter()
                .map(|format| match format {
                    ProjectTestReport::Human => TestReportArg::Human,
                    ProjectTestReport::Json => TestReportArg::Json,
                    ProjectTestReport::Junit => TestReportArg::Junit,
                    ProjectTestReport::Tap => TestReportArg::Tap,
                })
                .collect()
        }
    } else {
        args.reports.clone()
    };
    let mut reporters = ReporterFanout::default();
    for format in requested {
        match format {
            TestReportArg::Human => reporters.push(HumanReporter::default()),
            TestReportArg::Json => reporters.push(JsonReporter::default()),
            TestReportArg::Junit => reporters.push(JunitReporter),
            TestReportArg::Tap => reporters.push(TapReporter),
        }
    }
    reporters
}
