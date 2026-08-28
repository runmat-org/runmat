use std::collections::BTreeSet;

use futures::stream::{self, StreamExt};
use runmat_test::coverage::{merge_aggregates, merge_coverage, CoverageAggregate};
use runmat_test::event::TestEventPayload;
use runmat_test::identity::TestId;
use runmat_test::plan::TestPlan;
use runmat_test::result::{
    aggregate_run_state, merge_attempts, AttemptResult, RunResult, TerminalDisposition, TestResult,
};

use crate::host::{CancellationPort, Clock, IsolationMode};
use crate::reporter::{RenderedReport, ReporterFanout};
use crate::schedule::{local_lanes, RetryPolicy};
use crate::telemetry::{NoopTelemetry, TelemetryFields, TelemetryPort};
use crate::worker::{
    BackendErrorKind, ExecutionRequest, RunSubmission, SpawnRequest, WorkerBackend, WorkerExecution,
};
use crate::{RunnerError, RunnerResult};

use super::cancellation::{cancel_or_terminate, CancellationRequest};
use super::internal_cancellation::{CombinedCancellation, InternalCancellation};
use super::queue::build_queue;
use super::recovery::terminal_attempt;
use super::state::EventState;
use super::timeout::{execute_with_controls, ExecutionRace};

#[derive(Clone, Debug)]
pub struct CoordinatorConfig {
    pub isolation: IsolationMode,
    pub jobs: usize,
    pub timeout_ms: Option<u64>,
    pub cancellation_grace_ms: u64,
    pub retry: RetryPolicy,
    pub shard_index: Option<u32>,
    pub shard_count: Option<u32>,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            isolation: IsolationMode::Auto,
            jobs: 1,
            timeout_ms: None,
            cancellation_grace_ms: 1_000,
            retry: RetryPolicy::default(),
            shard_index: None,
            shard_count: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoordinatedRun {
    pub result: RunResult,
    pub events: Vec<runmat_test::event::TestEvent>,
    pub reports: Vec<RenderedReport>,
    pub infrastructure_failures: usize,
    pub plugin_failures: usize,
    pub isolation: IsolationMode,
    pub coverage: CoverageAggregate,
}

#[derive(Clone, Debug)]
pub struct Coordinator {
    config: CoordinatorConfig,
}

impl Coordinator {
    pub fn new(config: CoordinatorConfig) -> RunnerResult<Self> {
        if config.jobs == 0 {
            return Err(RunnerError::InvalidConfiguration(
                "jobs must be greater than zero".into(),
            ));
        }
        if config.retry.max_attempts == 0 {
            return Err(RunnerError::InvalidConfiguration(
                "retry max_attempts must be greater than zero".into(),
            ));
        }
        Ok(Self { config })
    }

    pub async fn run<B, C, X, T>(
        &self,
        submission: RunSubmission,
        backend: &B,
        clock: &C,
        cancellation: &X,
        telemetry: &T,
        reporters: &mut ReporterFanout,
    ) -> RunnerResult<CoordinatedRun>
    where
        B: WorkerBackend,
        C: Clock,
        X: CancellationPort,
        T: TelemetryPort,
    {
        let capabilities = backend.capabilities();
        let queue = build_queue(&submission.plan, &self.config)?;
        let lanes = local_lanes(self.config.jobs, capabilities.host.max_workers, queue.len())?;
        if lanes == 1 {
            return self
                .run_serial(
                    submission,
                    backend,
                    clock,
                    cancellation,
                    telemetry,
                    reporters,
                )
                .await;
        }
        self.run_parallel(
            submission,
            queue,
            lanes,
            backend,
            clock,
            cancellation,
            telemetry,
            reporters,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_parallel<B, C, X, T>(
        &self,
        submission: RunSubmission,
        queue: Vec<super::queue::GroupQueue>,
        lanes: usize,
        backend: &B,
        clock: &C,
        cancellation: &X,
        telemetry: &T,
        reporters: &mut ReporterFanout,
    ) -> RunnerResult<CoordinatedRun>
    where
        B: WorkerBackend,
        C: Clock,
        X: CancellationPort,
        T: TelemetryPort,
    {
        let isolation = backend.capabilities().host.resolve(self.config.isolation)?;
        telemetry.event(
            "test.run.started",
            &TelemetryFields::default()
                .public("run_id", submission.plan.run_id.as_str())
                .public("isolation", isolation.as_str()),
        );
        let mut state = EventState::new(submission.plan.run_id.clone(), reporters);
        state.emit(TestEventPayload::RunStarted)?;
        let config = CoordinatorConfig {
            jobs: 1,
            shard_index: None,
            shard_count: None,
            ..self.config.clone()
        };
        let internal_cancellation = InternalCancellation::default();
        let runs = stream::iter(queue.into_iter().enumerate().map(|(index, group)| {
            let group_submission = group_submission(&submission, &group);
            let config = config.clone();
            let internal_cancellation = internal_cancellation.clone();
            async move {
                let coordinator = Coordinator::new(config)?;
                let mut reporters = ReporterFanout::default();
                let combined =
                    CombinedCancellation::new(cancellation, internal_cancellation.clone());
                let run = coordinator
                    .run_serial(
                        group_submission,
                        backend,
                        clock,
                        &combined,
                        &NoopTelemetry,
                        &mut reporters,
                    )
                    .await?;
                if run
                    .result
                    .tests
                    .iter()
                    .flat_map(|test| &test.attempts)
                    .any(|attempt| attempt.abort_run)
                {
                    internal_cancellation.cancel("fatal test assertion requested run abort");
                }
                Ok((index, run))
            }
        }))
        .buffer_unordered(lanes)
        .collect::<Vec<_>>()
        .await;
        let mut runs = runs.into_iter().collect::<RunnerResult<Vec<_>>>()?;
        runs.sort_by_key(|(index, _)| *index);

        let mut tests = Vec::new();
        let mut infrastructure_failures = 0;
        let mut coverage = Vec::new();
        for (_, run) in runs {
            infrastructure_failures += run.infrastructure_failures;
            tests.extend(run.result.tests);
            coverage.push(run.coverage);
            for event in run.events {
                if !matches!(
                    event.payload,
                    TestEventPayload::RunStarted | TestEventPayload::RunFinished { .. }
                ) {
                    state.emit(event.payload)?;
                }
            }
        }
        let result = run_result(&submission.plan, tests);
        state.emit(TestEventPayload::RunFinished {
            result: result.clone(),
        })?;
        let events = state.finish();
        let reports = reporters.finish(&result)?;
        let coverage =
            merge_aggregates(coverage).map_err(|error| RunnerError::Protocol(error.to_string()))?;
        telemetry.event(
            "test.run.finished",
            &TelemetryFields::default()
                .public("run_id", submission.plan.run_id.as_str())
                .public("disposition", format!("{:?}", result.state.disposition)),
        );
        Ok(CoordinatedRun {
            result,
            events,
            reports,
            infrastructure_failures,
            plugin_failures: 0,
            isolation,
            coverage,
        })
    }

    async fn run_serial<B, C, X, T>(
        &self,
        submission: RunSubmission,
        backend: &B,
        clock: &C,
        cancellation: &X,
        telemetry: &T,
        reporters: &mut ReporterFanout,
    ) -> RunnerResult<CoordinatedRun>
    where
        B: WorkerBackend,
        C: Clock,
        X: CancellationPort,
        T: TelemetryPort,
    {
        let plan = &submission.plan;
        let capabilities = backend.capabilities();
        let isolation = capabilities.host.resolve(self.config.isolation)?;
        let queue = build_queue(plan, &self.config)?;
        let mut state = EventState::new(plan.run_id.clone(), reporters);
        let mut results = Vec::new();
        let mut infrastructure_failures = 0;
        let mut abort_reason = None;
        let mut coverage = Vec::new();

        telemetry.event(
            "test.run.started",
            &TelemetryFields::default()
                .public("run_id", plan.run_id.as_str())
                .public("isolation", isolation.as_str()),
        );
        state.emit(TestEventPayload::RunStarted)?;

        for group in queue {
            let mut session = None;
            for test_id in group.tests {
                if abort_reason.is_none() && cancellation.is_cancelled() {
                    abort_reason = Some(
                        cancellation
                            .reason()
                            .unwrap_or_else(|| "run cancelled".into()),
                    );
                }
                if let Some(reason) = &abort_reason {
                    let attempt = terminal_attempt(
                        test_id.clone(),
                        1,
                        TerminalDisposition::Cancelled,
                        "runmat:test:Cancelled",
                        reason.clone(),
                    );
                    emit_attempt(&mut state, &attempt)?;
                    results.push(
                        merge_attempts(test_id, vec![attempt])
                            .expect("one matching attempt always merges"),
                    );
                    continue;
                }

                let mut attempts = Vec::new();
                let mut attempt_number = 1;
                loop {
                    if session.is_none() {
                        match backend
                            .spawn(SpawnRequest {
                                submission: submission.clone(),
                                isolation,
                            })
                            .await
                        {
                            Ok(spawned) => session = Some(spawned),
                            Err(error) => {
                                infrastructure_failures += 1;
                                let attempt = terminal_attempt(
                                    test_id.clone(),
                                    attempt_number,
                                    TerminalDisposition::Crashed,
                                    "runmat:test:WorkerSpawn",
                                    error.to_string(),
                                );
                                emit_attempt(&mut state, &attempt)?;
                                attempts.push(attempt);
                                if self.config.retry.should_retry(attempt_number, true) {
                                    attempt_number += 1;
                                    continue;
                                }
                                break;
                            }
                        }
                    }
                    let active = session.as_ref().expect("session was spawned");
                    state.emit(TestEventPayload::TestStarted {
                        test_id: test_id.clone(),
                        attempt: attempt_number,
                    })?;
                    let deadline_ms = self
                        .config
                        .timeout_ms
                        .map(|timeout| clock.now_ms().saturating_add(timeout));
                    let race = execute_with_controls(
                        backend,
                        active,
                        ExecutionRequest {
                            test_id: test_id.clone(),
                            attempt: attempt_number,
                            deadline_ms,
                        },
                        clock,
                        cancellation,
                    )
                    .await;
                    let (execution, infrastructure_failure, lost_session) = match race {
                        ExecutionRace::Completed(Ok(execution)) => {
                            validate_execution(
                                &test_id,
                                attempt_number,
                                &plan.program_revision.canonical_identity(),
                                &execution,
                            )?;
                            (execution, false, false)
                        }
                        ExecutionRace::Completed(Err(error)) => {
                            infrastructure_failures += 1;
                            let disposition = match error.kind {
                                BackendErrorKind::Rejected => TerminalDisposition::Failed,
                                _ => TerminalDisposition::Crashed,
                            };
                            let attempt = terminal_attempt(
                                test_id.clone(),
                                attempt_number,
                                disposition,
                                "runmat:test:WorkerFailure",
                                error.to_string(),
                            );
                            let _ = backend.terminate(active).await;
                            (
                                WorkerExecution {
                                    result: attempt,
                                    events: Vec::new(),
                                    coverage: Vec::new(),
                                },
                                true,
                                true,
                            )
                        }
                        ExecutionRace::TimedOut => {
                            let (execution, terminated) = cancel_or_terminate(
                                backend,
                                active,
                                clock,
                                CancellationRequest {
                                    run_id: plan.run_id.clone(),
                                    test_id: test_id.clone(),
                                    attempt: attempt_number,
                                    reason: "deadline elapsed".into(),
                                    grace_ms: self.config.cancellation_grace_ms,
                                    disposition: TerminalDisposition::TimedOut,
                                },
                            )
                            .await;
                            (execution, false, terminated)
                        }
                        ExecutionRace::Cancelled(reason) => {
                            abort_reason = Some(reason.clone());
                            let (execution, terminated) = cancel_or_terminate(
                                backend,
                                active,
                                clock,
                                CancellationRequest {
                                    run_id: plan.run_id.clone(),
                                    test_id: test_id.clone(),
                                    attempt: attempt_number,
                                    reason,
                                    grace_ms: self.config.cancellation_grace_ms,
                                    disposition: TerminalDisposition::Cancelled,
                                },
                            )
                            .await;
                            (execution, false, terminated)
                        }
                    };
                    if lost_session {
                        session = None;
                    }
                    coverage.extend(execution.coverage.iter().cloned());
                    for event in execution.events {
                        state.forward(event)?;
                    }
                    let abort_run = execution.result.abort_run;
                    let result = execution.result;
                    state.emit(TestEventPayload::TestFinished {
                        result: result.clone(),
                    })?;
                    attempts.push(result);
                    if abort_run {
                        abort_reason = Some("fatal test assertion requested run abort".into());
                    }
                    if self
                        .config
                        .retry
                        .should_retry(attempt_number, infrastructure_failure)
                    {
                        attempt_number += 1;
                        continue;
                    }
                    break;
                }
                let result = merge_attempts(test_id.clone(), attempts).ok_or_else(|| {
                    RunnerError::Protocol(format!(
                        "worker attempts for '{}' were missing or inconsistent",
                        test_id.as_str()
                    ))
                })?;
                results.push(result);
            }
            if let Some(active) = &session {
                backend
                    .shutdown(active)
                    .await
                    .map_err(|error| RunnerError::Backend(error.to_string()))?;
            }
        }

        let result = run_result(plan, results);
        state.emit(TestEventPayload::RunFinished {
            result: result.clone(),
        })?;
        let events = state.finish();
        let reports = reporters.finish(&result)?;
        let coverage =
            merge_coverage(coverage).map_err(|error| RunnerError::Protocol(error.to_string()))?;
        telemetry.event(
            "test.run.finished",
            &TelemetryFields::default()
                .public("run_id", plan.run_id.as_str())
                .public("disposition", format!("{:?}", result.state.disposition)),
        );
        Ok(CoordinatedRun {
            result,
            events,
            reports,
            infrastructure_failures,
            plugin_failures: 0,
            isolation,
            coverage,
        })
    }
}

fn group_submission(submission: &RunSubmission, group: &super::queue::GroupQueue) -> RunSubmission {
    let selected = group.tests.iter().cloned().collect::<BTreeSet<_>>();
    let mut plan = submission.plan.clone();
    plan.suites.retain_mut(|suite| {
        suite
            .fixture_groups
            .retain(|candidate| candidate.id == group.group_id);
        for candidate in &mut suite.fixture_groups {
            candidate.tests.retain(|test| selected.contains(&test.id));
        }
        !suite.fixture_groups.is_empty()
    });
    RunSubmission {
        plan,
        snapshot: submission.snapshot.clone(),
    }
}

fn validate_execution(
    test_id: &TestId,
    attempt: u32,
    program_revision: &str,
    execution: &WorkerExecution,
) -> RunnerResult<()> {
    if execution.result.test_id != *test_id || execution.result.attempt != attempt {
        return Err(RunnerError::Protocol(format!(
            "worker completed the wrong test or attempt for '{}'",
            test_id.as_str()
        )));
    }
    if let Some(fragment) = execution
        .coverage
        .iter()
        .find(|fragment| fragment.program_revision != program_revision)
    {
        return Err(RunnerError::Protocol(format!(
            "worker coverage revision '{}' does not match plan revision '{}'",
            fragment.program_revision, program_revision
        )));
    }
    Ok(())
}

fn emit_attempt(state: &mut EventState<'_>, attempt: &AttemptResult) -> RunnerResult<()> {
    state.emit(TestEventPayload::TestStarted {
        test_id: attempt.test_id.clone(),
        attempt: attempt.attempt,
    })?;
    state.emit(TestEventPayload::TestFinished {
        result: attempt.clone(),
    })
}

fn run_result(plan: &TestPlan, tests: Vec<TestResult>) -> RunResult {
    RunResult {
        run_id: plan.run_id.clone(),
        state: aggregate_run_state(tests.iter().map(|result| &result.state)),
        tests,
    }
}
