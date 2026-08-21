use std::cell::{Cell, RefCell};
use std::future::ready;

use futures::channel::oneshot;
use futures::executor::block_on;
use futures::{poll, FutureExt};
use runmat_execution::{Digest, DomainContribution, ProgramEnvironment, ProgramRevision};
use runmat_test::descriptor::{
    ProcedureDescriptor, ProcedureKind, SourceDescriptor, SourceSpan, TestDescriptor,
};
use runmat_test::identity::{FixtureGroupId, SuiteId, TestId, TestIdentityInput};
use runmat_test::plan::{FixtureGroupPlan, SuitePlan, TestPlanBuilder};
use runmat_test::protocol::ProtocolHandshake;
use runmat_test::result::{AttemptResult, ResultState};
use runmat_test_runner::host::{HostCapabilities, IsolationMode};
use runmat_test_runner::worker::{
    BackendCapabilities, BackendFuture, CancelRequest, ExecutionRequest, RunSubmission,
    SpawnRequest, WorkerBackend, WorkerExecution,
};
use runmat_test_runner_execution::{
    decode_execution, encode_execution, ExecutionBackendConfig, ExecutionWorkerBackend,
    TestAttemptWorkload,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct FakeSession(u64);

struct FakeBackend {
    next_session: Cell<u64>,
    executions: RefCell<Vec<(FakeSession, ExecutionRequest)>>,
}

struct CancellingBackend {
    completion: RefCell<Option<oneshot::Receiver<WorkerExecution>>>,
}

impl CancellingBackend {
    fn new() -> (Self, oneshot::Sender<WorkerExecution>) {
        let (sender, completion) = oneshot::channel();
        (
            Self {
                completion: RefCell::new(Some(completion)),
            },
            sender,
        )
    }
}

impl FakeBackend {
    fn new() -> Self {
        Self {
            next_session: Cell::new(0),
            executions: RefCell::new(Vec::new()),
        }
    }
}

impl WorkerBackend for FakeBackend {
    type Session = FakeSession;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            host: HostCapabilities::new([IsolationMode::Process], 8).unwrap(),
            handshake: ProtocolHandshake::current("fake", Vec::new()),
        }
    }

    fn spawn<'a>(&'a self, _request: SpawnRequest) -> BackendFuture<'a, Self::Session> {
        let id = self.next_session.get();
        self.next_session.set(id + 1);
        Box::pin(ready(Ok(FakeSession(id))))
    }

    fn execute<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        self.executions
            .borrow_mut()
            .push((session.clone(), request.clone()));
        Box::pin(ready(Ok(passed(request.test_id, request.attempt))))
    }

    fn cancel<'a>(
        &'a self,
        _session: &'a Self::Session,
        _request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>> {
        Box::pin(ready(Ok(None)))
    }

    fn terminate<'a>(&'a self, _session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(ready(Ok(())))
    }

    fn shutdown<'a>(&'a self, _session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(ready(Ok(())))
    }
}

impl WorkerBackend for CancellingBackend {
    type Session = FakeSession;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            host: HostCapabilities::new([IsolationMode::Process], 1).unwrap(),
            handshake: ProtocolHandshake::current("cancelling", Vec::new()),
        }
    }

    fn spawn<'a>(&'a self, _request: SpawnRequest) -> BackendFuture<'a, Self::Session> {
        Box::pin(ready(Ok(FakeSession(0))))
    }

    fn execute<'a>(
        &'a self,
        _session: &'a Self::Session,
        _request: ExecutionRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        let completion = self.completion.borrow_mut().take().unwrap();
        Box::pin(async move { Ok(completion.await.unwrap()) })
    }

    fn cancel<'a>(
        &'a self,
        _session: &'a Self::Session,
        request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>> {
        let test_id = submission().plan.tests().next().unwrap().id.clone();
        Box::pin(ready(Ok(Some(passed(
            test_id,
            request.grace_deadline_ms as u32,
        )))))
    }

    fn terminate<'a>(&'a self, _session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(ready(Ok(())))
    }

    fn shutdown<'a>(&'a self, _session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(ready(Ok(())))
    }
}

#[test]
fn schedules_exact_attempt_without_owning_test_result_semantics() {
    let submission = submission();
    let revision = submission.plan.program_revision.clone();
    let test_id = submission.plan.tests().next().unwrap().id.clone();
    let backend =
        ExecutionWorkerBackend::new(FakeBackend::new(), ExecutionBackendConfig::local(2)).unwrap();
    let session = block_on(backend.spawn(SpawnRequest {
        submission,
        isolation: IsolationMode::Process,
    }))
    .unwrap();
    let execution = block_on(backend.execute(
        &session,
        ExecutionRequest {
            test_id: test_id.clone(),
            attempt: 1,
            deadline_ms: Some(42),
        },
    ))
    .unwrap();

    assert_eq!(session.program_revision(), &revision);
    assert_eq!(execution.result.test_id, test_id);
    assert_eq!(execution.result.state, ResultState::PASSED);
    assert_eq!(backend.capabilities().host.max_workers, 2);
}

#[test]
fn rejects_a_zero_capacity_composition() {
    let error = ExecutionWorkerBackend::new(FakeBackend::new(), ExecutionBackendConfig::local(0))
        .err()
        .expect("zero capacity must fail");
    assert!(error.to_string().contains("capacity"));
}

#[test]
fn remote_workload_preserves_revision_result_events_and_coverage_payload() {
    let submission = submission();
    let test_id = submission.plan.tests().next().unwrap().id.clone();
    let workload = TestAttemptWorkload::new(submission, test_id.clone(), 1).unwrap();
    let program = workload.program_request().unwrap();
    let decoded = TestAttemptWorkload::from_program_request(&program).unwrap();
    assert_eq!(
        decoded.submission.plan.program_revision,
        program.recipe.program_revision
    );
    assert_eq!(decoded.test_id, test_id);

    let execution = passed(decoded.test_id, decoded.attempt);
    assert_eq!(
        decode_execution(&encode_execution(&execution).unwrap()).unwrap(),
        execution
    );
}

#[test]
fn cancellation_fences_a_late_backend_completion_without_double_reporting() {
    block_on(async {
        let submission = submission();
        let run_id = submission.plan.run_id.clone();
        let test_id = submission.plan.tests().next().unwrap().id.clone();
        let (inner, completion) = CancellingBackend::new();
        let backend = ExecutionWorkerBackend::new(inner, ExecutionBackendConfig::local(1)).unwrap();
        let session = backend
            .spawn(SpawnRequest {
                submission,
                isolation: IsolationMode::Process,
            })
            .await
            .unwrap();
        let mut execution = backend
            .execute(
                &session,
                ExecutionRequest {
                    test_id: test_id.clone(),
                    attempt: 1,
                    deadline_ms: None,
                },
            )
            .boxed_local();
        assert!(poll!(&mut execution).is_pending());
        let cancelled = backend
            .cancel(
                &session,
                CancelRequest {
                    run_id,
                    reason: "test cancellation".into(),
                    grace_deadline_ms: 1,
                },
            )
            .await
            .unwrap()
            .unwrap();
        completion.send(passed(test_id.clone(), 1)).unwrap();
        let completed = execution.await.unwrap();
        assert_eq!(cancelled.result.test_id, test_id);
        assert_eq!(completed.result.test_id, test_id);
    });
}

fn submission() -> RunSubmission {
    let initial_revision = ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
    .with_domain_contribution(
        DomainContribution::new("runmat.test.config", Digest::sha256(b"config")).unwrap(),
    )
    .unwrap();
    let snapshot = runmat_test::discovery::FrozenTestRunSnapshot::freeze(
        initial_revision.graph_digest().to_string(),
        "source",
        initial_revision.environment(),
        initial_revision
            .domain_contribution("runmat.test.config")
            .unwrap()
            .to_string(),
        vec![runmat_test::discovery::SavedRunSource {
            owner_identity: "root".into(),
            relative_path: "tests/sample.m".into(),
            content: "%% sample\n".into(),
        }],
        Vec::new(),
    )
    .unwrap();
    let revision = snapshot.program_revision.clone();
    let suite_id = SuiteId::derive(&revision.canonical_identity(), "suite");
    let group_id = FixtureGroupId::derive(suite_id.as_str(), "group");
    let test_id = TestId::derive(&TestIdentityInput {
        owner_identity: "root",
        relative_source_identity: "tests/sample.m",
        semantic_scheme: "function",
        semantic_item_path: "sample",
        parameter_identity: "",
        fixture_identity: group_id.as_str(),
    });
    let test = TestDescriptor {
        id: test_id,
        suite_id: suite_id.clone(),
        fixture_group_id: group_id.clone(),
        display_name: "sample".into(),
        procedure: ProcedureDescriptor {
            semantic_path: "sample".into(),
            display_name: "sample".into(),
            kind: ProcedureKind::Function,
            source: SourceDescriptor {
                owner_identity: "root".into(),
                relative_path: "tests/sample.m".into(),
                semantic_path: "sample".into(),
                span: SourceSpan {
                    start_byte: 0,
                    end_byte: 1,
                    start_line: 1,
                    start_column: 1,
                    end_line: 1,
                    end_column: 2,
                },
            },
        },
        parameters: Vec::new(),
        tags: Vec::new(),
        requirements: Default::default(),
    };
    let plan = TestPlanBuilder::new(revision, "bridge-test")
        .add_suite(SuitePlan {
            id: suite_id,
            display_name: "suite".into(),
            fixture_groups: vec![FixtureGroupPlan {
                id: group_id,
                fixtures: Vec::new(),
                tests: vec![test],
            }],
        })
        .build()
        .unwrap();
    RunSubmission::new(plan, snapshot).unwrap()
}

fn passed(test_id: TestId, attempt: u32) -> WorkerExecution {
    WorkerExecution {
        result: AttemptResult {
            test_id,
            attempt,
            state: ResultState::PASSED,
            diagnostics: Vec::new(),
            artifacts: Vec::new(),
            output: String::new(),
            abort_run: false,
        },
        events: Vec::new(),
        coverage: Vec::new(),
    }
}
