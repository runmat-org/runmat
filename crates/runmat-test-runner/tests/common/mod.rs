#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::{pending, ready};

use runmat_test::descriptor::{
    ProcedureDescriptor, ProcedureKind, SourceDescriptor, SourceSpan, TestDescriptor,
};
use runmat_test::identity::{FixtureGroupId, SuiteId, TestId, TestIdentityInput};
use runmat_test::plan::{FixtureGroupPlan, ProgramRevision, SuitePlan, TestPlanBuilder};
use runmat_test::protocol::ProtocolHandshake;
use runmat_test::result::{AttemptResult, ResultState};
use runmat_test_runner::host::{
    CancellationPort, Clock, HostCapabilities, IsolationMode, PortFuture,
};
use runmat_test_runner::worker::{
    BackendCapabilities, BackendError, BackendErrorKind, BackendFuture, CancelRequest,
    ExecutionRequest, RunSubmission, SpawnRequest, WorkerBackend, WorkerExecution,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FakeSession(pub u64);

pub enum Step {
    Result(Result<WorkerExecution, BackendError>),
    Pending,
}

pub struct FakeBackend {
    capabilities: BackendCapabilities,
    steps: RefCell<VecDeque<Step>>,
    cancel_result: RefCell<Option<Result<Option<WorkerExecution>, BackendError>>>,
    next_session: Cell<u64>,
    pub spawned: RefCell<Vec<FakeSession>>,
    pub executions: RefCell<Vec<(FakeSession, ExecutionRequest)>>,
    pub cancelled: RefCell<Vec<FakeSession>>,
    pub terminated: RefCell<Vec<FakeSession>>,
    pub shutdown: RefCell<Vec<FakeSession>>,
}

impl FakeBackend {
    pub fn new(steps: impl IntoIterator<Item = Step>) -> Self {
        Self {
            capabilities: BackendCapabilities {
                host: HostCapabilities::new(
                    [
                        IsolationMode::Process,
                        IsolationMode::Session,
                        IsolationMode::None,
                    ],
                    8,
                )
                .unwrap(),
                handshake: ProtocolHandshake::current("fake", Vec::new()),
            },
            steps: RefCell::new(steps.into_iter().collect()),
            cancel_result: RefCell::new(None),
            next_session: Cell::new(0),
            spawned: RefCell::new(Vec::new()),
            executions: RefCell::new(Vec::new()),
            cancelled: RefCell::new(Vec::new()),
            terminated: RefCell::new(Vec::new()),
            shutdown: RefCell::new(Vec::new()),
        }
    }

    pub fn with_cancel_result(self, result: WorkerExecution) -> Self {
        *self.cancel_result.borrow_mut() = Some(Ok(Some(result)));
        self
    }
}

impl WorkerBackend for FakeBackend {
    type Session = FakeSession;

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn spawn<'a>(&'a self, _request: SpawnRequest) -> BackendFuture<'a, Self::Session> {
        let id = self.next_session.get();
        self.next_session.set(id + 1);
        let session = FakeSession(id);
        self.spawned.borrow_mut().push(session.clone());
        Box::pin(ready(Ok(session)))
    }

    fn execute<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        self.executions
            .borrow_mut()
            .push((session.clone(), request));
        match self.steps.borrow_mut().pop_front().expect("scripted step") {
            Step::Result(result) => Box::pin(ready(result)),
            Step::Pending => Box::pin(pending()),
        }
    }

    fn cancel<'a>(
        &'a self,
        session: &'a Self::Session,
        _request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>> {
        self.cancelled.borrow_mut().push(session.clone());
        Box::pin(ready(
            self.cancel_result.borrow_mut().take().unwrap_or(Ok(None)),
        ))
    }

    fn terminate<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        self.terminated.borrow_mut().push(session.clone());
        Box::pin(ready(Ok(())))
    }

    fn shutdown<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        self.shutdown.borrow_mut().push(session.clone());
        Box::pin(ready(Ok(())))
    }
}

pub struct ImmediateClock {
    now: u64,
}

impl ImmediateClock {
    pub fn new(now: u64) -> Self {
        Self { now }
    }
}

impl Clock for ImmediateClock {
    fn now_ms(&self) -> u64 {
        self.now
    }

    fn sleep_until<'a>(&'a self, _deadline_ms: u64) -> PortFuture<'a, ()> {
        Box::pin(ready(()))
    }
}

pub struct PendingClock;

impl Clock for PendingClock {
    fn now_ms(&self) -> u64 {
        0
    }

    fn sleep_until<'a>(&'a self, _deadline_ms: u64) -> PortFuture<'a, ()> {
        Box::pin(pending())
    }
}

pub struct ImmediateCancellation {
    reason: String,
}

impl ImmediateCancellation {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl CancellationPort for ImmediateCancellation {
    fn is_cancelled(&self) -> bool {
        false
    }

    fn reason(&self) -> Option<String> {
        None
    }

    fn cancelled<'a>(&'a self) -> PortFuture<'a, String> {
        Box::pin(ready(self.reason.clone()))
    }
}

pub fn plan(names: &[&str]) -> RunSubmission {
    let revision = ProgramRevision {
        graph_digest: "graph".into(),
        source_digest: "source".into(),
        semantic_schema: 1,
        compiler_schema: 1,
        test_config_digest: "config".into(),
    };
    let suite_id = SuiteId::derive(&revision.canonical_identity(), "suite");
    let group_id = FixtureGroupId::derive(suite_id.as_str(), "group");
    let tests = names
        .iter()
        .map(|name| {
            let id = TestId::derive(&TestIdentityInput {
                owner_identity: "root",
                relative_source_identity: "tests/sample.m",
                semantic_scheme: "function",
                semantic_item_path: name,
                parameter_identity: "",
                fixture_identity: group_id.as_str(),
            });
            TestDescriptor {
                id,
                suite_id: suite_id.clone(),
                fixture_group_id: group_id.clone(),
                display_name: (*name).into(),
                procedure: ProcedureDescriptor {
                    semantic_path: (*name).into(),
                    display_name: (*name).into(),
                    kind: ProcedureKind::Function,
                    source: SourceDescriptor {
                        owner_identity: "root".into(),
                        relative_path: "tests/sample.m".into(),
                        semantic_path: (*name).into(),
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
            }
        })
        .collect();
    let plan = TestPlanBuilder::new(revision.clone(), "test")
        .add_suite(SuitePlan {
            id: suite_id,
            display_name: "suite".into(),
            fixture_groups: vec![FixtureGroupPlan {
                id: group_id,
                fixtures: Vec::new(),
                tests,
            }],
        })
        .build()
        .unwrap();
    let snapshot = runmat_test::discovery::FrozenTestRunSnapshot::freeze(
        revision.graph_digest,
        "source",
        revision.semantic_schema,
        revision.compiler_schema,
        revision.test_config_digest,
        vec![runmat_test::discovery::SavedRunSource {
            owner_identity: "root".into(),
            relative_path: "tests/sample.m".into(),
            content: "%% sample\n".into(),
        }],
        Vec::new(),
    )
    .unwrap();
    let mut plan = plan;
    plan.program_revision = snapshot.program_revision.clone();
    RunSubmission::new(plan, snapshot).unwrap()
}

pub fn passed(test_id: TestId, attempt: u32) -> WorkerExecution {
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

pub fn crashed(message: &str) -> BackendError {
    BackendError::new(BackendErrorKind::Crashed, message)
}
