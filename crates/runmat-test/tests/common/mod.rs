#![allow(dead_code)]

use std::collections::BTreeMap;

use runmat_test::context::TestExecutionContext;
use runmat_test::descriptor::{
    FixtureScope, ProcedureDescriptor, ProcedureKind, SourceDescriptor, SourceSpan,
};
use runmat_test::event::{RedactionPolicy, SequencedEventSink, TestEvent};
use runmat_test::executor::{
    ExecutionFailure, ExecutionFault, ExecutionRequest, ExecutionResponse, TestExecutor,
};
use runmat_test::identity::{RunId, TestId, TestIdentityInput};
use runmat_test::lifecycle::{
    FixtureScopeKey, LifecycleCase, LifecycleEngine, LifecycleOutcome, LifecycleStep,
    NeverCancelled,
};

pub fn run_id() -> RunId {
    RunId::derive("revision", "invocation")
}

pub fn test_id(name: &str) -> TestId {
    TestId::derive(&TestIdentityInput {
        owner_identity: "standalone:fixture",
        relative_source_identity: "tests/example.m",
        semantic_scheme: "function",
        semantic_item_path: name,
        parameter_identity: "",
        fixture_identity: "fixture",
    })
}

pub fn procedure(name: &str) -> ProcedureDescriptor {
    ProcedureDescriptor {
        semantic_path: name.into(),
        display_name: name.into(),
        kind: ProcedureKind::Function,
        source: SourceDescriptor {
            owner_identity: "standalone:fixture".into(),
            relative_path: "tests/example.m".into(),
            semantic_path: name.into(),
            span: SourceSpan {
                start_byte: 0,
                end_byte: 1,
                start_line: 1,
                start_column: 1,
                end_line: 1,
                end_column: 2,
            },
        },
    }
}

pub fn scope(scope: FixtureScope, identity: &str) -> FixtureScopeKey {
    FixtureScopeKey {
        scope,
        identity: if scope == FixtureScope::Test {
            test_id("example").as_str().to_owned()
        } else {
            identity.into()
        },
    }
}

pub fn step(scope: FixtureScopeKey, name: &str) -> LifecycleStep {
    LifecycleStep {
        scope,
        procedure: procedure(name),
    }
}

pub fn lifecycle_case(
    setups: Vec<LifecycleStep>,
    body: &str,
    teardowns: Vec<LifecycleStep>,
) -> LifecycleCase {
    LifecycleCase {
        context: TestExecutionContext {
            run_id: run_id(),
            test_id: test_id("example"),
            attempt: 1,
            random_seed: 7,
        },
        setups,
        body: procedure(body),
        declared_teardowns: teardowns,
    }
}

#[derive(Default)]
pub struct FakeExecutor {
    pub responses: BTreeMap<String, Result<ExecutionResponse, ExecutionFailure>>,
    pub calls: Vec<String>,
}

impl FakeExecutor {
    pub fn responding(
        mut self,
        name: &str,
        response: Result<ExecutionResponse, ExecutionFailure>,
    ) -> Self {
        self.responses.insert(name.into(), response);
        self
    }

    pub fn faulting(self, name: &str, fault: ExecutionFault) -> Self {
        self.responding(name, Err(fault.into()))
    }
}

impl TestExecutor for FakeExecutor {
    fn execute(
        &mut self,
        request: &ExecutionRequest,
    ) -> Result<ExecutionResponse, ExecutionFailure> {
        self.calls.push(request.procedure.semantic_path.clone());
        self.responses
            .get(&request.procedure.semantic_path)
            .cloned()
            .unwrap_or_else(|| Ok(ExecutionResponse::default()))
    }
}

pub fn execute(
    case: &LifecycleCase,
    executor: &mut FakeExecutor,
) -> (LifecycleOutcome, Vec<TestEvent>) {
    let engine = LifecycleEngine::new(RedactionPolicy::new(["secret".into()], 1024));
    let mut events = Vec::new();
    let mut sequenced = SequencedEventSink::new(case.context.run_id.clone(), &mut events);
    let outcome = engine.execute(case, executor, &NeverCancelled, &mut sequenced);
    (outcome, events)
}
