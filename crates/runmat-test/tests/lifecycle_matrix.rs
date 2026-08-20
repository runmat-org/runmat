mod common;

use common::{execute, lifecycle_case, procedure, scope, step, FakeExecutor};
use runmat_test::context::TestCommand;
use runmat_test::descriptor::FixtureScope;
use runmat_test::executor::{ExecutionFailure, ExecutionFault, ExecutionResponse};
use runmat_test::lifecycle::{
    CancellationProbe, FixtureScopeKey, LifecycleEngine, QualificationKind,
};
use runmat_test::result::{Diagnostic, DiagnosticSeverity, ResultState, TerminalDisposition};

#[test]
fn successful_case_runs_dynamic_lifo_before_declared_teardowns() {
    let class_scope = scope(FixtureScope::Class, "class");
    let test_scope = scope(FixtureScope::Test, "test");
    let response = ExecutionResponse {
        commands: vec![
            TestCommand::AddTeardown {
                scope: test_scope.clone(),
                procedure: procedure("dynamic-1"),
            },
            TestCommand::AddTeardown {
                scope: test_scope.clone(),
                procedure: procedure("dynamic-2"),
            },
        ],
        output: "visible secret".into(),
    };
    let case = lifecycle_case(
        vec![
            step(class_scope.clone(), "class-setup"),
            step(test_scope.clone(), "test-setup"),
        ],
        "body",
        vec![
            step(test_scope, "test-teardown"),
            step(class_scope, "class-teardown"),
        ],
    );
    let mut executor = FakeExecutor::default().responding("body", Ok(response));
    let (outcome, events) = execute(&case, &mut executor);

    assert_eq!(
        executor.calls,
        [
            "class-setup",
            "test-setup",
            "body",
            "dynamic-2",
            "dynamic-1",
            "test-teardown",
            "class-teardown"
        ]
    );
    assert_eq!(outcome.attempt.state, ResultState::PASSED);
    assert_eq!(outcome.attempt.output, "visible [REDACTED]");
    assert!(events
        .windows(2)
        .all(|pair| pair[1].sequence == pair[0].sequence + 1));
}

#[test]
fn equal_level_fixture_scopes_unwind_by_activation_order_with_exact_scope_context() {
    let outer = FixtureScopeKey {
        scope: FixtureScope::Test,
        identity: "z-outer".into(),
    };
    let inner = FixtureScopeKey {
        scope: FixtureScope::Test,
        identity: "a-inner".into(),
    };
    let case = lifecycle_case(
        vec![
            step(outer.clone(), "outer-setup"),
            step(inner.clone(), "inner-setup"),
        ],
        "body",
        vec![
            step(inner.clone(), "inner-teardown"),
            step(outer.clone(), "outer-teardown"),
        ],
    );
    let mut executor = FakeExecutor::default().responding(
        "inner-setup",
        Ok(ExecutionResponse {
            commands: vec![TestCommand::AddTeardown {
                scope: inner.clone(),
                procedure: procedure("inner-dynamic"),
            }],
            output: String::new(),
        }),
    );
    let (outcome, _) = execute(&case, &mut executor);

    assert_eq!(
        executor.calls,
        [
            "outer-setup",
            "inner-setup",
            "body",
            "inner-dynamic",
            "inner-teardown",
            "outer-teardown"
        ]
    );
    assert_eq!(
        executor.scopes,
        [
            outer.clone(),
            inner.clone(),
            FixtureScopeKey {
                scope: FixtureScope::Test,
                identity: case.context.test_id.as_str().to_owned(),
            },
            inner.clone(),
            inner,
            outer,
        ]
    );
    assert_eq!(outcome.attempt.state, ResultState::PASSED);
}

#[test]
fn setup_failure_skips_body_but_runs_current_and_outer_teardown() {
    let class_scope = scope(FixtureScope::Class, "class");
    let test_scope = scope(FixtureScope::Test, "test");
    let case = lifecycle_case(
        vec![
            step(class_scope.clone(), "class-setup"),
            step(test_scope.clone(), "test-setup"),
        ],
        "body",
        vec![
            step(test_scope, "test-teardown"),
            step(class_scope, "class-teardown"),
        ],
    );
    let mut executor = FakeExecutor::default().responding(
        "test-setup",
        Err(ExecutionFailure {
            fault: ExecutionFault::Uncaught("setup failed".into()),
            partial: ExecutionResponse {
                commands: vec![TestCommand::AddTeardown {
                    scope: test_scope_for_failure(),
                    procedure: procedure("partial-cleanup"),
                }],
                output: String::new(),
            },
        }),
    );
    let (outcome, _) = execute(&case, &mut executor);

    assert_eq!(
        executor.calls,
        [
            "class-setup",
            "test-setup",
            "partial-cleanup",
            "test-teardown",
            "class-teardown"
        ]
    );
    assert_eq!(
        outcome.attempt.state,
        ResultState {
            failed: true,
            incomplete: true,
            disposition: TerminalDisposition::Failed,
        }
    );
}

fn test_scope_for_failure() -> runmat_test::lifecycle::FixtureScopeKey {
    scope(FixtureScope::Test, "test")
}

#[test]
fn verification_records_failure_without_aborting_later_phases() {
    let test_scope = scope(FixtureScope::Test, "test");
    let case = lifecycle_case(
        vec![step(test_scope.clone(), "setup")],
        "body",
        vec![step(test_scope, "teardown")],
    );
    let mut executor = FakeExecutor::default().responding(
        "setup",
        Ok(qualification(QualificationKind::VerificationFailed)),
    );
    let (outcome, _) = execute(&case, &mut executor);

    assert_eq!(executor.calls, ["setup", "body", "teardown"]);
    assert!(outcome.attempt.state.failed);
    assert!(!outcome.attempt.state.incomplete);
}

#[test]
fn teardown_fault_does_not_hide_primary_timeout_or_skip_outer_teardown() {
    let class_scope = scope(FixtureScope::Class, "class");
    let test_scope = scope(FixtureScope::Test, "test");
    let case = lifecycle_case(
        vec![
            step(class_scope.clone(), "class-setup"),
            step(test_scope.clone(), "test-setup"),
        ],
        "body",
        vec![
            step(test_scope, "test-teardown"),
            step(class_scope, "class-teardown"),
        ],
    );
    let mut executor = FakeExecutor::default()
        .faulting("body", ExecutionFault::TimedOut("deadline".into()))
        .faulting("test-teardown", ExecutionFault::Uncaught("cleanup".into()));
    let (outcome, _) = execute(&case, &mut executor);

    assert_eq!(
        executor.calls,
        [
            "class-setup",
            "test-setup",
            "body",
            "test-teardown",
            "class-teardown"
        ]
    );
    assert_eq!(
        outcome.attempt.state.disposition,
        TerminalDisposition::TimedOut
    );
    assert_eq!(outcome.attempt.diagnostics.len(), 2);
}

#[test]
fn cooperative_cancellation_skips_setup_and_body_but_runs_safe_test_teardown() {
    let test_scope = scope(FixtureScope::Test, "test");
    let case = lifecycle_case(
        vec![step(test_scope.clone(), "setup")],
        "body",
        vec![step(test_scope, "teardown")],
    );
    let mut executor = FakeExecutor::default();
    let mut events = Vec::new();
    let mut sink =
        runmat_test::event::SequencedEventSink::new(case.context.run_id.clone(), &mut events);
    let outcome = futures::executor::block_on(
        LifecycleEngine::new(runmat_test::event::RedactionPolicy::new(
            Vec::<String>::new(),
            1024,
        ))
        .execute(&case, &mut executor, &AlwaysCancelled, &mut sink),
    );

    assert_eq!(executor.calls, ["teardown"]);
    assert_eq!(
        outcome.attempt.state.disposition,
        TerminalDisposition::Cancelled
    );
    assert!(!outcome.attempt.state.failed);
    assert!(outcome.attempt.state.incomplete);
}

struct AlwaysCancelled;

impl CancellationProbe for AlwaysCancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

fn qualification(kind: QualificationKind) -> ExecutionResponse {
    ExecutionResponse {
        commands: vec![TestCommand::Qualify {
            qualification: kind,
            diagnostic: Diagnostic {
                identifier: "qualification".into(),
                message: "failed".into(),
                severity: DiagnosticSeverity::Error,
                phase: runmat_test::lifecycle::ExecutionPhase::TestBody,
                source: None,
                details: Vec::new(),
            },
        }],
        output: String::new(),
    }
}
