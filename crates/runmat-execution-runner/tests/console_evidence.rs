mod common;
use runmat_execution::{console::*, state::TaskState, task::RetryPolicy};
use runmat_execution_runner::{
    port::BackendReport, AttemptFailureKind, AttemptReport, DriverCommand,
};

fn transcript(complete: bool) -> ConsoleTranscript {
    let mut capture = ConsoleCapture::new(ConsoleLimits {
        max_records: 4,
        max_text_bytes: 100,
    })
    .unwrap();
    capture
        .push(ConsoleStream::Stdout, "original prefix".into())
        .unwrap();
    capture.finish(complete)
}

fn admit(fixture: &mut common::Fixture) -> runmat_execution_runner::AttemptRequest {
    let mut task = common::task(
        "console-authority",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    task.request.console = Some(transcript(true).limits());
    task.request.resources.memory_bytes = 16384;
    task.request.resources.max_artifact_bytes = 4096;
    common::submit(&mut fixture.driver, task)
}

#[test]
fn requested_console_blocks_success_without_complete_original_evidence() {
    for mode in 0..3 {
        let mut fixture = common::fixture(1, 1);
        let request = admit(&mut fixture);
        let mut report = BackendReport::for_request(&request, common::success(&request));
        report.console = match mode {
            0 => None,
            1 => Some(Box::new(transcript(false))),
            _ => Some(Box::new(transcript(true))),
        };
        fixture
            .driver
            .handle(DriverCommand::BackendReport(report))
            .unwrap();
        let cold: runmat_execution_runner::DriverSnapshot =
            serde_json::from_slice(&serde_json::to_vec(&fixture.driver.snapshot()).unwrap())
                .unwrap();
        cold.validate().unwrap();
        assert_eq!(
            cold.tasks[&request.task_id].state,
            if mode == 2 {
                TaskState::Succeeded
            } else {
                TaskState::Failed
            }
        );
        if mode == 2 {
            let retained = cold.attempts[&request.id]
                .terminal
                .as_ref()
                .unwrap()
                .console
                .as_ref()
                .unwrap();
            assert_eq!(retained.as_ref(), &transcript(true));
        }
    }
}

#[test]
fn stale_prefix_is_discarded_and_original_failed_prefix_survives_cold_history() {
    let mut fixture = common::fixture(1, 1);
    let request = admit(&mut fixture);
    let mut report = BackendReport::for_request(
        &request,
        AttemptReport::Failed {
            kind: AttemptFailureKind::Execution,
            message: "original failure".into(),
        },
    );
    report.console = Some(Box::new(transcript(false)));
    let mut stale = report.clone();
    stale.driver_fence += 1;
    fixture
        .driver
        .handle(DriverCommand::BackendReport(stale))
        .unwrap();
    assert!(fixture.driver.snapshot().attempts[&request.id]
        .terminal
        .is_none());
    fixture
        .driver
        .handle(DriverCommand::BackendReport(report))
        .unwrap();
    let cold: runmat_execution_runner::DriverSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&fixture.driver.snapshot()).unwrap()).unwrap();
    cold.validate().unwrap();
    assert_eq!(
        cold.attempts[&request.id]
            .terminal
            .as_ref()
            .unwrap()
            .console
            .as_deref(),
        Some(&transcript(false))
    );
    let mut wire = serde_json::to_value(&cold).unwrap();
    let key = request.id.to_string();
    wire["attempts"][&key]["terminal"]["console"]["limits"]["max_text_bytes"] =
        serde_json::json!(101);
    let altered: runmat_execution_runner::DriverSnapshot = serde_json::from_value(wire).unwrap();
    assert!(altered.validate().is_err());
}

#[test]
fn delegated_graph_success_commits_its_original_physical_console_root() {
    use runmat_execution::value::{
        GraphNode, GraphNodeRef, InlineValue, PayloadGraph, ValuePayload,
    };
    let mut fixture = common::fixture(1, 1);
    let request = admit(&mut fixture);
    let graph = PayloadGraph {
        roots: vec![GraphNodeRef(0)],
        nodes: vec![GraphNode::Leaf {
            value: ValuePayload::Inline(Box::new(InlineValue::Null)),
        }],
    };
    let mut report = BackendReport::for_request(
        &request,
        AttemptReport::Succeeded {
            result: runmat_execution_runner::AttemptSuccess::new_graph(&request, graph, 1, 0, 0)
                .unwrap(),
        },
    );
    let root = ScopedConsoleTranscript {
        scope_id: runmat_execution::ExecutionScopeId::derive(&[b"physical-root"]),
        pool_id: runmat_execution::PoolId::derive(&[b"physical-root"]),
        task_id: runmat_execution::TaskId::derive(&[b"physical-root"]),
        attempt_id: runmat_execution::identity::AttemptId::derive(&[b"physical-root"]),
        transcript: Box::new(transcript(true)),
    };
    report.delegated_console_root = Some(root.identity());
    report.console_scopes.push(root.clone());
    fixture
        .driver
        .handle(DriverCommand::BackendReport(report))
        .unwrap();
    let cold: runmat_execution_runner::DriverSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&fixture.driver.snapshot()).unwrap()).unwrap();
    cold.validate().unwrap();
    assert_eq!(cold.tasks[&request.task_id].state, TaskState::Succeeded);
    let terminal = cold.attempts[&request.id].terminal.as_ref().unwrap();
    assert_eq!(terminal.delegated_console_root, Some(root.identity()));
    assert!(terminal.console.is_none());
    assert_eq!(terminal.console_scopes, [root]);
}

#[test]
fn cold_uncertainty_requires_the_original_delegated_root_source() {
    let mut fixture = common::fixture(1, 1);
    let request = admit(&mut fixture);
    let mut report = BackendReport::for_request(
        &request,
        AttemptReport::Lost {
            message: "physical retirement is unverified".into(),
        },
    );
    report.console = Some(Box::new(transcript(false)));
    fixture
        .driver
        .handle(DriverCommand::BackendReport(report))
        .unwrap();
    let mut cold = fixture.driver.snapshot();
    cold.validate().unwrap();
    cold.attempts
        .get_mut(&request.id)
        .unwrap()
        .unverified_retirement
        .as_mut()
        .unwrap()
        .delegated_console_root = Some(ConsoleSourceIdentity {
        scope_id: request.scope_id,
        pool_id: request.task.pool_id,
        task_id: request.task_id,
        attempt_id: runmat_execution::identity::AttemptId::derive(&[b"unrecorded physical source"]),
    });
    assert!(cold.validate().is_err());
}
