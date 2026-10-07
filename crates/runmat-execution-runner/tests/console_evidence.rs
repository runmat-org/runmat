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
