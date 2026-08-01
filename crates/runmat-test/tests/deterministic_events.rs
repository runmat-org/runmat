mod common;

use common::{execute, lifecycle_case, run_id, test_id, FakeExecutor};
use runmat_test::event::{
    replay, RedactionPolicy, SequencedEventSink, TestEvent, TestEventPayload,
};
use runmat_test::result::{
    aggregate_run_state, merge_attempts, AttemptResult, ResultState, RunResult,
};

#[test]
fn event_encoding_is_deterministic_and_replay_recovers_terminal_results() {
    let case = lifecycle_case(Vec::new(), "body", Vec::new());
    let mut executor = FakeExecutor::default();
    let (outcome, lifecycle_events) = execute(&case, &mut executor);
    let mut events = Vec::new();
    let mut sink = SequencedEventSink::new(run_id(), &mut events);
    sink.emit(TestEventPayload::RunStarted);
    for event in lifecycle_events {
        sink.emit(event.payload);
    }
    let test_result = merge_attempts(test_id("example"), vec![outcome.attempt.clone()]).unwrap();
    let run_result = RunResult {
        run_id: run_id(),
        state: aggregate_run_state([&test_result.state]),
        tests: vec![test_result],
    };
    sink.emit(TestEventPayload::RunFinished {
        result: run_result.clone(),
    });
    drop(sink);

    let encoded = serde_json::to_vec(&events).unwrap();
    assert_eq!(encoded, serde_json::to_vec(&events).unwrap());
    let decoded: Vec<TestEvent> = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(events, decoded);
    let replayed = replay(&events).unwrap();
    assert_eq!(replayed.result, run_result);
    assert_eq!(replayed.attempts, vec![outcome.attempt]);
}

#[test]
fn redaction_is_secret_first_utf8_safe_and_bounded() {
    let policy = RedactionPolicy::new(["token-long".into(), "token".into()], 13);
    let redacted = policy.redact("é token-long trailing");
    assert_eq!(redacted.text, "é [REDACTED]");
    assert!(redacted.truncated);
}

#[test]
fn latest_attempt_is_authoritative_and_prior_failure_marks_flaky_pass() {
    let id = test_id("retry");
    let attempt = |attempt, state| AttemptResult {
        test_id: id.clone(),
        attempt,
        state,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
        output: String::new(),
        abort_run: false,
    };
    let result = merge_attempts(
        id.clone(),
        vec![
            attempt(
                1,
                ResultState {
                    failed: true,
                    incomplete: false,
                    disposition: runmat_test::result::TerminalDisposition::Failed,
                },
            ),
            attempt(2, ResultState::PASSED),
        ],
    )
    .unwrap();
    assert_eq!(result.state, ResultState::PASSED);
    assert!(result.flaky);
    assert_eq!(result.attempts.len(), 2);
}
