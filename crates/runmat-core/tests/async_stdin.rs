#![cfg(not(target_arch = "wasm32"))]

use anyhow::Result;
use runmat_builtins::Value;
use runmat_core::{
    InputHandlerAction, InputRequestKind, InputResponse, PendingInput, RunMatSession,
};
use runmat_runtime::interaction::force_interactive_stdin_for_tests;

struct InteractiveGuard;

impl InteractiveGuard {
    fn new() -> Self {
        force_interactive_stdin_for_tests(true);
        Self
    }
}

impl Drop for InteractiveGuard {
    fn drop(&mut self) {
        force_interactive_stdin_for_tests(false);
    }
}

fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        _ => None,
    }
}

fn value_as_char_row(value: &Value) -> Option<String> {
    match value {
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn expect_pending(input: &Option<PendingInput>) -> &PendingInput {
    input.as_ref().expect("execution should suspend for stdin")
}

fn assert_line_request(kind: &InputRequestKind) {
    match kind {
        InputRequestKind::Line { echo } => assert!(*echo, "line prompts should echo by default"),
        other => panic!("expected line request, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn input_prompts_suspend_and_resume() -> Result<()> {
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    session.install_input_handler(|_| InputHandlerAction::Pending);

    let first = session.execute("value = input('Enter value: '); value = value + 1; value;")?;
    let pending = expect_pending(&first.stdin_requested);
    assert_eq!(pending.request.prompt, "Enter value: ");
    assert_line_request(&pending.request.kind);
    assert_eq!(1, session.pending_requests().len());

    let resumed = session.resume_input(pending.id, Ok(InputResponse::Line("41".into())))?;
    assert!(resumed.stdin_requested.is_none());
    assert!(session.pending_requests().is_empty());
    let value = resumed.value.expect("execution should produce a value");
    assert_eq!(value_as_f64(&value), Some(42.0));
    assert_eq!(resumed.stdin_events.len(), 1);
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn multiple_inputs_queue_separate_requests() -> Result<()> {
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    session.install_input_handler(|_| InputHandlerAction::Pending);

    let first =
        session.execute("first = input('First: '); second = input('Second: ', \"s\"); second;")?;
    let pending_a = expect_pending(&first.stdin_requested);
    assert_eq!(pending_a.request.prompt, "First: ");
    assert_line_request(&pending_a.request.kind);

    let second = session.resume_input(pending_a.id, Ok(InputResponse::Line("5".into())))?;
    let pending_b = expect_pending(&second.stdin_requested);
    assert_eq!(pending_b.request.prompt, "Second: ");
    assert_line_request(&pending_b.request.kind);
    assert_eq!(1, session.pending_requests().len());

    let final_result =
        session.resume_input(pending_b.id, Ok(InputResponse::Line("code-42".into())))?;
    assert!(final_result.stdin_requested.is_none());
    assert!(session.pending_requests().is_empty());
    let value = final_result
        .value
        .expect("final execution should produce a value");
    assert_eq!(value_as_char_row(&value), Some("code-42".to_string()));
    assert_eq!(final_result.stdin_events.len(), 2);
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn char_literal_round_trips() -> Result<()> {
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    let result = session.execute("'s'")?;
    let value = result.value.expect("char literal should return a value");
    assert_eq!(value_as_char_row(&value), Some("s".to_string()));
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pause_without_args_suspends_and_resumes() -> Result<()> {
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    session.install_input_handler(|_| InputHandlerAction::Pending);

    let first = session.execute("pause; value = 1; value;")?;
    let pending = expect_pending(&first.stdin_requested);
    assert!(matches!(pending.request.kind, InputRequestKind::KeyPress));
    assert_eq!(1, session.pending_requests().len());

    let resumed = session.resume_input(pending.id, Ok(InputResponse::KeyPress))?;
    assert!(resumed.stdin_requested.is_none());
    assert!(session.pending_requests().is_empty());
    let value = resumed.value.expect("execution should produce a value");
    assert_eq!(value_as_f64(&value), Some(1.0));
    assert_eq!(resumed.stdin_events.len(), 1);
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn multiple_pauses_queue_separate_requests() -> Result<()> {
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    session.install_input_handler(|_| InputHandlerAction::Pending);

    let first = session.execute("pause; pause; 7;")?;
    let pending_a = expect_pending(&first.stdin_requested);
    assert!(matches!(pending_a.request.kind, InputRequestKind::KeyPress));

    let second = session.resume_input(pending_a.id, Ok(InputResponse::KeyPress))?;
    let pending_b = expect_pending(&second.stdin_requested);
    assert!(matches!(pending_b.request.kind, InputRequestKind::KeyPress));
    assert_eq!(1, session.pending_requests().len());

    let final_result = session.resume_input(pending_b.id, Ok(InputResponse::KeyPress))?;
    assert!(final_result.stdin_requested.is_none());
    assert!(session.pending_requests().is_empty());
    let value = final_result
        .value
        .expect("final execution should produce a value");
    assert_eq!(value_as_f64(&value), Some(7.0));
    assert_eq!(final_result.stdin_events.len(), 2);
    Ok(())
}
