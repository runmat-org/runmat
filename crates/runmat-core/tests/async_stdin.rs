#![cfg(not(target_arch = "wasm32"))]

use anyhow::Result;
use futures::executor::block_on;
use runmat_builtins::Value;
use runmat_core::{InputRequest, InputRequestKind, InputResponse, RunMatSession};
use runmat_runtime::interaction::force_interactive_stdin_for_tests;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn test_mutex() -> &'static Mutex<()> {
    TEST_MUTEX.get_or_init(|| Mutex::new(()))
}

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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn input_prompts_return_value() -> Result<()> {
    let _test_guard = test_mutex().lock().unwrap();
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let prompts_clone = Arc::clone(&prompts);
    session.install_async_input_handler(move |request: InputRequest| {
        let prompts_clone = Arc::clone(&prompts_clone);
        async move {
            prompts_clone.lock().unwrap().push(request.prompt.clone());
            Ok(InputResponse::Line("41".into()))
        }
    });

    let result =
        block_on(session.execute("value = input('Enter value: '); value = value + 1; value"))
            .map_err(anyhow::Error::new)?;
    let value = result.value.expect("execution should produce a value");
    assert_eq!(value_as_f64(&value), Some(42.0));
    assert_eq!(result.stdin_events.len(), 1);
    assert_eq!(prompts.lock().unwrap().as_slice(), &["Enter value: "]);
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn multiple_inputs_call_handler_in_order() -> Result<()> {
    let _test_guard = test_mutex().lock().unwrap();
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    let responses = Arc::new(Mutex::new(VecDeque::from([
        InputResponse::Line("5".into()),
        InputResponse::Line("code-42".into()),
    ])));
    let responses_clone = Arc::clone(&responses);
    session.install_async_input_handler(move |_request: InputRequest| {
        let responses_clone = Arc::clone(&responses_clone);
        async move {
            let response = responses_clone
                .lock()
                .unwrap()
                .pop_front()
                .expect("missing queued response");
            Ok(response)
        }
    });

    let result = block_on(
        session.execute("first = input('First: '); second = input('Second: ', \"s\"); second"),
    )
    .map_err(anyhow::Error::new)?;
    let value = result
        .value
        .expect("final execution should produce a value");
    assert_eq!(value_as_char_row(&value), Some("code-42".to_string()));
    assert_eq!(result.stdin_events.len(), 2);
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn char_literal_round_trips() -> Result<()> {
    let _test_guard = test_mutex().lock().unwrap();
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    let result = block_on(session.execute("'s'")).map_err(anyhow::Error::new)?;
    let value = result.value.expect("char literal should return a value");
    assert_eq!(value_as_char_row(&value), Some("s".to_string()));
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pause_uses_keypress_handler() -> Result<()> {
    let _test_guard = test_mutex().lock().unwrap();
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    let kinds = Arc::new(Mutex::new(Vec::new()));
    let kinds_clone = Arc::clone(&kinds);
    session.install_async_input_handler(move |request: InputRequest| {
        let kinds_clone = Arc::clone(&kinds_clone);
        async move {
            kinds_clone.lock().unwrap().push(request.kind.clone());
            Ok(InputResponse::KeyPress)
        }
    });

    let result =
        block_on(session.execute("pause; value = 1; value")).map_err(anyhow::Error::new)?;
    let value = result.value.expect("execution should produce a value");
    assert_eq!(value_as_f64(&value), Some(1.0));
    assert_eq!(result.stdin_events.len(), 1);
    assert!(matches!(
        kinds.lock().unwrap().as_slice(),
        [InputRequestKind::KeyPress]
    ));
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_handler_returns_error() -> Result<()> {
    let _test_guard = test_mutex().lock().unwrap();
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    session.install_async_input_handler(|_request: InputRequest| async move {
        Err("input handler is unavailable".to_string())
    });

    let result = block_on(session.execute("pause; value = 1; value"));
    assert!(result.is_err() || result.as_ref().is_ok_and(|res| res.error.is_some()));
    Ok(())
}
