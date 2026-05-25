#![cfg(not(target_arch = "wasm32"))]

use anyhow::Result;
use runmat_builtins::Value;
use runmat_core::{InputRequest, InputRequestKind, InputResponse, RunError, RunMatSession};
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

    let result = runmat_core::execute_text_request_for_testing(
        &mut session,
        "value = input('Enter value: '); value = value + 1; value",
    )
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

    let result = runmat_core::execute_text_request_for_testing(
        &mut session,
        "first = input('First: '); second = input('Second: ', \"s\"); second",
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
    let result = runmat_core::execute_text_request_for_testing(&mut session, "'s'")
        .map_err(anyhow::Error::new)?;
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
        runmat_core::execute_text_request_for_testing(&mut session, "pause; value = 1; value")
            .map_err(anyhow::Error::new)?;
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

    let result =
        runmat_core::execute_text_request_for_testing(&mut session, "pause; value = 1; value");
    match result {
        Err(RunError::Runtime(err)) => {
            assert_eq!(
                err.identifier(),
                Some("RunMat:interaction:AsyncHandlerError")
            );
        }
        Err(other) => panic!("expected runtime interaction error, got: {other:?}"),
        Ok(exec) => {
            let err = exec.error.expect("expected execution-level runtime error");
            assert_eq!(
                err.identifier(),
                Some("RunMat:interaction:AsyncHandlerError")
            );
        }
    }
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn spawn_of_async_function_triggers_pause_handler_before_await() -> Result<()> {
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

    let result = runmat_core::execute_text_request_for_testing(
        &mut session,
        "async function y = wait_for_key(); pause; y = 1; end; \
         t = spawn(wait_for_key()); marker = 7;",
    )
    .map_err(anyhow::Error::new)?;
    assert!(
        result.error.is_none(),
        "spawn of async function with pause handler should not raise runtime errors"
    );
    assert_eq!(
        result.stdin_events.len(),
        1,
        "spawn should trigger pause interaction before await in current runtime model"
    );
    assert!(matches!(
        kinds.lock().unwrap().as_slice(),
        [InputRequestKind::KeyPress]
    ));

    let marker = runmat_core::execute_text_request_for_testing(&mut session, "marker")
        .map_err(anyhow::Error::new)?;
    let marker_value = marker
        .value
        .expect("marker readback should produce a value");
    assert_eq!(value_as_f64(&marker_value), Some(7.0));
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn parallel_spawn_inputs_follow_spawn_order_not_await_order() -> Result<()> {
    let _test_guard = test_mutex().lock().unwrap();
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let prompts_clone = Arc::clone(&prompts);
    let responses = Arc::new(Mutex::new(VecDeque::from([
        InputResponse::Line("11".into()),
        InputResponse::Line("22".into()),
    ])));
    let responses_clone = Arc::clone(&responses);
    session.install_async_input_handler(move |request: InputRequest| {
        let prompts_clone = Arc::clone(&prompts_clone);
        let responses_clone = Arc::clone(&responses_clone);
        async move {
            prompts_clone.lock().unwrap().push(request.prompt.clone());
            let response = responses_clone
                .lock()
                .unwrap()
                .pop_front()
                .expect("missing queued response");
            Ok(response)
        }
    });

    let result = runmat_core::execute_text_request_for_testing(
        &mut session,
        "async function y = first(); input('first: '); y = 1; end; \
         async function y = second(); input('second: '); y = 2; end; \
         t1 = spawn(first()); t2 = spawn(second()); out2 = await(t2); out1 = await(t1);",
    )
    .map_err(anyhow::Error::new)?;
    assert!(
        result.error.is_none(),
        "parallel spawn/await input flow should not raise runtime errors"
    );
    assert_eq!(result.stdin_events.len(), 2);
    assert_eq!(prompts.lock().unwrap().as_slice(), &["first: ", "second: "]);

    let out1 = runmat_core::execute_text_request_for_testing(&mut session, "out1")
        .map_err(anyhow::Error::new)?;
    let out1_value = out1.value.expect("out1 readback should produce a value");
    assert_eq!(value_as_f64(&out1_value), Some(1.0));

    let out2 = runmat_core::execute_text_request_for_testing(&mut session, "out2")
        .map_err(anyhow::Error::new)?;
    let out2_value = out2.value.expect("out2 readback should produce a value");
    assert_eq!(value_as_f64(&out2_value), Some(2.0));
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn spawn_error_stops_later_spawn_from_running() -> Result<()> {
    let _test_guard = test_mutex().lock().unwrap();
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let prompts_clone = Arc::clone(&prompts);
    session.install_async_input_handler(move |request: InputRequest| {
        let prompts_clone = Arc::clone(&prompts_clone);
        async move {
            prompts_clone.lock().unwrap().push(request.prompt.clone());
            Err("handler failed".to_string())
        }
    });

    let result = runmat_core::execute_text_request_for_testing(
        &mut session,
        "async function y = first(); input('first: '); y = 1; end; \
         async function y = second(); input('second: '); y = 2; end; \
         t1 = spawn(first()); t2 = spawn(second()); marker = 7;",
    );

    match result {
        Err(RunError::Runtime(err)) => {
            assert_eq!(err.identifier(), Some("RunMat:input:InteractionFailed"));
        }
        Err(other) => panic!("expected runtime interaction error, got: {other:?}"),
        Ok(exec) => {
            let err = exec.error.expect("expected execution-level runtime error");
            assert_eq!(err.identifier(), Some("RunMat:input:InteractionFailed"));
        }
    }
    assert_eq!(
        prompts.lock().unwrap().as_slice(),
        &["first: "],
        "second spawn should not run after first spawn interaction failure"
    );
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn async_call_without_await_or_spawn_stays_lazy_until_await() -> Result<()> {
    let _test_guard = test_mutex().lock().unwrap();
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let prompts_clone = Arc::clone(&prompts);
    session.install_async_input_handler(move |request: InputRequest| {
        let prompts_clone = Arc::clone(&prompts_clone);
        async move {
            prompts_clone.lock().unwrap().push(request.prompt.clone());
            Ok(InputResponse::Line("11".into()))
        }
    });

    let result = runmat_core::execute_text_request_for_testing(
        &mut session,
        "async function y = asks(); input('lazy: '); y = 1; end; \
         fut = asks(); marker = 9;",
    )
    .map_err(anyhow::Error::new)?;
    assert!(
        result.error.is_none(),
        "direct async call should not raise runtime errors in current runtime model"
    );
    assert_eq!(
        result.stdin_events.len(),
        0,
        "direct async call should remain lazy before await/spawn boundaries"
    );
    assert!(
        prompts.lock().unwrap().is_empty(),
        "direct async call should not prompt until awaited or spawned"
    );

    let marker = runmat_core::execute_text_request_for_testing(&mut session, "marker")
        .map_err(anyhow::Error::new)?;
    let marker_value = marker
        .value
        .expect("marker readback should produce a value");
    assert_eq!(value_as_f64(&marker_value), Some(9.0));

    let fut = runmat_core::execute_text_request_for_testing(&mut session, "out = await(fut);")
        .map_err(anyhow::Error::new)?;
    assert!(
        fut.error.is_none(),
        "awaiting deferred future should complete without runtime error"
    );
    let out = runmat_core::execute_text_request_for_testing(&mut session, "out")
        .map_err(anyhow::Error::new)?;
    let fut_value = out
        .value
        .expect("awaited future result should be materialized in out");
    assert_eq!(
        value_as_f64(&fut_value),
        Some(1.0),
        "await should resolve the deferred async function call"
    );
    assert_eq!(
        prompts.lock().unwrap().as_slice(),
        &["lazy: "],
        "await should trigger the deferred interaction once"
    );
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn deferred_future_triggers_interaction_when_spawned() -> Result<()> {
    let _test_guard = test_mutex().lock().unwrap();
    let _guard = InteractiveGuard::new();
    let mut session = RunMatSession::with_options(false, false)?;
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let prompts_clone = Arc::clone(&prompts);
    session.install_async_input_handler(move |request: InputRequest| {
        let prompts_clone = Arc::clone(&prompts_clone);
        async move {
            prompts_clone.lock().unwrap().push(request.prompt.clone());
            Ok(InputResponse::Line("5".into()))
        }
    });

    let deferred = runmat_core::execute_text_request_for_testing(
        &mut session,
        "async function y = asks(); input('spawned: '); y = 5; end; \
         fut = asks(); marker = 3;",
    )
    .map_err(anyhow::Error::new)?;
    assert!(
        deferred.error.is_none(),
        "deferred async call should compile/execute without runtime errors"
    );
    assert_eq!(
        deferred.stdin_events.len(),
        0,
        "future creation should remain lazy before spawn/await boundaries"
    );
    assert!(
        prompts.lock().unwrap().is_empty(),
        "future creation should not trigger input prompts"
    );

    let spawned = runmat_core::execute_text_request_for_testing(
        &mut session,
        "t = spawn(fut); marker_after_spawn = 7;",
    )
    .map_err(anyhow::Error::new)?;
    assert!(
        spawned.error.is_none(),
        "spawning deferred future should not raise runtime errors"
    );
    assert_eq!(
        spawned.stdin_events.len(),
        1,
        "spawn should realize deferred future and trigger input interaction"
    );
    assert_eq!(
        prompts.lock().unwrap().as_slice(),
        &["spawned: "],
        "spawn should forward deferred future prompt through input handler"
    );

    let marker = runmat_core::execute_text_request_for_testing(&mut session, "marker_after_spawn")
        .map_err(anyhow::Error::new)?;
    let marker_value = marker
        .value
        .expect("spawn follow-up marker should be readable");
    assert_eq!(value_as_f64(&marker_value), Some(7.0));
    Ok(())
}
