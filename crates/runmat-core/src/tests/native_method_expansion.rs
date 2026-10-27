use crate::{execute_text_request_for_testing, RunMatSession};
use runmat_jit::tiering::TieringConfig;
use runmat_value::Value;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

#[path = "native_method_expansion/pending_managed.rs"]
mod pending_managed;
#[path = "native_method_expansion/pending_operator.rs"]
mod pending_operator;
#[path = "native_method_expansion/native_sequence_truth.rs"]
mod native_sequence_truth;

/// A pending input must wake its waiter when the test releases it. Merely
/// changing an atomic flag is insufficient for a queued native continuation.
struct InputReadiness {
    ready: AtomicBool,
    waker: futures::task::AtomicWaker,
}
impl InputReadiness {
    fn new(ready: bool) -> Self {
        Self {
            ready: AtomicBool::new(ready),
            waker: futures::task::AtomicWaker::new(),
        }
    }
    fn set(&self, ready: bool) {
        self.ready.store(ready, Ordering::SeqCst);
        if ready {
            self.waker.wake();
        }
    }
    async fn wait(&self) {
        std::future::poll_fn(|cx| {
            self.waker.register(cx.waker());
            if self.ready.load(Ordering::SeqCst) {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        })
        .await
    }
}

const COMPOUND_EXPANSION: &str = r#"
function output = first(value)
    output = struct('second', @(next) value + next);
end
s = struct('first', @first);
C = {7};
D = {3};
result = s.first(C{:}).second(D{:});
assert(result == 10);
"#;

const MULTI_ELEMENT_EXPANSION: &str = r#"
function output = first(left, right)
    output = struct('second', @(third, fourth) left + right + third + fourth);
end
s = struct('first', @first);
C = {7, 3};
D = {2, 4};
result = s.first(C{:}).second(D{:});
assert(result == 16);
"#;

const EMPTY_EXPANSION: &str = r#"
function output = first()
    output = struct('second', @() 42);
end
s = struct('first', @first);
C = {};
result = s.first(C{:}).second(C{:});
assert(result == 42);
"#;

const ORDERED_EFFECTS: &str = r#"
function output = first(value)
    global order_counter;
    order_counter = order_counter + 1;
    output = struct('second', @(next) value + next);
end
function output = later()
    global order_counter;
    output = order_counter;
end
global order_counter;
order_counter = 0;
s = struct('first', @first);
C = {7};
result = s.first(C{:}).second(later());
assert(result == 8);
"#;

const ORDERED_EFFECTS_GLOBAL_READ: &str = r#"
function output = first(value)
    global order_counter;
    order_counter = order_counter + 1;
    output = struct('second', @(next) value + next);
end
function output = later()
    global order_counter;
    output = order_counter;
end
global order_counter;
order_counter = 0;
s = struct('first', @first);
C = {7};
result = s.first(C{:}).second(later());
assert(result == 8);
assert(order_counter == 1);
"#;

const PREFIX_ERROR: &str = r#"
function output = first(value, fail)
    if fail
        error('RunMat:FirstInvocation', 'first invocation fails');
    end
    output = struct('second', @(next) value + next);
end
function output = later(fail)
    if fail
        error('RunMat:LaterArgument', 'later argument ran first');
    end
    output = 3;
end
s = struct('first', @first);
C = {7};
result = s.first(C{:}, fail).second(later(fail));
assert(result == 10);
"#;

const PENDING_LATER_ARGUMENT: &str = r#"
function output = first(value)
    global order_counter;
    order_counter = order_counter + 1;
    output = struct('second', @(next) value + next);
end
global order_counter;
order_counter = 0;
s = struct('first', @first);
C = {7};
result = s.first(C{:}).second(input('next: '));
assert(result == 10);
"#;

fn pending_argument_request(session: &RunMatSession) -> crate::abi::ExecutionRequest {
    crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<pending-native-method>".into(),
            text: PENDING_LATER_ARGUMENT.into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    )
}

fn hot_session() -> RunMatSession {
    let mut session = RunMatSession::with_options(true, false).unwrap();
    session.use_generic_ready_route_for_testing();
    session.set_native_tiering_config_for_testing(TieringConfig {
        generic_hot_threshold: 1,
        specialized_hot_threshold: 2,
        deterministic: true,
        ..Default::default()
    });
    session
}

fn poll_until_input<F>(mut execution: std::pin::Pin<&mut F>, calls: &AtomicUsize, expected: usize)
where
    F: Future<Output = crate::abi::ExecutionResponse>,
{
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    for _ in 0..64 {
        assert!(execution.as_mut().poll(&mut context).is_pending());
        if calls.load(Ordering::SeqCst) >= expected {
            return;
        }
    }
    panic!("native invocation did not reach its pending input handler");
}

#[test]
fn hot_native_compound_dotted_calls_preserve_per_segment_expansion() {
    runmat_gc::gc_test_context(|| {
        for jit in [false, true] {
            let mut session = RunMatSession::with_options(jit, false).unwrap();
            if jit {
                session.use_generic_ready_route_for_testing();
                session.set_native_tiering_config_for_testing(TieringConfig {
                    generic_hot_threshold: 1,
                    specialized_hot_threshold: 2,
                    deterministic: true,
                    ..Default::default()
                });
            }
            let mut native_executions = 0;
            for _ in 0..4 {
                let outcome = execute_text_request_for_testing(&mut session, COMPOUND_EXPANSION)
                    .expect("execute compound invocation");
                assert!(outcome.error.is_none(), "{:?}", outcome.error);
                native_executions += usize::from(outcome.used_jit);
                assert_eq!(
                    session.get_variables().unwrap().get("result"),
                    Some(&Value::Num(10.0))
                );
            }
            if jit {
                assert!(
                    native_executions >= 2,
                    "compound invocation must execute native"
                );
            } else {
                assert_eq!(native_executions, 0);
            }
        }
    });
}

#[test]
fn hot_native_dotted_calls_keep_empty_and_multiple_csl_cardinalities() {
    runmat_gc::gc_test_context(|| {
        for (source, expected) in [(MULTI_ELEMENT_EXPANSION, 16.0), (EMPTY_EXPANSION, 42.0)] {
            let mut session = hot_session();
            let mut native_executions = 0;
            for _ in 0..4 {
                let outcome = execute_text_request_for_testing(&mut session, source)
                    .expect("execute dotted invocation with captured sequence");
                assert!(outcome.error.is_none(), "{:?}", outcome.error);
                native_executions += usize::from(outcome.used_jit);
                assert_eq!(
                    session.get_variables().unwrap().get("result"),
                    Some(&Value::Num(expected))
                );
            }
            assert!(
                native_executions >= 2,
                "captured cardinality must execute native"
            );
        }
    });
}

#[test]
fn hot_native_prefix_invocation_precedes_later_effectful_argument_once() {
    runmat_gc::gc_test_context(|| {
        let mut session = hot_session();
        let mut native_executions = 0;
        for _ in 0..4 {
            let outcome = execute_text_request_for_testing(&mut session, ORDERED_EFFECTS)
                .expect("execute ordered effectful invocation");
            assert!(outcome.error.is_none(), "{:?}", outcome.error);
            native_executions += usize::from(outcome.used_jit);
            assert_eq!(
                session.get_variables().unwrap().get("result"),
                Some(&Value::Num(8.0))
            );
        }
        assert!(native_executions >= 2, "effect order must execute native");
    });
}

#[test]
fn hot_native_global_read_reflects_nested_invocation_before_top_level_assertion() {
    runmat_gc::gc_test_context(|| {
        for jit in [false, true] {
            let mut session = RunMatSession::with_options(jit, false).unwrap();
            if jit {
                session.use_generic_ready_route_for_testing();
                session.set_native_tiering_config_for_testing(TieringConfig {
                    generic_hot_threshold: 1,
                    specialized_hot_threshold: 2,
                    deterministic: true,
                    ..Default::default()
                });
            }
            let mut native_executions = 0;
            for _ in 0..4 {
                let outcome =
                    execute_text_request_for_testing(&mut session, ORDERED_EFFECTS_GLOBAL_READ)
                        .expect("execute global alias synchronization witness");
                assert!(outcome.error.is_none(), "{:?}", outcome.error);
                native_executions += usize::from(outcome.used_jit);
                assert_eq!(
                    session.get_variables().unwrap().get("result"),
                    Some(&Value::Num(8.0))
                );
            }
            if jit {
                assert!(native_executions >= 2, "global read must execute native");
            } else {
                assert_eq!(native_executions, 0);
            }
        }
    });
}

#[test]
fn warmed_prefix_failure_precedes_later_argument_preparation() {
    runmat_gc::gc_test_context(|| {
        let mut session = hot_session();
        let seed = execute_text_request_for_testing(&mut session, "fail = false;").unwrap();
        assert!(seed.error.is_none(), "{:?}", seed.error);
        let mut native_executions = 0;
        for _ in 0..4 {
            let outcome = execute_text_request_for_testing(&mut session, PREFIX_ERROR)
                .expect("execute ordered dotted invocation");
            native_executions += usize::from(outcome.used_jit);
            assert!(outcome.error.is_none(), "{:?}", outcome.error);
            assert_eq!(
                session.get_variables().unwrap().get("result"),
                Some(&Value::Num(10.0))
            );
        }
        assert!(
            native_executions >= 2,
            "ordered call must execute native before the failure check"
        );
        let seed = execute_text_request_for_testing(&mut session, "fail = true;").unwrap();
        assert!(seed.error.is_none(), "{:?}", seed.error);
        let before = session.stats().jit_compiled;
        let outcome = execute_text_request_for_testing(&mut session, PREFIX_ERROR)
            .expect("execute warmed ordered dotted invocation");
        assert_eq!(
            outcome.error.as_ref().and_then(|error| error.identifier()),
            Some("RunMat:FirstInvocation")
        );
        // Session metadata records native selection before invoking the entry.
        // The successful runs above prove actual native execution; this error
        // path proves ordering after warmup, not a native-entry observation.
        assert_eq!(
            session.stats().jit_compiled,
            before + 1,
            "failing invocation must select the already-warmed native product"
        );
    });
}

#[test]
fn hot_native_dotted_invocation_retains_prefix_across_pending_later_argument() {
    runmat_gc::gc_test_context(|| {
        let mut session = hot_session();
        let ready = Arc::new(InputReadiness::new(true));
        let calls = Arc::new(AtomicUsize::new(0));
        let prefix_seen = Arc::new(AtomicBool::new(false));
        session.install_async_input_handler({
            let ready = Arc::clone(&ready);
            let calls = Arc::clone(&calls);
            let prefix_seen = Arc::clone(&prefix_seen);
            move |_| {
                let ready = Arc::clone(&ready);
                let calls = Arc::clone(&calls);
                let prefix_seen = Arc::clone(&prefix_seen);
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    if runmat_runtime::workspace::session::global_value("order_counter")
                        == Some(Value::Num(1.0))
                    {
                        prefix_seen.store(true, Ordering::SeqCst);
                    }
                    ready.wait().await;
                    Ok(crate::execution::InputResponse::Line("3".into()))
                }
            }
        });

        let mut native_executions = 0;
        for _ in 0..4 {
            let request = pending_argument_request(&session);
            let response = futures::executor::block_on(session.execute_request(request));
            let outcome = response.result.expect("warm dotted invocation");
            assert!(
                outcome
                    .diagnostics
                    .iter()
                    .all(|diagnostic| diagnostic.severity != crate::abi::DiagnosticSeverity::Error),
                "{:?}",
                outcome.diagnostics
            );
            native_executions += usize::from(outcome.used_jit);
            assert_eq!(
                session.get_variables().unwrap().get("result"),
                Some(&Value::Num(10.0))
            );
        }
        assert!(native_executions >= 2, "warm calls must execute native");
        assert!(prefix_seen.load(Ordering::SeqCst));

        ready.set(false);
        prefix_seen.store(false, Ordering::SeqCst);
        let before = calls.load(Ordering::SeqCst);
        let request = pending_argument_request(&session);
        let mut execution = Box::pin(session.execute_request(request));
        poll_until_input(execution.as_mut(), &calls, before + 1);
        assert_eq!(calls.load(Ordering::SeqCst), before + 1);
        assert!(prefix_seen.load(Ordering::SeqCst));
        runmat_gc::gc_collect_major().expect("collect at pending method argument");
        ready.set(true);
        let response = futures::executor::block_on(execution);
        let outcome = response.result.expect("resume native method call");
        assert!(outcome.used_jit, "resumed invocation must execute native");
        assert!(
            outcome
                .diagnostics
                .iter()
                .all(|diagnostic| diagnostic.severity != crate::abi::DiagnosticSeverity::Error),
            "{:?}",
            outcome.diagnostics
        );
        assert_eq!(calls.load(Ordering::SeqCst), before + 1, "no input replay");
        assert_eq!(
            session.get_variables().unwrap().get("result"),
            Some(&Value::Num(10.0))
        );

        ready.set(false);
        let before_drop = calls.load(Ordering::SeqCst);
        let request = pending_argument_request(&session);
        let mut abandoned = Box::pin(session.execute_request(request));
        poll_until_input(abandoned.as_mut(), &calls, before_drop + 1);
        assert_eq!(calls.load(Ordering::SeqCst), before_drop + 1);
        drop(abandoned);
        runmat_gc::gc_collect_major().expect("collect after dropping pending invocation");
        assert_eq!(calls.load(Ordering::SeqCst), before_drop + 1);
        ready.set(true);
        let request = pending_argument_request(&session);
        let response = futures::executor::block_on(session.execute_request(request));
        let recovered = response
            .result
            .expect("execute after dropped pending invocation");
        assert!(
            recovered
                .diagnostics
                .iter()
                .all(|diagnostic| diagnostic.severity != crate::abi::DiagnosticSeverity::Error),
            "{:?}",
            recovered.diagnostics
        );
        assert!(recovered.used_jit, "recovery must execute native");
        assert_eq!(calls.load(Ordering::SeqCst), before_drop + 2);
    });
}
