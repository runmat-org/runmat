#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_runtime::interaction::{
    replace_async_handler, request_line_async, AsyncInteractionHandler, InteractionResponse,
};
use runmat_runtime::workspace::session::{global_value, update_global_slot};
use runmat_value::Value;
use std::future::Future;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct NoopWake;

impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

async fn session(label: &'static str) {
    let handler: Arc<AsyncInteractionHandler> = Arc::new(move |_| {
        Box::pin(async move { Ok(InteractionResponse::Line(label.to_string())) })
    });
    let _handler = replace_async_handler(Some(handler));
    let mut yielded = false;
    std::future::poll_fn(move |_| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            Poll::Pending
        }
    })
    .await;
    assert_eq!(request_line_async("", false).await.unwrap(), label);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn interleaved_runtime_contexts_retain_their_host_services() {
    let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let mut first_future = Box::pin(first.scope(session("first")));
    let mut second_future = Box::pin(second.scope(session("second")));
    let waker = Waker::from(Arc::new(NoopWake));
    let mut cx = Context::from_waker(&waker);

    assert!(first_future.as_mut().poll(&mut cx).is_pending());
    assert!(second_future.as_mut().poll(&mut cx).is_pending());
    assert!(first_future.as_mut().poll(&mut cx).is_ready());
    assert!(second_future.as_mut().poll(&mut cx).is_ready());
}

async fn session_variable(label: &'static str, value: f64) {
    update_global_slot(0, Some(label), &Value::Num(value));
    let mut yielded = false;
    std::future::poll_fn(move |_| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            Poll::Pending
        }
    })
    .await;
    assert_eq!(global_value(label), Some(Value::Num(value)));
    let other = if label == "first" { "second" } else { "first" };
    assert_eq!(global_value(other), None);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn interleaved_runtime_contexts_isolate_session_variables() {
    let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let mut first_future = Box::pin(first.scope(session_variable("first", 1.0)));
    let mut second_future = Box::pin(second.scope(session_variable("second", 2.0)));
    let waker = Waker::from(Arc::new(NoopWake));
    let mut cx = Context::from_waker(&waker);

    assert!(first_future.as_mut().poll(&mut cx).is_pending());
    assert!(second_future.as_mut().poll(&mut cx).is_pending());
    assert!(first_future.as_mut().poll(&mut cx).is_ready());
    assert!(second_future.as_mut().poll(&mut cx).is_ready());
}
