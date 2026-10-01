//! One browser task turn for cooperative CPU execution.
//!
//! A resolved Promise only schedules another microtask and can starve the
//! worker's message queue. MessageChannel posts a real task instead.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

use js_sys::Reflect;
use runmat_runtime::execution::ExecutionServiceError;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use web_sys::{MessageChannel, MessageEvent, MessagePort};

#[cfg(test)]
thread_local! {
    static TEST_POSTED_TURNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Default)]
struct TurnState {
    ready: bool,
    waker: Option<Waker>,
}

pub(super) struct BrowserTaskTurn {
    receive: MessagePort,
    send: MessagePort,
    _callback: Closure<dyn FnMut(MessageEvent)>,
    state: Rc<RefCell<TurnState>>,
    posted: bool,
}

impl BrowserTaskTurn {
    pub(super) fn new() -> Result<Self, ExecutionServiceError> {
        let channel = MessageChannel::new().map_err(schedule_error)?;
        let receive = channel.port1();
        let send = channel.port2();
        let state = Rc::new(RefCell::new(TurnState::default()));
        let callback_state = Rc::clone(&state);
        let callback = Closure::wrap(Box::new(move |_event: MessageEvent| {
            let waker = {
                let mut state = callback_state.borrow_mut();
                state.ready = true;
                state.waker.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        receive.set_onmessage(Some(callback.as_ref().unchecked_ref()));
        receive.start();
        Ok(Self {
            receive,
            send,
            _callback: callback,
            state,
            posted: false,
        })
    }
}

impl Future for BrowserTaskTurn {
    type Output = Result<(), ExecutionServiceError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.borrow_mut();
        if state.ready {
            return Poll::Ready(Ok(()));
        }
        state.waker = Some(context.waker().clone());
        drop(state);
        if !self.posted {
            // Mark before posting so a failed post never causes a second
            // submission. The owned future is dropped on error.
            self.posted = true;
            if let Err(error) = self.send.post_message(&JsValue::UNDEFINED) {
                return Poll::Ready(Err(schedule_error(error)));
            }
            #[cfg(test)]
            TEST_POSTED_TURNS.with(|count| count.set(count.get() + 1));
        }
        Poll::Pending
    }
}

impl Drop for BrowserTaskTurn {
    fn drop(&mut self) {
        self.receive.set_onmessage(None);
        self.receive.close();
        self.send.close();
        self.state.borrow_mut().waker = None;
        // The closure is retained as a field until after this listener is removed.
    }
}

fn schedule_error(error: JsValue) -> ExecutionServiceError {
    let name = Reflect::get(&error, &JsValue::from_str("name"))
        .ok()
        .and_then(|value| value.as_string());
    let message = error.as_string().or_else(|| {
        Reflect::get(&error, &JsValue::from_str("message"))
            .ok()
            .and_then(|value| value.as_string())
    });
    let cause = match (name, message) {
        (Some(name), Some(message)) => format!("{name}: {message}"),
        (None, Some(message)) => message,
        _ => format!("{error:?}"),
    };
    ExecutionServiceError::Infrastructure(format!(
        "browser CPU task-turn scheduling failed: {cause}"
    ))
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use futures::task::noop_waker;
    use serde::Serialize;
    use serde_json::json;
    use std::future::Future;
    use std::task::{Context, Poll};
    use wasm_bindgen_test::wasm_bindgen_test;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test(async)]
    async fn cpu_turn_requires_and_completes_a_browser_message_task() {
        let mut turn = Box::pin(BrowserTaskTurn::new().expect("browser MessageChannel"));
        let waker = noop_waker();
        assert!(turn
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        assert!(!turn.as_ref().get_ref().state.borrow().ready);
        assert!(matches!(turn.await, Ok(())));
    }

    #[wasm_bindgen_test(async)]
    async fn same_worker_cancel_execution_interrupts_a_running_multinomial_fit() {
        let runtime = Rc::new(
            crate::init_runmat(JsValue::NULL)
                .await
                .expect("initialize browser runtime"),
        );
        let request = json!({
            "source": {
                "kind": "text",
                "name": "mnrfit-browser-cancel.m",
                "text": "X = (1:65536)'; Y = mod(X,2)+1; B = mnrfit(X,Y,'IterationLimit',100000);"
            },
            "compatibility": "runmat"
        })
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .expect("encode execution request");
        let turns_before = TEST_POSTED_TURNS.with(std::cell::Cell::get);
        let mut execution = Box::pin(runtime.execute_request_js(request));
        let waker = noop_waker();
        match execution.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Pending => {}
            Poll::Ready(Ok(payload)) => {
                let diagnostic = js_sys::JSON::stringify(&payload)
                    .ok()
                    .and_then(|value| value.as_string());
                panic!("execution completed before the fit's cooperative turn: {diagnostic:?}");
            }
            Poll::Ready(Err(error)) => {
                panic!("execution failed before the fit's cooperative turn: {error:?}");
            }
        }
        assert!(
            TEST_POSTED_TURNS.with(std::cell::Cell::get) > turns_before,
            "the fit must post a browser task turn before cancellation"
        );
        let cancel_channel = MessageChannel::new().expect("browser cancellation task channel");
        let cancel_receive = cancel_channel.port1();
        let cancel_send = cancel_channel.port2();
        let (canceled_tx, canceled_rx) = futures::channel::oneshot::channel();
        let canceled_tx = Rc::new(RefCell::new(Some(canceled_tx)));
        let cancel_runtime = Rc::clone(&runtime);
        let cancel_callback = Closure::wrap(Box::new(move |_event: MessageEvent| {
            cancel_runtime.cancel_execution();
            if let Some(sender) = canceled_tx.borrow_mut().take() {
                let _ = sender.send(());
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        cancel_receive.set_onmessage(Some(cancel_callback.as_ref().unchecked_ref()));
        cancel_receive.start();
        cancel_send
            .post_message(&JsValue::UNDEFINED)
            .expect("queue same-worker cancellation task");
        canceled_rx
            .await
            .expect("the same-worker cancellation task must run");
        cancel_receive.set_onmessage(None);
        cancel_receive.close();
        cancel_send.close();
        drop(cancel_callback);
        let payload = execution
            .await
            .expect("execution returns a structured result");
        let error =
            Reflect::get(&payload, &JsValue::from_str("error")).expect("execution error field");
        let identifier = Reflect::get(&error, &JsValue::from_str("identifier"))
            .expect("structured cancellation identifier");
        assert_eq!(identifier.as_string().as_deref(), Some("RunMat:Cancelled"));
        let message = Reflect::get(&error, &JsValue::from_str("message"))
            .expect("structured cancellation message");
        assert!(
            message
                .as_string()
                .is_some_and(|text| text.contains("mnrfit: execution was cancelled")),
            "cancellation must reach the multinomial fit, not only an earlier pipeline stage"
        );
    }
}
