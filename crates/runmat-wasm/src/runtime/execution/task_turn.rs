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
mod browser_cancel_tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use futures::task::noop_waker;
    use std::future::Future;
    use std::rc::Rc;
    use std::task::Context;
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

    #[wasm_bindgen_test]
    fn abandoning_a_posted_turn_releases_its_callback_state() {
        let mut turn = Box::pin(BrowserTaskTurn::new().expect("browser MessageChannel"));
        let state = Rc::downgrade(&turn.as_ref().get_ref().state);
        let waker = noop_waker();
        assert!(turn
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        drop(turn);
        assert!(
            state.upgrade().is_none(),
            "drop must release the pending callback"
        );
    }
}
