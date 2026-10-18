//! Contextual ownership for the process driver's root future.

use super::control_owner::DriverStateOwner;
use runmat_runtime::context::RuntimeContext;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// The root is created, polled and destroyed under the invocation that
/// admitted it. Retaining the existing driver-state owner avoids a second
/// `RuntimeContext` snapshot while keeping failed fence retirement contextual.
pub(super) struct InvocationRoot<F> {
    state: DriverStateOwner,
    future: Option<Pin<Box<F>>>,
}

impl<F> InvocationRoot<F> {
    pub(super) fn new(state: DriverStateOwner, future: Pin<Box<F>>) -> Self {
        Self {
            state,
            future: Some(future),
        }
    }

    fn selected_context(&self) -> Option<RuntimeContext> {
        self.state.borrow().authority.clone()
    }

    fn drop_future(&mut self) {
        let Some(future) = self.future.take() else {
            return;
        };
        if let Some(context) = self.selected_context() {
            let _guard = context.enter();
            drop(future);
            drain_placement_feedback(&context);
        } else {
            drop(future);
        }
    }
}

impl<F: Future> InvocationRoot<F> {
    pub(super) fn poll(&mut self, context: &mut Context<'_>) -> Poll<F::Output> {
        // Build the guard while the state is borrowed, then release that
        // borrow before user code can reenter the driver. `enter` performs
        // the one context clone required by the TLS stack; avoid cloning the
        // complete context once more merely to obtain the guard.
        let _guard = {
            let state = self.state.borrow();
            state.authority.as_ref().map(RuntimeContext::enter)
        };
        self.future
            .as_mut()
            .expect("VM invocation polled after completion")
            .as_mut()
            .poll(context)
    }
}

impl<F> Drop for InvocationRoot<F> {
    fn drop(&mut self) {
        self.drop_future();
    }
}

pub(super) fn drain_placement_feedback(context: &RuntimeContext) {
    if let Some(placement) = context.service_ports().placement() {
        placement.drain_feedback();
    }
}
