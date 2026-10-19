//! Poll-local activation of an invocation's fusion plan and instruction group.

use super::{with_active_context, ActiveContext, FusionPlanRef};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Retains a frame's plan across suspension without leaving it installed on
/// the executor thread. The caller's plan is restored after polling or drop.
pub struct FusionPlanFuture<F> {
    future: Option<Pin<Box<F>>>,
    context: Option<ActiveContext>,
}

/// A frame with no fusion plan masks its caller's plan while it executes.
pub fn scope_fusion_plan<F: Future>(plan: Option<FusionPlanRef>, future: F) -> FusionPlanFuture<F> {
    FusionPlanFuture {
        future: Some(Box::pin(future)),
        context: plan.map(|plan| ActiveContext {
            plan,
            active_group: None,
        }),
    }
}

struct Activation<'a> {
    frame: &'a mut Option<ActiveContext>,
    caller: Option<ActiveContext>,
}

impl<'a> Activation<'a> {
    fn enter(frame: &'a mut Option<ActiveContext>) -> Self {
        let caller = with_active_context(|active| std::mem::replace(active, frame.take()));
        Self { frame, caller }
    }
}

impl Drop for Activation<'_> {
    fn drop(&mut self) {
        *self.frame = with_active_context(|active| std::mem::replace(active, self.caller.take()));
    }
}

impl<F: Future> Future for FusionPlanFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let _activation = Activation::enter(&mut this.context);
        let result = this
            .future
            .as_mut()
            .expect("fusion plan future polled after completion")
            .as_mut()
            .poll(cx);
        if result.is_ready() {
            // Frame destructors observe their own plan, not the caller's.
            drop(this.future.take());
        }
        result
    }
}

impl<F> Drop for FusionPlanFuture<F> {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            let _activation = Activation::enter(&mut self.context);
            drop(future);
        }
    }
}

// Moving this owner never moves the future pinned in its heap allocation.
impl<F> Unpin for FusionPlanFuture<F> {}

#[cfg(test)]
mod tests;
