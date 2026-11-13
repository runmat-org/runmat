use super::{WorkspaceFrameId, WorkspaceOwner};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

runmat_thread_local! {
    static ACTIVE_WORKSPACE_FRAMES: RefCell<HashMap<WorkspaceOwner, Vec<WorkspaceFrameId>>> =
        RefCell::new(HashMap::new());
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WorkspaceFrameActivation {
    owner: WorkspaceOwner,
    frame: WorkspaceFrameId,
}

impl WorkspaceFrameActivation {
    pub(super) fn new(owner: WorkspaceOwner, frame: WorkspaceFrameId) -> Self {
        Self { owner, frame }
    }

    pub(crate) fn scope<F: Future>(self, future: F) -> WorkspaceFrameFuture<F> {
        WorkspaceFrameFuture {
            activation: Some(self),
            future: Box::pin(future),
        }
    }

    pub(crate) fn frame(self) -> WorkspaceFrameId {
        self.frame
    }
}

pub(crate) struct WorkspaceFrameFuture<F> {
    activation: Option<WorkspaceFrameActivation>,
    future: Pin<Box<F>>,
}

impl<F: Future> WorkspaceFrameFuture<F> {
    /// Service-owned dynamic targets have no VM frame to activate. Their
    /// RuntimeContext and captured issuer remain the exact authority.
    pub(crate) fn without_frame(future: F) -> Self {
        Self {
            activation: None,
            future: Box::pin(future),
        }
    }
}

impl<F: Future> Future for WorkspaceFrameFuture<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let _guard = self.activation.map(PollActivationGuard::enter);
        self.future.as_mut().poll(context)
    }
}

pub(super) fn active_frame(owner: WorkspaceOwner) -> Option<WorkspaceFrameId> {
    ACTIVE_WORKSPACE_FRAMES.with(|frames| {
        frames
            .borrow()
            .get(&owner)
            .and_then(|active| active.last().copied())
    })
}

pub(super) fn reset_for_tests() {
    ACTIVE_WORKSPACE_FRAMES.with(|frames| frames.borrow_mut().clear());
}

struct PollActivationGuard {
    activation: WorkspaceFrameActivation,
}

impl PollActivationGuard {
    fn enter(activation: WorkspaceFrameActivation) -> Self {
        ACTIVE_WORKSPACE_FRAMES.with(|frames| {
            frames
                .borrow_mut()
                .entry(activation.owner)
                .or_default()
                .push(activation.frame);
        });
        Self { activation }
    }
}

impl Drop for PollActivationGuard {
    fn drop(&mut self) {
        ACTIVE_WORKSPACE_FRAMES.with(|frames| {
            let mut frames = frames.borrow_mut();
            if let Some(active) = frames.get_mut(&self.activation.owner) {
                let removed = active.pop();
                debug_assert_eq!(removed, Some(self.activation.frame));
                if active.is_empty() {
                    frames.remove(&self.activation.owner);
                }
            }
        });
    }
}
