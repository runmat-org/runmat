//! Own the exact frame completion across both polling and abandonment.
//!
//! Workspace mutations stage through the selected workspace service in source
//! order; the embedding host owns that service's canonical publication. This
//! owner never replays register values: it hands the moved completion to the
//! caller, or retires it on abandonment.
//! The entire owner is polled and destroyed inside the existing RuntimeContext
//! scope, so frame activation, publication, and draining select the same issuer's
//! thread-local tables even when the public caller has no ambient context.

use super::{InterpreterOutcome, VmResult, WorkspaceInterpreterOutcome};
use crate::runtime::workspace::{
    take_workspace_completion, WorkspaceExecutionCompletion, WorkspaceFrameFuture, WorkspaceFrameId,
};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub(super) struct PendingWorkspaceExecution<F> {
    // Field order is the cancellation protocol: destroying the execution first
    // closes the workspace guard and publishes its already-admitted node.
    // Only then may the ticket drain that exact node. No extra allocation or
    // retained copy of a workspace value is needed for this handoff.
    execution: WorkspaceFrameFuture<F>,
    completion: WorkspaceCompletionTicket,
}

impl<F> PendingWorkspaceExecution<F> {
    pub(super) fn new(execution: WorkspaceFrameFuture<F>, frame: WorkspaceFrameId) -> Self {
        Self {
            execution,
            completion: WorkspaceCompletionTicket(Some(frame)),
        }
    }
}

impl<F: Future<Output = VmResult<InterpreterOutcome>>> Future for PendingWorkspaceExecution<F> {
    type Output = WorkspaceInterpreterOutcome;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match Pin::new(&mut this.execution).poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(outcome) => Poll::Ready(WorkspaceInterpreterOutcome {
                outcome,
                // Move the completion before the wrapper is destroyed. The
                // disarmed ticket cannot consume the caller's returned owner.
                workspace: this.completion.take(),
            }),
        }
    }
}

/// Move-only authority to consume one completion, never the latest frame for
/// a runtime. Its lifetime is private to the pending execution above so it
/// cannot retire before the child future has closed its frame.
struct WorkspaceCompletionTicket(Option<WorkspaceFrameId>);

impl WorkspaceCompletionTicket {
    fn take(&mut self) -> Option<WorkspaceExecutionCompletion> {
        self.0.take().and_then(take_workspace_completion)
    }
}

impl Drop for WorkspaceCompletionTicket {
    fn drop(&mut self) {
        drop(self.take());
    }
}
