//! Queue-origin profiling and the task-submission handoff for one cleanup.

use runmat_execution::profiling::{
    CommandDeferredWork, CommandProfilingReceipt, DeferredWorkTerminal,
};

use super::{CleanupEntry, CleanupReceipt};

/// `Unowned` is an actual idle/no-command queue admission. `Unknown` means a
/// command origin existed but its receipt denied admission (sealed or invalid);
/// neither case can be turned into a successful zero-work measurement.
pub(crate) enum CleanupDeferredOrigin {
    Unowned,
    Unknown,
    Admitted(CommandDeferredWork),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CleanupFutureHandoff {
    NotStarted,
    Submitting,
    Accepted,
    RetiredBeforeAck,
    RejectedPendingRetirement,
    Retired,
}

impl CleanupEntry {
    pub(super) fn completion_retirement_pending(&self) -> bool {
        self.future_handoff.get() != CleanupFutureHandoff::NotStarted
            && !self.retirement_settled.get()
    }

    pub(crate) fn finish_future_retirement(&self) {
        self.retirement_settled.set(true);
        self.wake_retired();
    }

    fn wake_retired(&self) {
        let wake = self.retirement_waker.borrow_mut().take();
        if let Some(wake) = wake {
            wake.wake();
        }
    }

    pub(crate) async fn wait_future_retired(&self) {
        futures::future::poll_fn(|context| {
            if !self.completion_retirement_pending() {
                return std::task::Poll::Ready(());
            }
            let mut wake = self.retirement_waker.borrow_mut();
            if !wake
                .as_ref()
                .is_some_and(|wake| wake.will_wake(context.waker()))
            {
                *wake = Some(context.waker().clone());
            }
            std::task::Poll::Pending
        })
        .await
    }

    pub(super) fn queue_command_origin(&self, origin: Option<&CommandProfilingReceipt>) {
        *self.queued_command_origin.borrow_mut() = origin.cloned();
        *self.deferred_origin.borrow_mut() = match origin {
            None => CleanupDeferredOrigin::Unowned,
            Some(receipt) => receipt.begin_deferred_work().map_or(
                CleanupDeferredOrigin::Unknown,
                CleanupDeferredOrigin::Admitted,
            ),
        };
    }

    pub(super) fn settle_deferred(&self, terminal: DeferredWorkTerminal) {
        let origin = std::mem::replace(
            &mut *self.deferred_origin.borrow_mut(),
            CleanupDeferredOrigin::Unowned,
        );
        if let CleanupDeferredOrigin::Admitted(work) = origin {
            work.complete(terminal);
        }
    }

    pub(crate) fn settle_deferred_from_receipt(&self) {
        let terminal = match self.receipt.borrow().as_ref() {
            Some(CleanupReceipt::Completed(Ok(()))) => DeferredWorkTerminal::Completed,
            Some(CleanupReceipt::Completed(Err(_))) | Some(CleanupReceipt::DispatchFailed(_)) => {
                DeferredWorkTerminal::Failed
            }
            Some(CleanupReceipt::Cancelled | CleanupReceipt::Abandoned) | None => {
                DeferredWorkTerminal::Missing
            }
        };
        self.settle_deferred(terminal);
    }

    pub(crate) fn prepare_future_handoff(&self) {
        debug_assert_eq!(self.future_handoff.get(), CleanupFutureHandoff::NotStarted);
        self.future_handoff.set(CleanupFutureHandoff::Submitting);
    }

    pub(crate) fn acknowledge_future_handoff(&self, accepted: bool) -> bool {
        let next = match (self.future_handoff.get(), accepted) {
            (CleanupFutureHandoff::Submitting, true) => CleanupFutureHandoff::Accepted,
            (CleanupFutureHandoff::Submitting, false) => {
                CleanupFutureHandoff::RejectedPendingRetirement
            }
            (CleanupFutureHandoff::RetiredBeforeAck, _) => CleanupFutureHandoff::Retired,
            (current, _) => current,
        };
        self.future_handoff.set(next);
        next == CleanupFutureHandoff::Retired
    }

    /// The wrapper has explicitly dropped its inner callback future first.
    /// A pre-return retirement waits for the submit result to distinguish a
    /// real dispatch failure from an accepted task that never began.
    pub(crate) fn note_future_retired(&self) -> bool {
        match self.future_handoff.get() {
            CleanupFutureHandoff::Submitting => {
                self.future_handoff
                    .set(CleanupFutureHandoff::RetiredBeforeAck);
                false
            }
            CleanupFutureHandoff::Accepted | CleanupFutureHandoff::RejectedPendingRetirement => {
                self.future_handoff.set(CleanupFutureHandoff::Retired);
                true
            }
            _ => false,
        }
    }

    pub(super) fn future_may_own_resources(&self) -> bool {
        matches!(
            self.future_handoff.get(),
            CleanupFutureHandoff::Submitting
                | CleanupFutureHandoff::Accepted
                | CleanupFutureHandoff::RejectedPendingRetirement
        )
    }
}
