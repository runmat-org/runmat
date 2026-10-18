//! Awaitable proof that one exact submitted allocation bundle has retired.
//!
//! The queue-completion owner holds the notifier. Dropping the caller's
//! waiter never cancels queue retirement or releases submitted allocations.

use futures::task::AtomicWaker;
use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

const PENDING: u8 = 0;
const RETIRED: u8 = 1;
const FAILED: u8 = 2;

struct RetirementState {
    status: AtomicU8,
    waker: AtomicWaker,
}

impl RetirementState {
    fn finish(&self, succeeded: bool) {
        let terminal = if succeeded { RETIRED } else { FAILED };
        if self
            .status
            .compare_exchange(PENDING, terminal, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.waker.wake();
        }
    }
}

/// Caller-owned proof for the exact submission that received the paired
/// prepared retention. It resolves only after the queue-completion owner has
/// released the retained allocation bundle.
pub(crate) struct SubmittedAllocationRetirement {
    state: Arc<RetirementState>,
}

impl Future for SubmittedAllocationRetirement {
    type Output = anyhow::Result<()>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let status = self.state.status.load(Ordering::Acquire);
        if status != PENDING {
            return Poll::Ready(terminal_result(status));
        }
        self.state.waker.register(context.waker());
        let status = self.state.status.load(Ordering::Acquire);
        if status == PENDING {
            Poll::Pending
        } else {
            Poll::Ready(terminal_result(status))
        }
    }
}

fn terminal_result(status: u8) -> anyhow::Result<()> {
    match status {
        RETIRED => Ok(()),
        FAILED => Err(anyhow::anyhow!(
            "exact submitted allocation retirement failed"
        )),
        _ => Err(anyhow::anyhow!(
            "submitted allocation retirement entered an invalid state"
        )),
    }
}

pub(crate) struct SubmittedAllocationRetirementNotifier {
    state: Arc<RetirementState>,
}

impl SubmittedAllocationRetirementNotifier {
    pub(super) fn finish(self, succeeded: bool) {
        self.state.finish(succeeded);
    }
}

impl Drop for SubmittedAllocationRetirementNotifier {
    fn drop(&mut self) {
        self.state.finish(false);
    }
}

pub(super) fn retirement_pair() -> (
    SubmittedAllocationRetirement,
    SubmittedAllocationRetirementNotifier,
) {
    let state = Arc::new(RetirementState {
        status: AtomicU8::new(PENDING),
        waker: AtomicWaker::new(),
    });
    (
        SubmittedAllocationRetirement {
            state: state.clone(),
        },
        SubmittedAllocationRetirementNotifier { state },
    )
}

/// Conservative allocation bound for the `Arc` payload and its strong/weak
/// counters. The waiter and notifier handles are inline in their existing
/// owners and allocate no additional control block.
pub(super) fn retirement_host_upper_bytes() -> Option<u64> {
    let arc_header = std::mem::size_of::<usize>().checked_mul(2)?;
    std::mem::size_of::<RetirementState>()
        .checked_add(arc_header)?
        .checked_add(std::mem::align_of::<RetirementState>())?
        .try_into()
        .ok()
}
