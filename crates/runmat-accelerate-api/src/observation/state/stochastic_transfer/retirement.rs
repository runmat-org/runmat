//! Exact completion acknowledgement for one selected source transfer.

use crate::observation::ProviderObservationFailure;
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

struct State {
    status: AtomicU8,
    waker: AtomicWaker,
}

impl State {
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

/// Proof that the exact queue submission which fenced one selected source
/// transfer has released its completion owners.
pub struct StochasticEvolutionSourceTransferRetirement {
    state: Arc<State>,
}

impl Future for StochasticEvolutionSourceTransferRetirement {
    type Output = Result<(), ProviderObservationFailure>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let status = self.state.status.load(Ordering::Acquire);
        if status != PENDING {
            return Poll::Ready(terminal(status));
        }
        self.state.waker.register(context.waker());
        let status = self.state.status.load(Ordering::Acquire);
        if status == PENDING {
            Poll::Pending
        } else {
            Poll::Ready(terminal(status))
        }
    }
}

fn terminal(status: u8) -> Result<(), ProviderObservationFailure> {
    match status {
        RETIRED => Ok(()),
        FAILED => Err(ProviderObservationFailure::ProviderFailed),
        _ => Err(ProviderObservationFailure::ExecutionUnproven),
    }
}

/// Provider completion-side half of an armed source-transfer retirement.
/// Dropping it without successful exact completion fails the caller's proof.
#[doc(hidden)]
pub struct StochasticEvolutionSourceTransferRetirementNotifier {
    state: Arc<State>,
}

impl StochasticEvolutionSourceTransferRetirementNotifier {
    #[doc(hidden)]
    pub fn finish(self, succeeded: bool) {
        self.state.finish(succeeded);
    }
}

impl Drop for StochasticEvolutionSourceTransferRetirementNotifier {
    fn drop(&mut self) {
        self.state.finish(false);
    }
}

pub(super) fn pair() -> (
    StochasticEvolutionSourceTransferRetirement,
    StochasticEvolutionSourceTransferRetirementNotifier,
) {
    let state = Arc::new(State {
        status: AtomicU8::new(PENDING),
        waker: AtomicWaker::new(),
    });
    (
        StochasticEvolutionSourceTransferRetirement {
            state: state.clone(),
        },
        StochasticEvolutionSourceTransferRetirementNotifier { state },
    )
}

/// Allocation bound for the shared acknowledgement state. Both handles are
/// inline in owners already accounted by their respective phases.
pub fn stochastic_evolution_source_transfer_retirement_host_upper_bytes() -> Option<u64> {
    let arc_header = std::mem::size_of::<usize>().checked_mul(2)?;
    std::mem::size_of::<State>()
        .checked_add(arc_header)?
        .checked_add(std::mem::align_of::<State>())?
        .try_into()
        .ok()
}
