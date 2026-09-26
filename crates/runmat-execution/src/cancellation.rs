//! Cooperative cancellation for one execution generation.
//!
//! A source never resets: create a new source for new work. Cloned tokens keep
//! the original generation alive and cannot observe a later generation. This
//! is a wakeup protocol, not an interrupt for noncooperative foreign code.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use crate::CancellationReason;

#[derive(Debug, Default)]
struct State {
    cancelled: AtomicBool,
    listeners: Mutex<Listeners>,
}

#[derive(Debug, Default)]
struct Listeners {
    reason: Option<CancellationReason>,
    slots: Vec<Option<Waker>>,
}

impl State {
    fn listeners(&self) -> MutexGuard<'_, Listeners> {
        self.listeners
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

/// Authority to cancel one generation. The first cancellation reason wins.
#[derive(Clone, Debug, Default)]
pub struct CancellationSource {
    state: Arc<State>,
}

impl CancellationSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn token(&self) -> CancellationToken {
        CancellationToken {
            states: TokenStates::Single(Arc::clone(&self.state)),
        }
    }

    /// Publish cancellation and wake every registered listener. Returns true
    /// only for the first request. Wakers may reenter this source safely.
    pub fn cancel(&self, reason: CancellationReason) -> bool {
        let wakers = {
            let mut listeners = self.state.listeners();
            if listeners.reason.is_some() {
                return false;
            }
            listeners.reason = Some(reason);
            self.state.cancelled.store(true, Ordering::Release);
            std::mem::take(&mut listeners.slots)
        };
        for waker in wakers.into_iter().flatten() {
            waker.wake();
        }
        true
    }
}

/// Read-only view of one immutable execution generation's cancellation state.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    states: TokenStates,
}

#[derive(Clone, Debug)]
enum TokenStates {
    Single(Arc<State>),
    Combined(Arc<[Arc<State>]>),
}

impl CancellationToken {
    fn states(&self) -> &[Arc<State>] {
        match &self.states {
            TokenStates::Single(state) => std::slice::from_ref(state),
            TokenStates::Combined(states) => states,
        }
    }

    /// Observe either parent without transferring cancellation authority.
    /// Nested compositions flatten and deduplicate by generation identity.
    /// If several parents are cancelled, the earliest parent in this stable
    /// order supplies the reason; wall-clock racing does not select a reason.
    pub fn or(&self, other: &Self) -> Self {
        if other.states().iter().all(|state| {
            self.states()
                .iter()
                .any(|existing| Arc::ptr_eq(existing, state))
        }) {
            return self.clone();
        }
        let mut states = self.states().to_vec();
        for state in other.states() {
            if !states.iter().any(|existing| Arc::ptr_eq(existing, state)) {
                states.push(Arc::clone(state));
            }
        }
        Self {
            states: TokenStates::Combined(states.into()),
        }
    }

    /// Allocation-free, lock-free admission/safepoint check.
    pub fn is_cancelled(&self) -> bool {
        self.states()
            .iter()
            .any(|state| state.cancelled.load(Ordering::Acquire))
    }

    pub fn reason(&self) -> Option<CancellationReason> {
        self.states().iter().find_map(|state| {
            if state.cancelled.load(Ordering::Acquire) {
                state.listeners().reason
            } else {
                None
            }
        })
    }

    /// Each future registers its own listener on first poll. Dropping a future
    /// unregisters it; dropping an unpolled future has no registration effects.
    pub fn cancelled(&self) -> Cancelled {
        Cancelled {
            token: self.clone(),
            slots: vec![None; self.states().len()],
        }
    }
}

/// A cancellation listener, safe to move between polls and across threads.
#[derive(Debug)]
#[must_use = "cancellation listeners must be polled to receive wakeups"]
pub struct Cancelled {
    token: CancellationToken,
    slots: Vec<Option<usize>>,
}

impl Future for Cancelled {
    type Output = CancellationReason;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Some(reason) = this.token.reason() {
            this.unregister();
            return Poll::Ready(reason);
        }
        // RawWaker clone/drop hooks belong to the executor and may reenter.
        // Never invoke them while the cancellation registry is locked.
        for (state, slot) in this.token.states().iter().zip(&mut this.slots) {
            let replacement = context.waker().clone();
            let retired = {
                let mut listeners = state.listeners();
                // Cancellation and registration share this lock. This recheck
                // closes the race between the fast path and listener installation.
                if listeners.reason.is_some() {
                    // The complete composition is rechecked below so reason
                    // precedence does not depend on registration order/races.
                    continue;
                }
                let index = match *slot {
                    Some(index) => index,
                    None => {
                        let index = listeners
                            .slots
                            .iter()
                            .position(Option::is_none)
                            .unwrap_or(listeners.slots.len());
                        if index == listeners.slots.len() {
                            listeners.slots.push(None);
                        }
                        *slot = Some(index);
                        index
                    }
                };
                listeners.slots[index].replace(replacement)
            };
            drop(retired);
        }
        if let Some(reason) = this.token.reason() {
            this.unregister();
            return Poll::Ready(reason);
        }
        Poll::Pending
    }
}

impl Cancelled {
    fn unregister(&mut self) {
        for (state, slot) in self.token.states().iter().zip(&mut self.slots) {
            let retired = slot.take().and_then(|index| {
                state
                    .listeners()
                    .slots
                    .get_mut(index)
                    .and_then(Option::take)
            });
            drop(retired);
        }
    }
}

impl Drop for Cancelled {
    fn drop(&mut self) {
        self.unregister();
    }
}

#[cfg(test)]
#[path = "cancellation/tests.rs"]
mod tests;
