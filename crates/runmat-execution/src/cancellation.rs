//! Cooperative cancellation for one execution generation.
//!
//! A source never resets: create a new source for new work. Cloned tokens keep
//! the original generation alive and cannot observe a later generation. This
//! is a wakeup protocol, not an interrupt for noncooperative foreign code.
//!
//! A token may additionally observe live host interrupt flags (shared
//! `Arc<AtomicBool>` controls owned by sessions, workers and domain adapters)
//! and a wall-clock deadline. Flags and deadlines are evaluated at each
//! checkpoint (`is_cancelled`/`stop_reason`); only generations wake listeners.

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
            interrupts: Arc::from([]),
            deadline_unix_ms: None,
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

/// Read-only view of one immutable execution generation's cancellation state,
/// plus any live host interrupt flags and the tightest inherited deadline.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    states: TokenStates,
    /// Host interrupt flags. The first entry of a token built by
    /// [`CancellationToken::from_shared`] is its primary flag.
    interrupts: Arc<[Arc<AtomicBool>]>,
    deadline_unix_ms: Option<u64>,
}

#[derive(Clone, Debug)]
enum TokenStates {
    Single(Arc<State>),
    Combined(Arc<[Arc<State>]>),
}

impl Default for CancellationToken {
    /// A standalone token owning a fresh primary interrupt flag.
    fn default() -> Self {
        Self::from_shared(Arc::new(AtomicBool::new(false)))
    }
}

impl CancellationToken {
    /// Observe an existing live host interrupt flag (session, worker or
    /// adapter control). The flag becomes this token's primary flag.
    pub fn from_shared(primary: Arc<AtomicBool>) -> Self {
        Self {
            states: TokenStates::Combined(Arc::from([])),
            interrupts: Arc::from([primary]),
            deadline_unix_ms: None,
        }
    }

    /// Additionally observe a live host interrupt flag; deduplicated by identity.
    pub fn with_source(mut self, source: Arc<AtomicBool>) -> Self {
        if !self
            .interrupts
            .iter()
            .any(|flag| Arc::ptr_eq(flag, &source))
        {
            let mut interrupts = self.interrupts.to_vec();
            interrupts.push(source);
            self.interrupts = interrupts.into();
        }
        self
    }

    /// Tighten the deadline. A child can never extend or clear a parent deadline.
    pub fn with_deadline(mut self, deadline_unix_ms: Option<u64>) -> Self {
        self.deadline_unix_ms = match (self.deadline_unix_ms, deadline_unix_ms) {
            (Some(parent), Some(child)) => Some(parent.min(child)),
            (parent, child) => parent.or(child),
        };
        self
    }

    pub fn deadline_unix_ms(&self) -> Option<u64> {
        self.deadline_unix_ms
    }

    /// Raise this token's primary host interrupt flag. Returns false for
    /// generation tokens, which are read-only: cancel those via their source.
    pub fn cancel(&self) -> bool {
        match self.interrupts.first() {
            Some(primary) if matches!(&self.states, TokenStates::Combined(s) if s.is_empty()) => {
                primary.store(true, Ordering::Release);
                true
            }
            _ => false,
        }
    }

    /// Exact invocation-generation identity, without observing mutable
    /// cancelled state. A rebound token cannot inherit an earlier permit.
    pub fn same_generation(&self, other: &Self) -> bool {
        self.states().len() == other.states().len()
            && self
                .states()
                .iter()
                .zip(other.states())
                .all(|(left, right)| Arc::ptr_eq(left, right))
    }

    fn states(&self) -> &[Arc<State>] {
        match &self.states {
            TokenStates::Single(state) => std::slice::from_ref(state),
            TokenStates::Combined(states) => states,
        }
    }

    /// Observe either parent without transferring cancellation authority.
    /// Nested compositions flatten and deduplicate by generation identity;
    /// interrupt flags are merged and the tighter deadline wins.
    /// If several parents are cancelled, the earliest parent in this stable
    /// order supplies the reason; wall-clock racing does not select a reason.
    pub fn or(&self, other: &Self) -> Self {
        let mut combined = self.clone();
        if !other.states().iter().all(|state| {
            self.states()
                .iter()
                .any(|existing| Arc::ptr_eq(existing, state))
        }) {
            let mut states = self.states().to_vec();
            for state in other.states() {
                if !states.iter().any(|existing| Arc::ptr_eq(existing, state)) {
                    states.push(Arc::clone(state));
                }
            }
            combined.states = TokenStates::Combined(states.into());
        }
        for flag in other.interrupts.iter() {
            combined = combined.with_source(Arc::clone(flag));
        }
        combined.with_deadline(other.deadline_unix_ms)
    }

    fn interrupted(&self) -> bool {
        self.interrupts
            .iter()
            .any(|flag| flag.load(Ordering::Acquire))
    }

    /// Allocation-free, lock-free admission/safepoint check, including any
    /// live interrupt flag and an elapsed deadline.
    pub fn is_cancelled(&self) -> bool {
        self.states()
            .iter()
            .any(|state| state.cancelled.load(Ordering::Acquire))
            || self.interrupted()
            || self
                .deadline_unix_ms
                .is_some_and(|deadline| runmat_time::unix_timestamp_ms() >= u128::from(deadline))
    }

    /// Reason from a cancelled generation or a raised interrupt flag (`User`).
    pub fn reason(&self) -> Option<CancellationReason> {
        self.states()
            .iter()
            .find_map(|state| {
                if state.cancelled.load(Ordering::Acquire) {
                    state.listeners().reason
                } else {
                    None
                }
            })
            .or_else(|| self.interrupted().then_some(CancellationReason::User))
    }

    /// Checkpoint reason at an explicit wall-clock instant.
    pub fn stop_reason_at(&self, unix_ms: u128) -> Option<CancellationReason> {
        self.reason().or_else(|| {
            self.deadline_unix_ms
                .filter(|deadline| unix_ms >= u128::from(*deadline))
                .map(|_| CancellationReason::Deadline)
        })
    }

    /// Checkpoint reason now: generation/interrupt reason, else elapsed deadline.
    pub fn stop_reason(&self) -> Option<CancellationReason> {
        self.stop_reason_at(runmat_time::unix_timestamp_ms())
    }

    /// Each future registers its own listener on first poll. Dropping a future
    /// unregisters it; dropping an unpolled future has no registration effects.
    /// Only generation cancellation wakes the future; interrupt flags and
    /// deadlines are observed when it is polled.
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

#[cfg(test)]
#[path = "cancellation/interrupt_tests.rs"]
mod interrupt_tests;
