//! Atomic replacement fence for one session-owned graphics callback claim.
use super::GraphicsCallbackDisposition;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const RETIRED: usize = 1 << (usize::BITS - 1);
const INDETERMINATE: usize = 1 << (usize::BITS - 2);
const PUBLISHED: usize = 1 << (usize::BITS - 3);
const ACTIVE_MASK: usize = PUBLISHED - 1;

#[derive(Default)]
pub(super) struct CallbackClaim(AtomicUsize);

impl CallbackClaim {
    pub(super) fn is_live(&self) -> bool {
        self.0.load(Ordering::Acquire) & RETIRED == 0
    }

    pub(super) fn begin(self: &Arc<Self>) -> Option<ActiveCallbackClaim> {
        self.0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                if state & (RETIRED | INDETERMINATE) == 0 && state & ACTIVE_MASK < ACTIVE_MASK {
                    Some(state + 1)
                } else {
                    None
                }
            })
            .ok()?;
        Some(ActiveCallbackClaim {
            claim: Arc::clone(self),
            terminal: false,
        })
    }

    pub(super) fn retire(&self) -> GraphicsCallbackDisposition {
        let prior = self.0.fetch_or(RETIRED, Ordering::AcqRel);
        if prior & (ACTIVE_MASK | INDETERMINATE) != 0 {
            GraphicsCallbackDisposition::Indeterminate
        } else if prior & PUBLISHED != 0 {
            GraphicsCallbackDisposition::Published
        } else {
            GraphicsCallbackDisposition::NotPublished
        }
    }
}

pub(super) struct ActiveCallbackClaim {
    claim: Arc<CallbackClaim>,
    terminal: bool,
}

impl ActiveCallbackClaim {
    /// A host ticket retired after claim acquisition but before dispatch.
    /// No callback ran, so neither publication nor an indeterminate effect
    /// may be recorded by this guard's Drop.
    pub(super) fn abort_before_invocation(&mut self) {
        self.terminal = true;
    }

    pub(super) fn terminal(&mut self) {
        self.claim.0.fetch_or(PUBLISHED, Ordering::AcqRel);
        self.terminal = true;
    }
}

impl Drop for ActiveCallbackClaim {
    fn drop(&mut self) {
        if !self.terminal {
            self.claim.0.fetch_or(INDETERMINATE, Ordering::AcqRel);
        }
        let prior = self.claim.0.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prior & ACTIVE_MASK > 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_and_begin_have_one_atomic_winner() {
        let claim = Arc::new(CallbackClaim::default());
        let active = claim.begin().expect("began");
        assert_eq!(claim.retire(), GraphicsCallbackDisposition::Indeterminate);
        assert!(claim.begin().is_none());
        drop(active);
        assert_eq!(claim.retire(), GraphicsCallbackDisposition::Indeterminate);

        let claim = Arc::new(CallbackClaim::default());
        assert_eq!(claim.retire(), GraphicsCallbackDisposition::NotPublished);
        assert!(claim.begin().is_none());

        let claim = Arc::new(CallbackClaim::default());
        let mut active = claim.begin().expect("began");
        active.terminal();
        drop(active);
        assert_eq!(claim.retire(), GraphicsCallbackDisposition::Published);

        let exhausted = Arc::new(CallbackClaim(AtomicUsize::new(ACTIVE_MASK)));
        assert!(
            exhausted.begin().is_none(),
            "capacity must not set a state bit"
        );
    }

    #[test]
    fn host_ticket_retirement_before_invocation_does_not_publish_claim() {
        let claim = Arc::new(CallbackClaim::default());
        let mut active = claim.begin().expect("claim acquired");
        active.abort_before_invocation();
        drop(active);
        assert_eq!(claim.retire(), GraphicsCallbackDisposition::NotPublished);
    }
}
