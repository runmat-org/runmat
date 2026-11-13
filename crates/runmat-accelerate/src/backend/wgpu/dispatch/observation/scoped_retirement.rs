//! One pre-emission signal for a diagnostic submission whose scope outcome is
//! not yet authenticated. GPU retirement never waits for the scope future.
use runmat_execution::resource::ExactShared;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

struct State {
    queue_retired: AtomicBool,
    device_lost: AtomicBool,
    callback_registered: AtomicBool,
    outcome: AtomicU8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum ScopeOutcome {
    Pending = 0,
    Confirmed = 1,
    Rejected = 2,
    Abandoned = 3,
}
impl ScopeOutcome {
    fn load(value: &AtomicU8) -> Self {
        match value.load(Ordering::Acquire) {
            value if value == Self::Confirmed as u8 => Self::Confirmed,
            value if value == Self::Rejected as u8 => Self::Rejected,
            value if value == Self::Abandoned as u8 => Self::Abandoned,
            _ => Self::Pending,
        }
    }
}

/// Constructed only from the actual queue callback or actual device-loss hook.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RetirementKind {
    QueueCallback,
    DeviceLost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScopedRetirementProof {
    kind: RetirementKind,
}

impl ScopedRetirementProof {
    pub(crate) fn queue_completed(self) -> bool {
        self.kind == RetirementKind::QueueCallback
    }
}

#[derive(Clone)]
pub(crate) struct ScopedRetirement {
    state: ExactShared<State>,
}

impl ScopedRetirement {
    pub(crate) fn prepare() -> anyhow::Result<Self> {
        Ok(Self {
            state: ExactShared::try_new(State {
                queue_retired: AtomicBool::new(false),
                device_lost: AtomicBool::new(false),
                callback_registered: AtomicBool::new(false),
                outcome: AtomicU8::new(ScopeOutcome::Pending as u8),
            })
            .map_err(|_| anyhow::anyhow!("scoped FFT retirement signal allocation failed"))?,
        })
    }

    pub(crate) fn host_upper_bytes() -> Option<u64> {
        ExactShared::<State>::requested_bytes().try_into().ok()
    }

    /// The callback records a fact only; it never unmaps or drops GPU owners.
    pub(crate) fn queue_retired(&self) {
        self.state
            .get()
            .queue_retired
            .store(true, Ordering::Release);
    }

    pub(crate) fn is_retired(&self) -> bool {
        self.proof().is_some()
    }

    pub(crate) fn proof(&self) -> Option<ScopedRetirementProof> {
        if self.state.get().queue_retired.load(Ordering::Acquire) {
            Some(ScopedRetirementProof {
                kind: RetirementKind::QueueCallback,
            })
        } else if self.state.get().device_lost.load(Ordering::Acquire) {
            Some(ScopedRetirementProof {
                kind: RetirementKind::DeviceLost,
            })
        } else {
            None
        }
    }

    pub(crate) fn actual_device_lost(&self) {
        self.state.get().device_lost.store(true, Ordering::Release);
    }

    pub(crate) fn claim_callback_registration(&self) -> bool {
        !self
            .state
            .get()
            .callback_registered
            .swap(true, Ordering::AcqRel)
    }

    pub(crate) fn confirm(&self) {
        self.state
            .get()
            .outcome
            .store(ScopeOutcome::Confirmed as u8, Ordering::Release);
    }

    pub(crate) fn is_confirmed(&self) -> bool {
        ScopeOutcome::load(&self.state.get().outcome) == ScopeOutcome::Confirmed
    }

    pub(crate) fn reject(&self) {
        self.state
            .get()
            .outcome
            .store(ScopeOutcome::Rejected as u8, Ordering::Release);
    }
}

/// Dropping a scope future records abandonment but does not revoke an already
/// queued physical retirement job or keep native shutdown waiting on software.
pub(crate) struct ScopeConfirmation(pub(crate) ScopedRetirement);
impl Drop for ScopeConfirmation {
    fn drop(&mut self) {
        let _ = self.0.state.get().outcome.compare_exchange(
            ScopeOutcome::Pending as u8,
            ScopeOutcome::Abandoned as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_retirement_is_independent_of_confirmation_or_future_drop() {
        let signal = ScopedRetirement::prepare().unwrap();
        let guard = ScopeConfirmation(signal.clone());
        signal.queue_retired();
        assert!(signal.is_retired());
        assert!(!signal.is_confirmed());
        drop(guard);
        assert!(signal.is_retired());
        assert!(!signal.is_confirmed());
    }

    #[test]
    fn positive_evidence_requires_confirmation_not_just_queue_retirement() {
        let signal = ScopedRetirement::prepare().unwrap();
        signal.confirm();
        assert!(!signal.is_retired());
        signal.queue_retired();
        assert!(signal.is_confirmed());
        assert!(signal.is_retired());
    }
}
