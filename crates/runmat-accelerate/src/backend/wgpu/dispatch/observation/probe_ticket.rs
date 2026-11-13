//! Passive status of the existing optional provider fence. No polling or
//! scheduler lives here; only PendingSubmissions callbacks may complete it.
use std::sync::{
    atomic::{AtomicU64, AtomicU8, Ordering},
    Arc,
};

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProbeFenceState {
    Armed = 0,
    Absent = 1,
    Pending = 2,
    Completed = 3,
    Failed = 4,
}

#[derive(Debug)]
struct Inner {
    state: AtomicU8,
    fence_bytes: AtomicU64,
}

#[derive(Clone, Debug)]
pub(crate) struct ProbeFenceTicket(Arc<Inner>);

impl ProbeFenceTicket {
    pub(crate) fn armed() -> Self {
        Self(Arc::new(Inner {
            state: AtomicU8::new(ProbeFenceState::Armed as u8),
            fence_bytes: AtomicU64::new(0),
        }))
    }

    pub(crate) fn state(&self) -> ProbeFenceState {
        match self.0.state.load(Ordering::Acquire) {
            0 => ProbeFenceState::Armed,
            1 => ProbeFenceState::Absent,
            2 => ProbeFenceState::Pending,
            3 => ProbeFenceState::Completed,
            _ => ProbeFenceState::Failed,
        }
    }

    #[cfg(test)]
    pub(crate) fn fence_requested_bytes(&self) -> Option<u64> {
        match self.0.fence_bytes.load(Ordering::Acquire) {
            0 => None,
            bytes => Some(bytes),
        }
    }

    pub(super) fn submitted_without_fence(&self, observation_failed: bool) {
        let next = if observation_failed {
            ProbeFenceState::Failed
        } else {
            ProbeFenceState::Absent
        };
        let _ = self.0.state.compare_exchange(
            ProbeFenceState::Armed as u8,
            next as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub(super) fn submitted_with_fence(&self, bytes: u64) {
        self.0.fence_bytes.store(bytes, Ordering::Release);
        let _ = self.0.state.compare_exchange(
            ProbeFenceState::Armed as u8,
            ProbeFenceState::Pending as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub(super) fn complete(&self) {
        let _ = self.0.state.compare_exchange(
            ProbeFenceState::Pending as u8,
            ProbeFenceState::Completed as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub(super) fn fail(&self) {
        let state = self.state();
        if matches!(state, ProbeFenceState::Armed | ProbeFenceState::Pending) {
            let _ = self.0.state.compare_exchange(
                state as u8,
                ProbeFenceState::Failed as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

#[cfg(test)]
#[path = "probe_ticket/tests.rs"]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "probe_ticket/browser_tests.rs"]
mod browser_tests;
