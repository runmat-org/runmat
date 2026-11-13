//! Linear accounting for admitted future work that may not have submitted any
//! physical device command yet. This is not a task scheduler or proof that all
//! possible deferred routes have been observed.

use super::{CommandProfilingReceipt, ReceiptState};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeferredWorkCounts {
    pub admitted: u64,
    pub pending: u64,
    pub completed: u64,
    pub failed: u64,
    pub missing: u64,
}

/// A terminal fact established at the owned future's actual retirement.
/// Cancellation is not success: if retirement has no explicit failure result,
/// dropping the unsettled lease records `Missing`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeferredWorkTerminal {
    Completed,
    Failed,
    Missing,
}

/// Non-Clone, Send-safe owner of one admitted deferred future. Transfer this
/// lease with that future and settle once at its actual terminal boundary.
pub struct CommandDeferredWork {
    receipt: Arc<Mutex<ReceiptState>>,
    settled: bool,
}

#[derive(Default)]
pub(super) struct DeferredWorkState {
    counts: DeferredWorkCounts,
}

impl DeferredWorkState {
    pub(super) fn counts(&self) -> DeferredWorkCounts {
        self.counts
    }
}

impl CommandProfilingReceipt {
    /// Admission must occur before the future is queued. `None` means a
    /// sealed, poisoned, or overflowed owner, not evidence of zero work; a
    /// caller must preserve that unknown-origin fact rather than fabricate a
    /// new command root. Only the non-Clone root may seal this receipt.
    pub fn begin_deferred_work(&self) -> Option<CommandDeferredWork> {
        let mut state = self.0.lock().ok()?;
        if state.sealed || state.invalid {
            return None;
        }
        let counts = &mut state.deferred.counts;
        let (Some(admitted), Some(pending)) = (
            counts.admitted.checked_add(1),
            counts.pending.checked_add(1),
        ) else {
            state.invalid = true;
            return None;
        };
        counts.admitted = admitted;
        counts.pending = pending;
        Some(CommandDeferredWork {
            receipt: Arc::clone(&self.0),
            settled: false,
        })
    }
}

impl CommandDeferredWork {
    pub fn complete(mut self, terminal: DeferredWorkTerminal) {
        self.settle(terminal);
    }

    fn settle(&mut self, terminal: DeferredWorkTerminal) {
        if self.settled {
            return;
        }
        self.settled = true;
        let Ok(mut state) = self.receipt.lock() else {
            return;
        };
        if state.sealed || state.invalid {
            return;
        }
        let counts = &mut state.deferred.counts;
        let (counter, next_pending) = match terminal {
            DeferredWorkTerminal::Completed => {
                (&mut counts.completed, counts.pending.checked_sub(1))
            }
            DeferredWorkTerminal::Failed => (&mut counts.failed, counts.pending.checked_sub(1)),
            DeferredWorkTerminal::Missing => (&mut counts.missing, counts.pending.checked_sub(1)),
        };
        match (next_pending, counter.checked_add(1)) {
            (Some(next_pending), Some(next_count)) => {
                counts.pending = next_pending;
                *counter = next_count;
            }
            _ => state.invalid = true,
        }
    }
}

impl Drop for CommandDeferredWork {
    fn drop(&mut self) {
        self.settle(DeferredWorkTerminal::Missing);
    }
}

#[cfg(test)]
mod tests;
