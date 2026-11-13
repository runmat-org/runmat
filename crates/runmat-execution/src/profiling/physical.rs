//! One linear contribution per submitted physical device encoder. Provider
//! operation receipts are deliberately separate: nested operations can observe
//! the same encoder, whereas the emission site owns exactly one submission.

use super::{CommandProfilingReceipt, ObservedMetricState, ReceiptState};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhysicalSubmissionCounts {
    pub pending: u64,
    pub completed: u64,
    pub incomplete: u64,
    pub failed: u64,
    pub missing: u64,
}

/// Completeness among admitted physical submissions, not whole-command route
/// coverage. A copy-only submission can have a verified zero dispatch count
/// while device timestamp support remains unknown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalMetricCoverage {
    pub gpu_execution: bool,
    pub kernel_count: bool,
}

/// Sums for the attributed physical submissions only. `None` is unknown,
/// never an inferred zero. This type is deliberately not a command total.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalAttributedMetrics {
    pub gpu_execution: Option<Duration>,
    pub kernel_count: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalSubmissionTerminal {
    Completed {
        gpu_execution: Option<Duration>,
        kernel_count: Option<u64>,
    },
    Failed,
    Missing,
}

/// Send-safe, non-Clone ownership of one physical submission. Its producer
/// must admit before submitting and transfer this token to the actual progress
/// completion; dropping it records missing evidence while the root is open.
pub struct CommandPhysicalSubmission {
    receipt: Arc<Mutex<ReceiptState>>,
    settled: bool,
}

#[derive(Default)]
pub(super) struct PhysicalSubmissionState {
    admitted: u64,
    counts: PhysicalSubmissionCounts,
    metrics: ObservedMetricState,
}

impl PhysicalSubmissionState {
    pub(super) fn snapshot(
        &self,
    ) -> (
        PhysicalSubmissionCounts,
        PhysicalMetricCoverage,
        PhysicalAttributedMetrics,
    ) {
        let has_terminal = self.counts.completed != 0
            || self.counts.incomplete != 0
            || self.counts.failed != 0
            || self.counts.missing != 0;
        let coverage = PhysicalMetricCoverage {
            gpu_execution: has_terminal && self.counts.pending == 0 && self.metrics.gpu_complete,
            kernel_count: has_terminal && self.counts.pending == 0 && self.metrics.count_complete,
        };
        let metrics = PhysicalAttributedMetrics {
            gpu_execution: coverage.gpu_execution.then_some(self.metrics.gpu_execution),
            kernel_count: coverage.kernel_count.then_some(self.metrics.kernel_count),
        };
        (self.counts, coverage, metrics)
    }
}

impl CommandProfilingReceipt {
    /// Called only by the canonical physical-emission owner. Admission alone
    /// never changes `CommandRouteCoverage::Incomplete` or public GPU totals.
    pub fn begin_physical_submission(&self) -> Option<CommandPhysicalSubmission> {
        let mut state = self.0.lock().ok()?;
        if state.sealed || state.invalid {
            return None;
        }
        let physical = &mut state.physical;
        let (Some(admitted), Some(pending)) = (
            physical.admitted.checked_add(1),
            physical.counts.pending.checked_add(1),
        ) else {
            state.invalid = true;
            return None;
        };
        physical.admitted = admitted;
        physical.counts.pending = pending;
        Some(CommandPhysicalSubmission {
            receipt: Arc::clone(&self.0),
            settled: false,
        })
    }
}

impl CommandPhysicalSubmission {
    pub fn complete(mut self, terminal: PhysicalSubmissionTerminal) {
        self.settle(terminal);
    }

    fn settle(&mut self, terminal: PhysicalSubmissionTerminal) {
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
        let physical = &mut state.physical;
        match terminal {
            PhysicalSubmissionTerminal::Completed {
                gpu_execution,
                kernel_count,
            } => {
                if !physical.metrics.complete(gpu_execution, kernel_count) {
                    state.invalid = true;
                    return;
                }
            }
            PhysicalSubmissionTerminal::Failed | PhysicalSubmissionTerminal::Missing => {
                physical.metrics.missing();
            }
        }
        let counts = &mut physical.counts;
        let (counter, next_pending) = match terminal {
            PhysicalSubmissionTerminal::Completed {
                gpu_execution: Some(_),
                kernel_count: Some(_),
            } => (&mut counts.completed, counts.pending.checked_sub(1)),
            PhysicalSubmissionTerminal::Completed { .. } => {
                (&mut counts.incomplete, counts.pending.checked_sub(1))
            }
            PhysicalSubmissionTerminal::Failed => {
                (&mut counts.failed, counts.pending.checked_sub(1))
            }
            PhysicalSubmissionTerminal::Missing => {
                (&mut counts.missing, counts.pending.checked_sub(1))
            }
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

impl Drop for CommandPhysicalSubmission {
    fn drop(&mut self) {
        self.settle(PhysicalSubmissionTerminal::Missing);
    }
}

#[cfg(test)]
mod tests;
