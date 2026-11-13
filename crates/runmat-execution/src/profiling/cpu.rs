//! Portable command-thread CPU accounting. Platform counter acquisition lives
//! in Runtime's timing service; this owner only checks attributed segments.

use super::CommandProfilingReceipt;
use std::sync::Arc;
use std::time::Duration;

/// A monotonic CPU counter for the thread making this call. Values from two
/// different sampler instances must never be subtracted from one another.
pub trait CommandThreadCpuSampler: Send + Sync {
    fn sample_current_thread(&self) -> Option<Duration>;
}

#[derive(Default)]
pub(super) struct CommandCpuState {
    sampler: Option<Arc<dyn CommandThreadCpuSampler>>,
    total: Duration,
    measured: bool,
    invalid: bool,
}

impl CommandCpuState {
    pub(super) fn new(sampler: Option<Arc<dyn CommandThreadCpuSampler>>) -> Self {
        Self {
            sampler,
            ..Self::default()
        }
    }

    pub(super) fn snapshot(&self) -> Option<Duration> {
        (self.measured && !self.invalid).then_some(self.total)
    }

    fn add(&mut self, start: Duration, end: Duration) {
        let Some(elapsed) = end.checked_sub(start) else {
            self.invalid = true;
            return;
        };
        let Some(total) = self.total.checked_add(elapsed) else {
            self.invalid = true;
            return;
        };
        self.total = total;
        self.measured = true;
    }
}

impl CommandProfilingReceipt {
    pub(super) fn thread_cpu_sampler(&self) -> Option<Arc<dyn CommandThreadCpuSampler>> {
        let state = self.0.lock().ok()?;
        (!state.sealed && !state.cpu.invalid)
            .then(|| state.cpu.sampler.as_ref().cloned())
            .flatten()
    }

    pub(super) fn record_thread_cpu_segment(&self, start: Duration, end: Duration) {
        if let Ok(mut state) = self.0.lock() {
            if !state.sealed {
                state.cpu.add(start, end);
            }
        }
    }

    pub(super) fn invalidate_thread_cpu(&self) {
        if let Ok(mut state) = self.0.lock() {
            if !state.sealed {
                state.cpu.invalid = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn checked_cpu_overflow_and_late_settlement_remain_unknown() {
        let mut state = CommandCpuState::new(None);
        state.total = Duration::MAX;
        state.add(Duration::ZERO, Duration::from_nanos(1));
        assert_eq!(state.snapshot(), None);

        let root = super::super::CommandProfilingRoot::new();
        let receipt = root.receipt();
        assert_eq!(root.seal().cpu_thread_attributed, None);
        receipt.record_thread_cpu_segment(Duration::ZERO, Duration::from_nanos(1));
        assert_eq!(receipt.0.lock().unwrap().cpu.snapshot(), None);
    }
}
