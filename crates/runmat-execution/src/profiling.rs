//! One command's provider-observation accounting. This is not a provider
//! clock, a kernel counter, or evidence that every device route was observed.
//! In particular, an empty/pending-free receipt never certifies GPU coverage.

use std::sync::{Arc, Mutex};
use std::time::Duration;

mod activation;
mod cpu;
mod deferred;
mod physical;
pub use activation::{
    current_command_profiling, observe_command_future, select_command_profiling,
    with_command_profiling, CommandProfilingSelectionGuard,
};
pub use cpu::CommandThreadCpuSampler;
pub use deferred::{CommandDeferredWork, DeferredWorkCounts, DeferredWorkTerminal};
pub use physical::{
    CommandPhysicalSubmission, PhysicalAttributedMetrics, PhysicalMetricCoverage,
    PhysicalSubmissionCounts, PhysicalSubmissionTerminal,
};

#[derive(Clone)]
pub struct CommandProfilingReceipt(Arc<Mutex<ReceiptState>>);

/// Sole, non-cloneable publication authority for one admitted command.
/// Runtime contexts and deferred operations receive only the cloneable
/// contribution handle; a nested child cannot close its parent's accounting.
pub struct CommandProfilingRoot {
    receipt: CommandProfilingReceipt,
    sealed: bool,
}

#[derive(Default)]
struct ReceiptState {
    sealed: bool,
    invalid: bool,
    unproven_device_route: bool,
    unproven_device_time: bool,
    admitted: u64,
    observations: ProviderObservationCounts,
    metrics: ObservedMetricState,
    deferred: deferred::DeferredWorkState,
    physical: physical::PhysicalSubmissionState,
    cpu: cpu::CommandCpuState,
}

/// Checked receipt-local sums are retained separately from coverage. Nested
/// receipts may overlap the same physical work, so even complete fields here
/// are never publishable command totals without future emission deduplication
/// and all-route authority.
struct ObservedMetricState {
    gpu_execution: Duration,
    kernel_count: u64,
    gpu_complete: bool,
    count_complete: bool,
}

impl Default for ObservedMetricState {
    fn default() -> Self {
        Self {
            gpu_execution: Duration::ZERO,
            kernel_count: 0,
            gpu_complete: true,
            count_complete: true,
        }
    }
}

impl ObservedMetricState {
    fn complete(&mut self, gpu_execution: Option<Duration>, kernel_count: Option<u64>) -> bool {
        match gpu_execution {
            Some(duration) => {
                let Some(total) = self.gpu_execution.checked_add(duration) else {
                    return false;
                };
                self.gpu_execution = total;
            }
            None => self.gpu_complete = false,
        }
        match kernel_count {
            Some(count) => {
                let Some(total) = self.kernel_count.checked_add(count) else {
                    return false;
                };
                self.kernel_count = total;
            }
            None => self.count_complete = false,
        }
        true
    }

    fn missing(&mut self) {
        self.gpu_complete = false;
        self.count_complete = false;
    }
}

/// Counts unique *receipt settlements*, not physical dispatches. Nested
/// observations can cover the same device work and must not be summed as
/// command-wide time or launch count.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProviderObservationCounts {
    pub pending: u64,
    pub completed: u64,
    pub incomplete: u64,
    pub failed: u64,
    pub missing: u64,
}

/// Completeness among the receipts this command observed, not proof that all
/// GPU routes were attributed. Time and dispatch coverage are independent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservedMetricCoverage {
    pub gpu_execution: bool,
    pub kernel_count: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandRouteCoverage {
    /// Direct device routes can bypass provider-operation receipts. Only an
    /// eventual all-route device-emission authority may change this status.
    Incomplete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommandProfilingSnapshot {
    /// None means overflow or poisoned state, never a plausible zero sample.
    pub observations: Option<ProviderObservationCounts>,
    pub observed_metric_coverage: Option<ObservedMetricCoverage>,
    /// Admitted future work not necessarily emitted to a device yet. A zero
    /// pending count alone never proves all-route command coverage.
    pub deferred_work: Option<DeferredWorkCounts>,
    /// Unique physical submission settlements, distinct from potentially
    /// overlapping provider-operation observations.
    pub physical_submissions: Option<PhysicalSubmissionCounts>,
    /// Validity only among admitted physical submissions; it does not certify
    /// that every device route was attributed to this command.
    pub physical_metric_coverage: Option<PhysicalMetricCoverage>,
    /// Checked sums only for attributed physical submissions. These are not
    /// whole-command totals while route coverage remains incomplete.
    pub physical_attributed_metrics: Option<PhysicalAttributedMetrics>,
    /// Checked CPU time on explicitly selected origin threads only. This is
    /// diagnostic partial attribution, not all-route public `cpu_ms`.
    pub cpu_thread_attributed: Option<Duration>,
    /// An admitted effect escaped the command's canonical physical-emission
    /// owner. False alone is not proof that every route was observed.
    pub unproven_device_route: bool,
    /// An attributed device transfer escaped timed encoder intervals. It
    /// affects device duration, but does not erase verified launch counts.
    pub unproven_device_time: bool,
    pub route_coverage: CommandRouteCoverage,
    pub gpu_execution: Option<Duration>,
    pub kernel_count: Option<u64>,
}

/// A terminal fact supplied at the same one-shot receipt-consumption site as
/// placement feedback. Host elapsed/submission count cannot populate these
/// device-only measurements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderObservationTerminal {
    Completed {
        gpu_execution: Option<Duration>,
        kernel_count: Option<u64>,
    },
    Failed,
    Missing,
}

/// Linear, Send-safe ownership of one provider observation. Dropping before
/// settlement records missing evidence while the command is still open.
pub struct CommandProviderContribution {
    receipt: Arc<Mutex<ReceiptState>>,
    settled: bool,
}

impl CommandProfilingRoot {
    pub fn new() -> Self {
        Self::new_with_thread_cpu_sampler(None)
    }

    pub fn new_with_thread_cpu_sampler(sampler: Option<Arc<dyn CommandThreadCpuSampler>>) -> Self {
        let state = ReceiptState {
            cpu: cpu::CommandCpuState::new(sampler),
            ..ReceiptState::default()
        };
        Self {
            receipt: CommandProfilingReceipt(Arc::new(Mutex::new(state))),
            sealed: false,
        }
    }

    pub fn receipt(&self) -> CommandProfilingReceipt {
        self.receipt.clone()
    }

    pub fn seal(mut self) -> CommandProfilingSnapshot {
        self.sealed = true;
        self.receipt.seal_owned()
    }
}

impl Default for CommandProfilingRoot {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CommandProfilingRoot {
    fn drop(&mut self) {
        if !self.sealed {
            let _ = self.receipt.seal_owned();
        }
    }
}

impl CommandProfilingReceipt {
    /// A monotone negative fact. This never grants route coverage and remains
    /// meaningful if the command root was created outside Core.
    pub fn mark_unproven_device_route(&self) {
        if let Ok(mut state) = self.0.lock() {
            state.unproven_device_route = true;
        }
    }

    pub fn mark_unproven_device_time(&self) {
        if let Ok(mut state) = self.0.lock() {
            state.unproven_device_time = true;
        }
    }

    pub fn same_command(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    fn seal_owned(&self) -> CommandProfilingSnapshot {
        let valid = self.0.lock().ok().and_then(|mut state| {
            state.sealed = true;
            let has_terminal = state.observations.completed != 0
                || state.observations.incomplete != 0
                || state.observations.failed != 0
                || state.observations.missing != 0;
            (!state.invalid).then_some((
                state.observations,
                ObservedMetricCoverage {
                    gpu_execution: has_terminal
                        && state.observations.pending == 0
                        && state.metrics.gpu_complete,
                    kernel_count: has_terminal
                        && state.observations.pending == 0
                        && state.metrics.count_complete,
                },
                state.deferred.counts(),
                state.physical.snapshot(),
                state.cpu.snapshot(),
                state.unproven_device_route,
                state.unproven_device_time,
            ))
        });
        CommandProfilingSnapshot {
            observations: valid.map(|(observations, _, _, _, _, _, _)| observations),
            observed_metric_coverage: valid.map(|(_, coverage, _, _, _, _, _)| coverage),
            deferred_work: valid.map(|(_, _, deferred, _, _, _, _)| deferred),
            physical_submissions: valid.map(|(_, _, _, (counts, _, _), _, _, _)| counts),
            physical_metric_coverage: valid.map(|(_, _, _, (_, coverage, _), _, _, _)| coverage),
            physical_attributed_metrics: valid.map(|(_, _, _, (_, _, metrics), _, _, _)| metrics),
            cpu_thread_attributed: valid.and_then(|(_, _, _, _, cpu, _, _)| cpu),
            unproven_device_route: valid
                .map(|(_, _, _, _, _, unknown, _)| unknown)
                .unwrap_or(true),
            unproven_device_time: valid
                .map(|(_, _, _, _, _, _, unknown)| unknown)
                .unwrap_or(true),
            route_coverage: CommandRouteCoverage::Incomplete,
            gpu_execution: None,
            kernel_count: None,
        }
    }

    /// No allocation occurs for a runtime without a command owner: callers
    /// first inspect their optional RuntimeContext receipt.
    pub fn begin_provider_operation(&self) -> Option<CommandProviderContribution> {
        let mut state = self.0.lock().ok()?;
        if state.sealed || state.invalid {
            return None;
        }
        let (Some(next_admitted), Some(next_pending)) = (
            state.admitted.checked_add(1),
            state.observations.pending.checked_add(1),
        ) else {
            state.invalid = true;
            return None;
        };
        state.admitted = next_admitted;
        state.observations.pending = next_pending;
        Some(CommandProviderContribution {
            receipt: Arc::clone(&self.0),
            settled: false,
        })
    }
}

impl CommandProviderContribution {
    pub fn complete(mut self, terminal: ProviderObservationTerminal) {
        self.settle(terminal);
    }

    fn settle(&mut self, terminal: ProviderObservationTerminal) {
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
        match terminal {
            ProviderObservationTerminal::Completed {
                gpu_execution,
                kernel_count,
            } => {
                if !state.metrics.complete(gpu_execution, kernel_count) {
                    state.invalid = true;
                    return;
                }
            }
            ProviderObservationTerminal::Failed | ProviderObservationTerminal::Missing => {
                state.metrics.missing();
            }
        }
        let counts = &mut state.observations;
        let (counter, next_pending) = match terminal {
            ProviderObservationTerminal::Completed {
                gpu_execution: Some(_),
                kernel_count: Some(_),
            } => (&mut counts.completed, counts.pending.checked_sub(1)),
            ProviderObservationTerminal::Completed { .. } => {
                (&mut counts.incomplete, counts.pending.checked_sub(1))
            }
            ProviderObservationTerminal::Failed => {
                (&mut counts.failed, counts.pending.checked_sub(1))
            }
            ProviderObservationTerminal::Missing => {
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

impl Drop for CommandProviderContribution {
    fn drop(&mut self) {
        self.settle(ProviderObservationTerminal::Missing);
    }
}

#[cfg(test)]
mod tests;
