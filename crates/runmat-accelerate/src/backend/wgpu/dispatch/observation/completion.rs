//! Exact physical-submission completion owners for native and browser WGPU.
use runmat_accelerate_api::observation::{
    ProviderCompletedSubmission, ProviderObservationFailure, ProviderOperationRecorder,
    StochasticEvolutionSourceTransferRetirementNotifier, WgpuSubmissionMeasurement,
};
#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::observation::{
    WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use runmat_accelerate_api::WgpuDeviceAccess;
use runmat_execution::profiling::{CommandPhysicalSubmission, PhysicalSubmissionTerminal};

use super::completion_signal::{self, Receiver};
use super::compute_pass::{DecodedTimestamps, ObservationBuffer, TimestampReadback};
use super::probe_ticket::ProbeFenceTicket;
use super::submitted_retention::SubmittedAllocationRetention;
use super::InlineSlots;

pub(super) struct SourceTransferRetirement {
    notifier: Option<StochasticEvolutionSourceTransferRetirementNotifier>,
    succeeded: bool,
}

impl SourceTransferRetirement {
    pub(super) fn new(notifier: StochasticEvolutionSourceTransferRetirementNotifier) -> Self {
        Self {
            notifier: Some(notifier),
            succeeded: false,
        }
    }

    fn complete(&mut self, succeeded: bool) {
        self.succeeded = succeeded;
    }
}

impl Drop for SourceTransferRetirement {
    fn drop(&mut self) {
        if let Some(notifier) = self.notifier.take() {
            notifier.finish(self.succeeded);
        }
    }
}

pub(super) fn map_timing(readback: &TimestampReadback) -> Receiver<bool> {
    let (send, result) = completion_signal::channel();
    readback.staging.map_async(move |status| {
        // The callback never enters runtime or WGPU cleanup authority.
        send.send(status.is_ok());
    });
    result
}

#[cfg(target_arch = "wasm32")]
pub(super) fn map_direct_fence(fence: &ObservationBuffer) -> Receiver<bool> {
    let (send, result) = completion_signal::channel();
    fence.map_async(move |status| {
        send.send(status.is_ok());
    });
    result
}

pub(super) struct DirectCompletion {
    pub(super) access: WgpuDeviceAccess,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) unretired_submission: Option<wgpu::SubmissionIndex>,
    pub(super) readback: Option<TimestampReadback>,
    pub(super) fence: Option<ObservationBuffer>,
    #[cfg(target_arch = "wasm32")]
    pub(super) _fence_permit: Option<runmat_accelerate_api::observation::WgpuQueryPermit>,
    #[cfg(target_arch = "wasm32")]
    pub(super) _tail_permit: Option<runmat_accelerate_api::observation::WgpuTailPermit>,
    pub(super) result: Option<Receiver<bool>>,
    pub(super) physical: Option<CommandPhysicalSubmission>,
    pub(super) measurement: WgpuSubmissionMeasurement,
    pub(super) submitted_allocation_retention: Option<SubmittedAllocationRetention>,
    // Last field: the waiter is released only after preceding completion
    // resources and retained owners have been dropped.
    pub(super) source_transfer_retirement: Option<SourceTransferRetirement>,
}

impl DirectCompletion {
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn retire_scoped(mut self, confirmed: bool) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.unretired_submission = None;
        }
        if !confirmed {
            drop(self.physical.take());
        }
        self.retire(confirmed, false);
    }
    pub(super) fn retire(mut self, progressed: bool, mapped: bool) {
        let access = self.access.clone();
        let elapsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            access.with_cleanup(|cleanup| {
                let elapsed = self
                    .readback
                    .as_ref()
                    .and_then(|readback| mapped.then(|| readback.decode_mapped()));
                if let Some(readback) = self.readback.take() {
                    cleanup.unmap_buffer(&readback.staging);
                }
                if let Some(fence) = self.fence.take() {
                    cleanup.unmap_buffer(&fence);
                }
                elapsed.unwrap_or_default()
            })
        }));
        let cleaned = matches!(&elapsed, Ok(Ok(_)));
        if let Some(physical) = self.physical.take() {
            match elapsed {
                Ok(Ok(elapsed)) if progressed => {
                    physical.complete(PhysicalSubmissionTerminal::Completed {
                        gpu_execution: elapsed.device_elapsed.or(self.measurement.device_elapsed),
                        kernel_count: self.measurement.dispatches,
                    })
                }
                _ => physical.complete(PhysicalSubmissionTerminal::Failed),
            }
        }
        if let Some(retention) = self.submitted_allocation_retention.take() {
            retention.retire(progressed && cleaned);
        }
        if let Some(retirement) = self.source_transfer_retirement.as_mut() {
            retirement.complete(progressed && cleaned);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn after_poll(mut self, status: Result<(), WgpuObservationProgressError>) {
        if status.is_err() {
            if let Some(submission) = self.unretired_submission.take() {
                settle_failed_progress(&self.access, submission, self.readback.as_ref());
            }
        }
        let mapped = self
            .result
            .as_mut()
            .and_then(|result| result.take_after_progress(status.is_ok()))
            .unwrap_or(false);
        // Keep the exact submission in this owner through a successful map
        // wait, so unwind still takes the Drop retirement path before unmap.
        self.unretired_submission = None;
        self.retire(status.is_ok(), mapped);
    }
}

impl Drop for DirectCompletion {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(submission) = self.unretired_submission.take() {
            // A custom progress port may discard its accepted callback or
            // panic before invoking it. The completion itself owns the exact
            // submission until either callback or this fallback settles it.
            settle_failed_progress(&self.access, submission, self.readback.as_ref());
        }
        if let Some(readback) = self.readback.take() {
            let access = self.access.clone();
            let retired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                access.with_cleanup(|cleanup| cleanup.unmap_buffer(&readback.staging))
            }));
            if !matches!(retired, Ok(Ok(()))) {
                log::error!("WGPU direct timestamp readback teardown failed");
            }
        }
        if let Some(fence) = self.fence.take() {
            let access = self.access.clone();
            let retired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                access.with_cleanup(|cleanup| cleanup.unmap_buffer(&fence))
            }));
            if !matches!(retired, Ok(Ok(()))) {
                log::error!("WGPU direct completion fence teardown failed");
            }
        }
        drop(self.physical.take());
        if let Some(retention) = self.submitted_allocation_retention.take() {
            retention.retire(false);
        }
        // A remaining physical submission records Missing on drop.
    }
}

pub(super) struct FenceCompletion {
    pub(super) access: WgpuDeviceAccess,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) unretired_submission: Option<wgpu::SubmissionIndex>,
    pub(super) fence: Option<ObservationBuffer>,
    pub(super) recorder: ProviderOperationRecorder,
    pub(super) result: Receiver<Option<ProviderCompletedSubmission>>,
    pub(super) measurement: WgpuSubmissionMeasurement,
    pub(super) readback: Option<TimestampReadback>,
    pub(super) readback_result: Option<Receiver<bool>>,
    pub(super) probe: Option<ProbeFenceTicket>,
    pub(super) resource_ticket: Option<runmat_accelerate_api::observation::ProviderResourceTicket>,
    pub(super) submitted_resources:
        InlineSlots<runmat_accelerate_api::observation::ProviderResourceTicket, 3>,
    pub(super) submitted_allocation_retention: Option<SubmittedAllocationRetention>,
    pub(super) physical: Option<CommandPhysicalSubmission>,
    // Last field for exact host-control/resource retirement ordering.
    pub(super) source_transfer_retirement: Option<SourceTransferRetirement>,
}

impl FenceCompletion {
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn retire_scoped(mut self, confirmed: bool) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.unretired_submission = None;
        }
        if !confirmed {
            drop(self.physical.take());
        }
        let completed = self.result.take_after_progress(confirmed).flatten();
        // Queue retirement is proven, but timestamps are optional. Do not
        // wait on mapping/Promise delivery to retire GPU ownership.
        self.cleanup(completed, confirmed, false);
    }
    fn release_fence(&mut self, timing_mapped: bool) -> (bool, DecodedTimestamps) {
        let access = self.access.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            access.with_cleanup(|cleanup| {
                let elapsed = self
                    .readback
                    .as_ref()
                    .and_then(|readback| timing_mapped.then(|| readback.decode_mapped()));
                if let Some(readback) = self.readback.take() {
                    cleanup.unmap_buffer(&readback.staging);
                }
                if let Some(fence) = self.fence.take() {
                    cleanup.unmap_buffer(&fence);
                    drop(fence);
                }
                elapsed.unwrap_or_default()
            })
        }));
        let (released, elapsed) = match result {
            Ok(Ok(elapsed)) => (true, elapsed),
            _ => (false, DecodedTimestamps::default()),
        };
        if !released {
            if let Some(ticket) = &self.resource_ticket {
                ticket.fail_owner();
            }
            for ticket in self.submitted_resources.iter() {
                ticket.fail_owner();
            }
        }
        (released, elapsed)
    }

    pub(super) fn cleanup(
        mut self,
        completed: Option<ProviderCompletedSubmission>,
        progress_ok: bool,
        timing_mapped: bool,
    ) {
        log::debug!(
            "WGPU provider observation cleanup: progress_ok={progress_ok}, fence_delivered={}, timing_mapped={timing_mapped}",
            completed.is_some()
        );
        let completion_delivered = completed.is_some();
        if !progress_ok || !completion_delivered {
            self.recorder
                .fail(ProviderObservationFailure::ProviderFailed);
        }
        // Outside the original emission. unmap can deliver a cancelled map
        // callback synchronously; that callback only sends its result.
        let (cleaned, elapsed) = self.release_fence(timing_mapped);
        log::debug!("WGPU provider observation fence cleanup: released={cleaned}");
        if progress_ok && cleaned {
            if let Some(completed) = completed {
                self.measurement.compute_elapsed =
                    elapsed.compute_elapsed.or(self.measurement.compute_elapsed);
                self.measurement.device_elapsed =
                    elapsed.device_elapsed.or(self.measurement.device_elapsed);
                if let Some(physical) = self.physical.take() {
                    physical.complete(PhysicalSubmissionTerminal::Completed {
                        gpu_execution: self.measurement.device_elapsed,
                        kernel_count: self.measurement.dispatches,
                    });
                }
                completed.complete_with_wgpu(self.measurement);
                if let Some(probe) = &self.probe {
                    probe.complete();
                }
            } else {
                if let Some(physical) = self.physical.take() {
                    physical.complete(PhysicalSubmissionTerminal::Failed);
                }
                if let Some(probe) = &self.probe {
                    probe.fail();
                }
            }
        } else {
            if let Some(physical) = self.physical.take() {
                physical.complete(PhysicalSubmissionTerminal::Failed);
            }
            self.recorder
                .fail(ProviderObservationFailure::ProviderFailed);
            if let Some(completed) = completed {
                completed.fail();
            }
            if let Some(probe) = &self.probe {
                probe.fail();
            }
        }
        drop(self.resource_ticket.take());
        self.submitted_resources.clear();
        if let Some(retention) = self.submitted_allocation_retention.take() {
            retention.retire(progress_ok && cleaned && completion_delivered);
        }
        if let Some(retirement) = self.source_transfer_retirement.as_mut() {
            retirement.complete(progress_ok && cleaned && completion_delivered);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn after_poll(mut self, status: Result<(), WgpuObservationProgressError>) {
        log::debug!("WGPU provider observation progress callback: {status:?}");
        if status.is_err() {
            if let Some(submission) = self.unretired_submission.take() {
                settle_failed_progress(&self.access, submission, self.readback.as_ref());
            }
        }
        let completed = self.result.take_after_progress(status.is_ok()).flatten();
        let timing_mapped = self
            .readback_result
            .as_mut()
            .and_then(|result| result.take_after_progress(status.is_ok()))
            .unwrap_or(false);
        self.unretired_submission = None;
        self.cleanup(completed, status.is_ok(), timing_mapped);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn settle_failed_progress(
    access: &WgpuDeviceAccess,
    submission: wgpu::SubmissionIndex,
    readback: Option<&TimestampReadback>,
) {
    // The progress port's Stopped status is not proof that the submitted work
    // retired. This exceptional wait is outside the emission gate and targets
    // exactly this submission; it never turns failed evidence into success.
    exact_wait_or_poison(
        || {
            access.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
        },
        || {
            if let Some(readback) = readback {
                readback.poison_query_budget();
            }
        },
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn exact_wait_or_poison(wait: impl FnOnce(), poison_query_budget: impl FnOnce()) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(wait)).is_err() {
        // wgpu-core owns live encoder resources until the backend retires its
        // submission. Disable *optional* query re-admission before the owned
        // readback/permit is dropped; required submission slots are unchanged.
        poison_query_budget();
        log::error!("WGPU exact submission retirement failed after observation progress stopped");
    }
}

impl Drop for FenceCompletion {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(submission) = self.unretired_submission.take() {
            settle_failed_progress(&self.access, submission, self.readback.as_ref());
        }
        if self.fence.is_some() {
            if let Some(probe) = &self.probe {
                probe.fail();
            }
            self.recorder.fail(ProviderObservationFailure::Abandoned);
            if !self.release_fence(false).0 {
                log::error!("WGPU observation fence teardown failed; no retry attempted");
            }
        }
        drop(self.physical.take());
        drop(self.resource_ticket.take());
        self.submitted_resources.clear();
        if let Some(retention) = self.submitted_allocation_retention.take() {
            retention.retire(false);
        }
    }
}

pub(super) struct PendingObservation {
    pub(super) completion: FenceCompletion,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) submission: wgpu::SubmissionIndex,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) progress: Option<Box<dyn WgpuObservationProgressReservation>>,
}

pub(super) struct PendingDirectObservation {
    pub(super) completion: DirectCompletion,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) submission: wgpu::SubmissionIndex,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) progress: Option<Box<dyn WgpuObservationProgressReservation>>,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::exact_wait_or_poison;
    use runmat_accelerate_api::observation::ProviderObservationBudget;
    use std::{num::NonZeroU32, sync::Arc};

    #[test]
    fn failed_exact_wait_poisons_optional_admission_before_permit_drop() {
        let budget = Arc::new(ProviderObservationBudget::with_wgpu_query_bytes(
            NonZeroU32::new(1).unwrap(),
            256,
        ));
        let permit = budget.try_reserve_wgpu_query(256).expect("query admitted");
        exact_wait_or_poison(
            || panic!("simulated device poll failure"),
            || permit.poison_budget(),
        );
        drop(permit);
        assert!(budget.try_reserve_wgpu_query(1).is_none());
        let unpoisoned = Arc::new(ProviderObservationBudget::with_wgpu_query_bytes(
            NonZeroU32::new(1).unwrap(),
            256,
        ));
        let permit = unpoisoned
            .try_reserve_wgpu_query(256)
            .expect("query admitted");
        exact_wait_or_poison(|| {}, || permit.poison_budget());
        drop(permit);
        assert!(unpoisoned.try_reserve_wgpu_query(256).is_some());
    }
}
