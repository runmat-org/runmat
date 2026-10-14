//! Exact physical-submission completion owners for native and browser WGPU.
use futures::channel::oneshot;
use runmat_accelerate_api::observation::{
    ProviderCompletedSubmission, ProviderObservationFailure, ProviderOperationRecorder,
    WgpuSubmissionMeasurement,
};
#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::observation::{
    WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use runmat_accelerate_api::WgpuDeviceAccess;
use runmat_execution::profiling::{CommandPhysicalSubmission, PhysicalSubmissionTerminal};

use super::compute_pass::{DecodedTimestamps, TimestampReadback};
use super::probe_ticket::ProbeFenceTicket;

#[cfg(not(target_arch = "wasm32"))]
mod native_map;

pub(super) fn map_timing(readback: &TimestampReadback) -> oneshot::Receiver<bool> {
    let (send, result) = oneshot::channel();
    readback
        .staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |status| {
            // The callback never enters runtime or WGPU cleanup authority.
            let _ = send.send(status.is_ok());
        });
    result
}

#[cfg(target_arch = "wasm32")]
pub(super) fn map_direct_fence(
    fence: &runmat_accelerate_api::WgpuBufferOwner,
) -> oneshot::Receiver<bool> {
    let (send, result) = oneshot::channel();
    fence
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |status| {
            let _ = send.send(status.is_ok());
        });
    result
}

pub(super) struct DirectCompletion {
    pub(super) access: WgpuDeviceAccess,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) unretired_submission: Option<wgpu::SubmissionIndex>,
    pub(super) readback: Option<TimestampReadback>,
    pub(super) fence: Option<runmat_accelerate_api::WgpuBufferOwner>,
    #[cfg(target_arch = "wasm32")]
    pub(super) _fence_permit: Option<runmat_accelerate_api::observation::WgpuQueryPermit>,
    pub(super) result: Option<oneshot::Receiver<bool>>,
    pub(super) physical: Option<CommandPhysicalSubmission>,
    pub(super) measurement: WgpuSubmissionMeasurement,
}

impl DirectCompletion {
    fn retire(mut self, progressed: bool, mapped: bool) {
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
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn after_poll(mut self, status: Result<(), WgpuObservationProgressError>) {
        if status.is_err() {
            if let Some(submission) = self.unretired_submission.take() {
                settle_failed_progress(&self.access, submission, self.readback.as_ref());
            }
        }
        let mapped = native_map::settle_after_progress(status.is_ok(), self.result.as_mut())
            .unwrap_or(false);
        // Keep the exact submission in this owner through a successful map
        // wait, so unwind still takes the Drop retirement path before unmap.
        self.unretired_submission = None;
        self.retire(status.is_ok(), mapped);
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) async fn after_callback(mut self) {
        let mapped = match self.result.take() {
            Some(result) => result.await.unwrap_or(false),
            None => false,
        };
        self.retire(mapped, mapped);
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
        // A remaining physical submission records Missing on drop.
    }
}

pub(super) struct FenceCompletion {
    pub(super) access: WgpuDeviceAccess,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) unretired_submission: Option<wgpu::SubmissionIndex>,
    pub(super) fence: Option<runmat_accelerate_api::WgpuBufferOwner>,
    pub(super) recorder: ProviderOperationRecorder,
    pub(super) result: oneshot::Receiver<Option<ProviderCompletedSubmission>>,
    pub(super) measurement: WgpuSubmissionMeasurement,
    pub(super) readback: Option<TimestampReadback>,
    pub(super) readback_result: Option<oneshot::Receiver<bool>>,
    pub(super) probe: Option<ProbeFenceTicket>,
    pub(super) resource_ticket: Option<runmat_accelerate_api::observation::ProviderResourceTicket>,
    pub(super) submitted_resources: Vec<runmat_accelerate_api::observation::ProviderResourceTicket>,
    pub(super) physical: Option<CommandPhysicalSubmission>,
}

impl FenceCompletion {
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
            for ticket in &self.submitted_resources {
                ticket.fail_owner();
            }
        }
        (released, elapsed)
    }

    fn cleanup(
        mut self,
        completed: Option<ProviderCompletedSubmission>,
        progress_ok: bool,
        timing_mapped: bool,
    ) {
        log::debug!(
            "WGPU provider observation cleanup: progress_ok={progress_ok}, fence_delivered={}, timing_mapped={timing_mapped}",
            completed.is_some()
        );
        if !progress_ok || completed.is_none() {
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
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn after_poll(mut self, status: Result<(), WgpuObservationProgressError>) {
        log::debug!("WGPU provider observation progress callback: {status:?}");
        if status.is_err() {
            if let Some(submission) = self.unretired_submission.take() {
                settle_failed_progress(&self.access, submission, self.readback.as_ref());
            }
        }
        let completed =
            native_map::settle_after_progress(status.is_ok(), Some(&mut self.result)).flatten();
        let timing_mapped =
            native_map::settle_after_progress(status.is_ok(), self.readback_result.as_mut())
                .unwrap_or(false);
        self.unretired_submission = None;
        self.cleanup(completed, status.is_ok(), timing_mapped);
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) async fn after_callback(mut self) {
        let completed = (&mut self.result).await.ok().flatten();
        let timing_mapped = match self.readback_result.take() {
            Some(result) => result.await.unwrap_or(false),
            None => false,
        };
        self.cleanup(completed, true, timing_mapped);
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
    }
}

pub(super) struct PendingObservation {
    pub(super) completion: FenceCompletion,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) submission: wgpu::SubmissionIndex,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) progress: Box<dyn WgpuObservationProgressReservation>,
}

pub(super) struct PendingDirectObservation {
    pub(super) completion: DirectCompletion,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) submission: wgpu::SubmissionIndex,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) progress: Box<dyn WgpuObservationProgressReservation>,
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
