//! Candidate-total feedback, separate from enqueue/stage tracing.
use super::*;
use runmat_accelerate_api::observation::{
    ProviderObservationFailure, ProviderObservationStatus, ProviderObservationSupport,
    ProviderOperationReceipt, ProviderOperationRecorder,
};
use runmat_runtime::context::RuntimePlacementFeedbackSource;

pub(super) struct FusionPlacementSelection {
    pub(super) observer: FusionPlacementObserver,
    feedback: Option<LocalPlacementFeedback>,
    receipt: Option<ProviderOperationReceipt>,
    started: Instant,
}

impl FusionPlacementSelection {
    pub(super) fn new(
        observer: FusionPlacementObserver,
        feedback: Option<LocalPlacementFeedback>,
        provider: &dyn AccelProvider,
        query: Option<&runmat_accelerate_api::ProviderFeasibilityQuery>,
    ) -> Self {
        // This boundary precedes selected-route staging, not just kernel enqueue.
        let started = Instant::now();
        let receipt = feedback.as_ref().and_then(|_| query).and_then(|query| {
            match provider.begin_operation_observation(query) {
                ProviderObservationSupport::Observed(receipt) => Some(receipt),
                ProviderObservationSupport::Unsupported => None,
            }
        });
        Self {
            observer,
            feedback,
            receipt,
            started,
        }
    }

    pub(super) fn recorder(&self) -> Option<ProviderOperationRecorder> {
        self.receipt
            .as_ref()
            .map(ProviderOperationReceipt::recorder)
    }

    pub(super) fn submitted(mut self) {
        let Some(mut receipt) = self.receipt.take() else {
            self.feedback = None; // Explicit unsupported observation, not success.
            return;
        };
        let Some(feedback) = self.feedback.take() else {
            return;
        };
        let cancellation = feedback.runtime.cancellation();
        if cancellation.is_cancelled() {
            receipt.fail(ProviderObservationFailure::Cancelled);
        } else if receipt.publish().is_err() {
            receipt.fail(ProviderObservationFailure::PublicationFailed);
        }
        let Some(service) = feedback.runtime.service_ports().placement() else {
            return;
        };
        let mut source = DeferredFusionFeedback {
            receipt,
            signature: feedback.signature,
            candidate: feedback.candidate,
            cancellation,
        };
        if let Some(ready) = source.take_ready() {
            let _ = service.observe(ready);
        } else if service.defer_feedback(Box::new(source)).is_err() {
            log::debug!(
                "fusion observation discarded: placement service cannot retain pending completion"
            );
        }
    }
}

impl Drop for FusionPlacementSelection {
    fn drop(&mut self) {
        let Some(feedback) = &self.feedback else {
            return;
        };
        let Some(service) = feedback.runtime.service_ports().placement() else {
            return;
        };
        let _ = service.observe(runmat_execution::PlacementFeedback {
            signature: feedback.signature.clone(),
            candidate: feedback.candidate.clone(),
            total_elapsed_ns: runmat_time::duration_ns_saturating(self.started.elapsed()),
            succeeded: false,
        });
    }
}

struct DeferredFusionFeedback {
    receipt: ProviderOperationReceipt,
    signature: runmat_execution::PlacementSignature,
    candidate: String,
    cancellation: runmat_execution::cancellation::CancellationToken,
}

impl RuntimePlacementFeedbackSource for DeferredFusionFeedback {
    fn signature(&self) -> &runmat_execution::PlacementSignature {
        &self.signature
    }
    fn take_ready(&mut self) -> Option<runmat_execution::PlacementFeedback> {
        if self.cancellation.is_cancelled() {
            self.receipt.fail(ProviderObservationFailure::Cancelled);
        }
        let observation = self.receipt.take_completed()?;
        let (elapsed, succeeded) = match observation.status {
            ProviderObservationStatus::Completed { total_elapsed, .. } => {
                (runmat_time::duration_ns_saturating(total_elapsed), true)
            }
            ProviderObservationStatus::Failed(_) => (0, false),
            ProviderObservationStatus::Pending => return None,
        };
        Some(runmat_execution::PlacementFeedback {
            signature: self.signature.clone(),
            candidate: self.candidate.clone(),
            total_elapsed_ns: elapsed,
            succeeded,
        })
    }
}

#[cfg(test)]
mod tests;
