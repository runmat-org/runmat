//! Candidate-total feedback, separate from enqueue/stage tracing.
use super::*;
use runmat_accelerate_api::observation::{
    ProviderObservationSupport, ProviderOperationReceipt, ProviderOperationRecorder,
};

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
        let Some(receipt) = self.receipt.take() else {
            self.feedback = None; // Explicit unsupported observation, not success.
            return;
        };
        let Some(feedback) = self.feedback.take() else {
            return;
        };
        crate::placement::publish_provider_feedback(receipt, feedback);
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

#[cfg(test)]
mod tests;
