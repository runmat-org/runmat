//! Candidate-total feedback, separate from enqueue/stage tracing.
use super::*;
use runmat_accelerate_api::observation::{
    ProviderObservationSupport, ProviderOperationReceipt, ProviderOperationRecorder,
};
use runmat_runtime::context::VmFusionCompletionRight;

pub(super) struct FusionPlacementSelection {
    pub(super) observer: FusionPlacementObserver,
    feedback: Option<LocalPlacementFeedback>,
    receipt: Option<ProviderOperationReceipt>,
    started: Instant,
    drop_reports_failure: bool,
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
        let receipt = feedback
            .as_ref()
            .filter(|feedback| feedback.is_current())
            .and(query)
            .and_then(|query| match provider.begin_operation_observation(query) {
                ProviderObservationSupport::Observed(receipt) => Some(receipt),
                ProviderObservationSupport::Unsupported => None,
            });
        Self {
            observer,
            feedback,
            receipt,
            started,
            drop_reports_failure: true,
        }
    }

    pub(super) fn new_vm(
        observer: FusionPlacementObserver,
        feedback: Option<LocalPlacementFeedback>,
        provider: &dyn AccelProvider,
        query: Option<&runmat_accelerate_api::ProviderFeasibilityQuery>,
    ) -> Self {
        let mut selection = Self::new(observer, feedback, provider, query);
        selection.drop_reports_failure = false;
        selection
    }

    pub(super) fn recorder(&self) -> Option<ProviderOperationRecorder> {
        self.receipt
            .as_ref()
            .map(ProviderOperationReceipt::recorder)
    }

    pub(super) fn require_vm_provider_execution(
        &mut self,
        device_id: u32,
        operation: runmat_accelerate_api::ProviderOperationIdentity,
    ) {
        if let Some(receipt) = self.receipt.as_mut() {
            receipt.require_provider_execution(device_id, operation);
        }
    }

    pub(super) fn into_vm_completion(
        mut self,
        right: VmFusionCompletionRight,
    ) -> Option<FusionProviderCompletion> {
        let receipt = self.receipt.take();
        let feedback = self.feedback.take();
        FusionProviderCompletion::new(receipt?, feedback?, right)
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
        let Some(feedback) = self.feedback.as_ref().filter(|_| self.drop_reports_failure) else {
            return;
        };
        feedback.clone().observe(
            runmat_time::duration_ns_saturating(self.started.elapsed()),
            false,
        );
    }
}

#[cfg(test)]
mod tests;
