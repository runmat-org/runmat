//! Owning-session delivery of provider-completed candidate totals.
use super::LocalPlacementFeedback;
use runmat_accelerate_api::observation::{
    ProviderObservationFailure, ProviderObservationStatus, ProviderOperationReceipt,
};
use runmat_runtime::context::RuntimePlacementFeedbackSource;

/// Publication follows output validation/adoption and temporary cleanup. The
/// deferred source retains only compact identity and cancellation, not runtime
/// or session, so a session owning that source cannot form a reference cycle.
pub(crate) fn publish_provider_feedback(
    mut receipt: ProviderOperationReceipt,
    feedback: LocalPlacementFeedback,
) {
    let cancellation = feedback.runtime.cancellation();
    if cancellation.is_cancelled() {
        receipt.fail(ProviderObservationFailure::Cancelled);
    } else if receipt.publish().is_err() {
        receipt.fail(ProviderObservationFailure::PublicationFailed);
    }
    let Some(service) = feedback.runtime.service_ports().placement() else {
        return;
    };
    let mut source = DeferredProviderFeedback {
        receipt,
        signature: feedback.signature,
        candidate: feedback.candidate,
        cancellation,
    };
    if let Some(ready) = source.take_ready() {
        let _ = service.observe(ready);
    } else if service.defer_feedback(Box::new(source)).is_err() {
        log::debug!(
            "provider observation discarded: placement service cannot retain pending completion"
        );
    }
}

struct DeferredProviderFeedback {
    receipt: ProviderOperationReceipt,
    signature: runmat_execution::PlacementSignature,
    candidate: String,
    cancellation: runmat_execution::cancellation::CancellationToken,
}

impl RuntimePlacementFeedbackSource for DeferredProviderFeedback {
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
