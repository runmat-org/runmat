//! Completion-qualified feedback for a selected route whose publication is a
//! provider operation. Typed query, resource, and output admission stay with
//! the caller; this owner never chooses or replays an execution route.

use super::{
    RuntimeContext, RuntimePlacementFeedbackSource, RuntimeWholeWorkProbeLease,
    RuntimeWholeWorkSelection,
};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationFailure, ProviderObservationStatus, ProviderOperationReceipt,
    ProviderOperationRecorder,
};
use runmat_execution::{
    cancellation::CancellationToken, PlacementFeedback, PlacementSignature,
    SelectedExecutionCandidate,
};
use runmat_time::duration_ns_saturating;

mod admission;
mod host_sink;
mod publication;
pub(crate) use admission::combine_host_scratch;
pub(crate) use admission::{
    select_exact_host_upload, select_exact_host_upload_with_readbacks, ExactHostUpload,
};
pub(crate) use host_sink::{
    select_exact_host_sink, select_exact_host_sink_with_readbacks, ExactHostSink, HostSinkReceipt,
};
use publication::{AdmittedHostUploadOutputs, SelectedUploadPublication, UploadPublication};

#[cfg(test)]
#[path = "host_presentation/tests.rs"]
pub(crate) mod tests;

/// A borrowed provider recorder must surround the actual upload/restore call.
/// Dropping this owner before `finish` abandons the sample and releases the
/// selection's exploratory probe without turning cancellation into failure.
pub(crate) struct HostPresentationReceipt {
    runtime: RuntimeContext,
    selected: RuntimeWholeWorkSelection,
    receipt: ProviderOperationReceipt,
    upload_publication: UploadPublication,
}

impl HostPresentationReceipt {
    pub(crate) fn new(
        runtime: RuntimeContext,
        selected: RuntimeWholeWorkSelection,
        receipt: ProviderOperationReceipt,
    ) -> Self {
        Self {
            runtime,
            selected,
            receipt,
            upload_publication: UploadPublication::NotApplicable,
        }
    }

    fn with_admitted_upload_outputs(mut self, admitted: AdmittedHostUploadOutputs) -> Self {
        self.upload_publication = UploadPublication::Selected {
            admitted,
            state: SelectedUploadPublication::Pending,
        };
        self
    }

    /// Validate the realized output against the same typed owner and exact
    /// physical request used by selection. This is a publication check, not
    /// a substitute for the caller's semantic class/provenance rules.
    pub(crate) fn validate_published_outputs(
        &mut self,
        outputs: &[&runmat_accelerate_api::GpuTensorHandle],
    ) -> BuiltinResult<()> {
        if !self.upload_publication.validate(outputs) {
            return Err(build_runtime_error(
                "published host-upload result does not match its admitted owner, representation or physical allocation",
            )
            .with_identifier("RunMat:Placement:InvalidPublishedOutput")
            .build());
        }
        Ok(())
    }

    pub(crate) fn recorder(&self) -> ProviderOperationRecorder {
        self.receipt.recorder()
    }

    /// A provider-first conditional route may publish a host result only
    /// after the owning hook actually returned Unsupported. Keep the same
    /// receipt and probe until all attempted submissions have completed.
    pub(crate) fn finish_conditional_host_fallback(mut self) {
        self.receipt.confirm_conditional_host_fallback();
        self.finish(Ok(true));
    }

    pub(crate) fn finish_preparation_host_fallback(mut self) {
        self.receipt.confirm_preparation_host_fallback();
        self.finish(Ok(true));
    }

    /// An Unsupported hook may instead publish the admitted host result on
    /// its exact provider owner. The upload must have completed publication
    /// validation before this terminal witness is supplied.
    pub(crate) fn finish_conditional_provider_upload(mut self) {
        self.receipt.confirm_conditional_provider_upload();
        self.finish(Ok(true));
    }

    pub(crate) fn begin_conditional_provider_upload(&mut self) {
        self.receipt.begin_conditional_provider_upload();
    }

    pub(crate) fn begin_preparation_provider_upload(&mut self) {
        self.receipt.begin_preparation_provider_upload();
    }

    pub(crate) fn finish_preparation_provider_upload(mut self) {
        self.receipt.confirm_preparation_provider_upload();
        self.finish(Ok(true));
    }

    /// `published` means every requested output was validated as owned by the
    /// admitted provider and matched its admitted representation or bound.
    pub(crate) fn finish(mut self, published: Result<bool, ProviderObservationFailure>) {
        if self.runtime.is_cancelled() {
            self.receipt.fail(ProviderObservationFailure::Cancelled);
        } else if self.upload_publication.rejected() {
            self.receipt
                .fail(ProviderObservationFailure::PublicationFailed);
        } else {
            match published {
                Err(failure) => self.receipt.fail(failure),
                Ok(false) => self
                    .receipt
                    .fail(ProviderObservationFailure::SharedRuntimeFallback),
                Ok(true) => {
                    if !self.upload_publication.can_publish() {
                        self.receipt
                            .fail(ProviderObservationFailure::PublicationFailed);
                    } else if self.receipt.publish().is_err() {
                        self.receipt
                            .fail(ProviderObservationFailure::PublicationFailed);
                    }
                }
            }
        }
        let mut deferred = DeferredPresentationFeedback {
            receipt: self.receipt,
            signature: self.selected.decision.signature.clone(),
            selected: self.selected.decision.selected.clone(),
            // Exploration remains reserved until completion evidence or a
            // terminal non-sample; enqueue/publication is not completion.
            _probe: self.selected.probe.take(),
            cancellation: self.runtime.cancellation(),
            terminal_without_feedback: false,
        };
        if let Some(service) = self.runtime.service_ports().placement() {
            if let Some(ready) = deferred.take_ready() {
                let _ = service.observe(ready);
            } else if !deferred.terminal_without_feedback()
                && service.defer_feedback(Box::new(deferred)).is_err()
            {
                tracing::debug!(target: "runmat_placement", "provider presentation receipt could not be retained");
            }
        }
    }
}

struct DeferredPresentationFeedback {
    receipt: ProviderOperationReceipt,
    signature: PlacementSignature,
    selected: SelectedExecutionCandidate,
    _probe: Option<Box<dyn RuntimeWholeWorkProbeLease>>,
    cancellation: CancellationToken,
    terminal_without_feedback: bool,
}

impl RuntimePlacementFeedbackSource for DeferredPresentationFeedback {
    fn signature(&self) -> &PlacementSignature {
        &self.signature
    }

    fn take_ready(&mut self) -> Option<PlacementFeedback> {
        if self.cancellation.is_cancelled() {
            self.receipt.fail(ProviderObservationFailure::Cancelled);
        }
        let observation = self.receipt.take_completed()?;
        let (elapsed, succeeded) = match observation.status {
            ProviderObservationStatus::Completed { total_elapsed, .. } => {
                (duration_ns_saturating(total_elapsed), true)
            }
            ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed) => {
                (0, false)
            }
            ProviderObservationStatus::Failed(_) => {
                self.terminal_without_feedback = true;
                return None;
            }
            ProviderObservationStatus::Pending => return None,
        };
        Some(PlacementFeedback {
            signature: self.signature.clone(),
            candidate: self.selected.candidate.clone(),
            kind: self.selected.kind,
            total_elapsed_ns: elapsed,
            succeeded,
        })
    }

    fn terminal_without_feedback(&self) -> bool {
        self.terminal_without_feedback
    }
}
