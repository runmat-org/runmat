//! Completion-qualified feedback for a selected route whose publication is a
//! provider operation. Typed query, resource, and output admission stay with
//! the caller; this owner never chooses or replays an execution route.

use super::{
    RuntimeContext, RuntimeManagedWholeWorkSelection, RuntimePlacementFeedbackDelivery,
    RuntimePlacementFeedbackSource, RuntimeWholeWorkSelection,
};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationFailure, ProviderObservationStatus, ProviderOperationReceipt,
    ProviderOperationRecorder,
};
use runmat_execution::profiling::{CommandProviderContribution, ProviderObservationTerminal};
use runmat_execution::resource::RetainedHostAllocationClaim;
use runmat_execution::{
    cancellation::CancellationToken, PlacementSignature, WholeWorkObservationOutcome,
};
use runmat_time::duration_ns_saturating;
use std::sync::Arc;

mod admission;
mod host_compute;
mod host_sink;
mod publication;
pub(crate) use admission::combine_host_scratch;
pub(crate) use admission::{
    select_exact_host_upload, select_exact_host_upload_with_readbacks, select_prepared_host_upload,
    selected_upload_metadata_bytes, selected_upload_metadata_claim, upload_candidate,
    ExactHostUpload, UploadStageResources,
};
pub(crate) use host_compute::{
    select_host_computed_result, ExactHostComputedResult, ExactHostComputedUpload,
    HostComputedSelection,
};
pub(crate) use host_sink::{
    select_exact_host_sink, select_exact_host_sink_with_readbacks, ExactHostSink, HostSinkReceipt,
};
pub(crate) use publication::AdmittedHostUploadOutputs;
use publication::{SelectedUploadPublication, UploadPublication};

#[cfg(test)]
#[path = "host_presentation/tests.rs"]
pub(crate) mod tests;

/// A borrowed provider recorder must surround the actual upload/restore call.
/// Dropping this owner before `finish` abandons the sample and releases the
/// selection's exploratory probe without turning cancellation into failure.
pub(crate) struct HostPresentationReceipt {
    runtime: RuntimeContext,
    selected: PresentationSelection,
    receipt: ProviderOperationReceipt,
    command_contribution: Option<CommandProviderContribution>,
    upload_publication: UploadPublication,
    // The selected request's cloned output/operation metadata may survive
    // the temporary query-preparation receipt and the invoking frame.
    publication_metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

enum PresentationSelection {
    Portable(RuntimeWholeWorkSelection),
    Managed(RuntimeManagedWholeWorkSelection),
}

/// One callback-shared, acyclic owner. The provider observation Shared keeps
/// this alive after a caller drops its completion future; this owner never
/// holds a receipt, recorder, or callback back into that Shared.
struct ManagedPresentationRetention {
    _lease: Arc<super::ManagedObservationLease>,
    // Declared after the lease so its funding is released last.
    _metadata_claim: Arc<dyn RetainedHostAllocationClaim>,
}

impl std::fmt::Debug for ManagedPresentationRetention {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ManagedPresentationRetention")
            .finish_non_exhaustive()
    }
}

impl HostPresentationReceipt {
    /// Requested Arc backing for the callback-retained lease/claim aggregate.
    /// The caller includes this in its retained metadata payload before
    /// constructing the selected operation or issuing its first effect.
    pub(crate) fn managed_retention_host_upper_bytes() -> Option<u64> {
        std::mem::size_of::<ManagedPresentationRetention>()
            .checked_add(std::mem::size_of::<usize>().checked_mul(2)?)?
            .checked_add(std::mem::align_of::<ManagedPresentationRetention>())?
            .try_into()
            .ok()
    }

    pub(crate) fn new(
        runtime: RuntimeContext,
        selected: RuntimeWholeWorkSelection,
        receipt: ProviderOperationReceipt,
    ) -> Self {
        let command_contribution = runtime
            .command_profiling()
            .and_then(|command| command.begin_provider_operation());
        Self {
            runtime,
            selected: PresentationSelection::Portable(selected),
            receipt,
            command_contribution,
            upload_publication: UploadPublication::NotApplicable,
            publication_metadata_claim: None,
        }
    }

    /// A finite selection keeps its precharged compact observation lease in
    /// callback-held Shared state. If the caller drops the completion future,
    /// no sample is delivered, but the lease and metadata funding remain
    /// through actual callback retirement.
    pub(crate) fn new_managed(
        runtime: RuntimeContext,
        selected: RuntimeManagedWholeWorkSelection,
        mut receipt: ProviderOperationReceipt,
        metadata_claim: Arc<dyn RetainedHostAllocationClaim>,
    ) -> Result<Self, ProviderObservationFailure> {
        let retention = Arc::new(ManagedPresentationRetention {
            _lease: Arc::clone(&selected.observation_lease),
            _metadata_claim: metadata_claim,
        });
        if let Err(error) = receipt.retain_operation_host_metadata(retention.clone()) {
            // The original claim must outlive destruction of the newly
            // allocated receipt even when pre-effect installation fails.
            drop(receipt);
            drop(retention);
            return Err(error);
        }
        let command_contribution = runtime
            .command_profiling()
            .and_then(|command| command.begin_provider_operation());
        Ok(Self {
            runtime,
            selected: PresentationSelection::Managed(selected),
            receipt,
            command_contribution,
            upload_publication: UploadPublication::NotApplicable,
            publication_metadata_claim: None,
        })
    }

    pub(crate) fn with_publication_metadata_claim(
        mut self,
        claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Self {
        self.publication_metadata_claim = claim;
        self
    }

    pub(crate) fn with_admitted_upload_outputs(
        mut self,
        admitted: AdmittedHostUploadOutputs,
    ) -> Self {
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
    fn settle_publication(&mut self, published: Result<bool, ProviderObservationFailure>) {
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
                    if !self.upload_publication.can_publish() || self.receipt.publish().is_err() {
                        self.receipt
                            .fail(ProviderObservationFailure::PublicationFailed);
                    }
                }
            }
        }
    }

    /// Await managed completion, or hand portable completion to its existing
    /// deferred owner. The receipt alone owns this distinction; domain
    /// executors need not duplicate its selected-authority state machine.
    pub(crate) async fn finish_selected(self, published: Result<bool, ProviderObservationFailure>) {
        if matches!(&self.selected, PresentationSelection::Managed(_)) {
            self.finish_managed(published).await;
        } else {
            self.finish(published);
        }
    }

    pub(crate) fn finish(mut self, published: Result<bool, ProviderObservationFailure>) {
        self.settle_publication(published);
        let selected = match self.selected {
            PresentationSelection::Portable(selected) => selected,
            PresentationSelection::Managed(_) => {
                unreachable!("managed receipt must await completion on its owning thread")
            }
        };
        // Release the selected output descriptors while their retained claim
        // is still here. The operation identity remains inside the provider
        // receipt and moves with the claim into deferred completion.
        self.upload_publication = UploadPublication::NotApplicable;
        let mut deferred = DeferredPresentationFeedback {
            receipt: self.receipt,
            command_contribution: self.command_contribution.take(),
            signature: selected.decision.signature.clone(),
            // Exploration remains reserved until completion evidence or a
            // terminal non-sample; enqueue/publication is not completion.
            selected: Some(selected),
            cancellation: self.runtime.cancellation(),
            terminal_without_feedback: false,
            _publication_metadata_claim: self.publication_metadata_claim.take(),
        };
        if let Some(service) = self.runtime.service_ports().placement() {
            if let Some(ready) = deferred.take_ready() {
                let _ = service.observe_deferred_feedback(ready);
            } else if !deferred.terminal_without_feedback()
                && service.defer_feedback(Box::new(deferred)).is_err()
            {
                tracing::debug!(target: "runmat_placement", "provider presentation receipt could not be retained");
            }
        }
    }

    /// Finite provider work cannot leave an unpriced boxed feedback source
    /// behind. Await the provider's real completion witness on this owner
    /// thread, retaining the managed selection lease and metadata claim in
    /// callback-held Shared state even if this waiter is dropped. A failure
    /// status alone is not physical retirement, so wait for issued resource
    /// tickets to quiesce before consuming timing-only evidence.
    pub(crate) async fn finish_managed(
        mut self,
        published: Result<bool, ProviderObservationFailure>,
    ) {
        self.settle_publication(published);
        self.upload_publication = UploadPublication::NotApplicable;
        let _physical_result =
            std::future::poll_fn(|cx| self.receipt.poll_resource_quiescence(cx)).await;
        let observation = std::future::poll_fn(|cx| self.receipt.poll_completed_timing(cx)).await;
        let Some(observation) = observation else {
            if let Some(contribution) = self.command_contribution.take() {
                contribution.complete(ProviderObservationTerminal::Missing);
            }
            return;
        };
        let selected = match &self.selected {
            PresentationSelection::Managed(selected) => selected,
            PresentationSelection::Portable(_) => {
                unreachable!("portable receipt uses deferred completion")
            }
        };
        if self.runtime.is_cancelled() || !selected.observation_is_current() {
            if let Some(contribution) = self.command_contribution.take() {
                contribution.complete(ProviderObservationTerminal::Missing);
            }
            return;
        }
        if let Some(contribution) = self.command_contribution.take() {
            contribution.complete(observation_terminal(&observation));
        }
        let outcome = match observation.status {
            ProviderObservationStatus::Completed { .. } => WholeWorkObservationOutcome::Success,
            ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed) => {
                WholeWorkObservationOutcome::OperationalFailure
            }
            ProviderObservationStatus::Failed(_) | ProviderObservationStatus::Pending => return,
        };
        let elapsed_ns = match observation.status {
            ProviderObservationStatus::Completed { total_elapsed, .. } => {
                duration_ns_saturating(total_elapsed)
            }
            ProviderObservationStatus::Failed(_) => 0,
            ProviderObservationStatus::Pending => return,
        };
        if let Some(service) = self.runtime.service_ports().placement() {
            let _ = service.observe_managed_whole_work(selected, elapsed_ns, outcome);
        }
    }
}

fn observation_terminal(
    observation: &runmat_accelerate_api::observation::ProviderOperationObservation,
) -> ProviderObservationTerminal {
    match observation.status {
        ProviderObservationStatus::Completed { .. } => ProviderObservationTerminal::Completed {
            gpu_execution: observation.verified_wgpu_compute_elapsed,
            kernel_count: observation.verified_wgpu_dispatches,
        },
        ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed) => {
            ProviderObservationTerminal::Failed
        }
        ProviderObservationStatus::Failed(_) | ProviderObservationStatus::Pending => {
            ProviderObservationTerminal::Missing
        }
    }
}

struct DeferredPresentationFeedback {
    receipt: ProviderOperationReceipt,
    command_contribution: Option<CommandProviderContribution>,
    signature: PlacementSignature,
    selected: Option<RuntimeWholeWorkSelection>,
    cancellation: CancellationToken,
    terminal_without_feedback: bool,
    // Must drop after `receipt`, which retains the selected operation name.
    _publication_metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

impl RuntimePlacementFeedbackSource for DeferredPresentationFeedback {
    fn signature(&self) -> &PlacementSignature {
        &self.signature
    }

    fn take_ready(&mut self) -> Option<RuntimePlacementFeedbackDelivery> {
        if self
            .selected
            .as_ref()
            .is_none_or(|selected| !selected.observation_is_current())
        {
            if let Some(contribution) = self.command_contribution.take() {
                contribution.complete(ProviderObservationTerminal::Missing);
            }
            self.terminal_without_feedback = true;
            return None;
        }
        if self.cancellation.is_cancelled() {
            self.receipt.fail(ProviderObservationFailure::Cancelled);
        }
        let observation = self.receipt.take_completed()?;
        if let Some(contribution) = self.command_contribution.take() {
            contribution.complete(observation_terminal(&observation));
        }
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
        Some(RuntimePlacementFeedbackDelivery::WholeWork {
            selected: self.selected.take()?,
            elapsed_ns: elapsed,
            outcome: if succeeded {
                WholeWorkObservationOutcome::Success
            } else {
                WholeWorkObservationOutcome::OperationalFailure
            },
        })
    }

    fn terminal_without_feedback(&self) -> bool {
        self.terminal_without_feedback
    }
}
