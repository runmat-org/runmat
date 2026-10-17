//! Typed provider-operation inputs and the selected completion owner.

use super::super::host_presentation::HostPresentationReceipt;
use runmat_accelerate_api::observation::{ProviderObservationFailure, ProviderOperationRecorder};
use runmat_accelerate_api::{
    provider_for_handle, same_provider_instance, AccelProvider, GpuTensorHandle,
    ProviderFeasibilityQuery, ProviderProducedNumericReadbackEvidence, ProviderRepresentation,
};
use runmat_execution::WholeWorkCandidate;
use runmat_execution::{Digest, ProviderInputStageDemands, ResourceByteBound};

mod direct_publication;
pub(super) use direct_publication::{DirectProviderOutputContract, DirectProviderPublication};

/// An original resident operand whose native readback occurs during this
/// provider-first route. `prior_created_bytes` is operation-created provider
/// storage still live during this stage, excluding authenticated borrowed
/// allocations already in that owner's resource snapshot.
pub(crate) struct ExactProviderReadback<'a> {
    pub(crate) source: &'a GpuTensorHandle,
    pub(crate) prior_created_bytes: ResourceByteBound,
    pub(crate) phase: ExactProviderReadbackPhase,
}

#[derive(Clone, Copy, Eq, PartialEq, serde::Serialize)]
pub(crate) enum ExactProviderReadbackPhase {
    Preparation,
    Conditional,
}

/// A provider-created output, not an original input, is read back after the
/// hook. The public host result and conversion workspace are distinct from
/// the provider's native physical result and mapped staging.
#[derive(Clone, Copy)]
pub(crate) struct ExactProducedProviderReadback {
    pub(crate) output_index: usize,
    pub(crate) public_host_bytes: ResourceByteBound,
    pub(crate) conversion_host_workspace: ResourceByteBound,
}

pub(super) struct PreparedProducedReadback {
    pub(super) output: ProviderRepresentation,
    pub(super) evidence: ProviderProducedNumericReadbackEvidence,
    pub(super) public_host_bytes: ResourceByteBound,
}

/// Physical continuation admitted only for an Unsupported provider hook that
/// computes on host and restores to this same original provider owner.
#[derive(Clone, Copy)]
pub(crate) struct ExactConditionalProviderUpload {
    pub(crate) provider_output_requested_bytes: u64,
    pub(crate) transfer_bytes: Option<u64>,
    pub(crate) host_result_bytes: ResourceByteBound,
    pub(crate) host_scratch_bytes: ResourceByteBound,
    pub(crate) provider_scratch_bytes: ResourceByteBound,
    pub(crate) queue_slots: u32,
}

pub(crate) struct ExactProviderOperation<'a> {
    pub(crate) provider: &'static dyn AccelProvider,
    pub(crate) query: &'a ProviderFeasibilityQuery,
    pub(crate) semantic_operation: &'static str,
    pub(crate) variant: &'static str,
    /// Typed controls and original operand provenance. Never a textual
    /// substitute for the live handles that the domain owner authenticates.
    pub(crate) domain_facts: Digest,
    /// Optional request-owner physical calibration class. When present, this
    /// replaces per-invocation controls in placement feedback only; the full
    /// query and live owner still govern selection and execution validation.
    pub(crate) calibration_facts: Option<ProviderPhysicalCalibration>,
    /// Peak operation-owned host workspace, including any work before the
    /// provider hook but excluding provider internal host workspace.
    pub(crate) host_scratch: ResourceByteBound,
    /// Provider-hook internal host workspace, certified by its typed physical
    /// route contract. This remains separate from operation-owned preparation.
    pub(crate) provider_host_scratch: ResourceByteBound,
    pub(crate) host_cpu_millicores: u32,
    /// Physical bytes of authenticated borrowed inputs already represented
    /// in the provider's live-allocation resource snapshot.
    pub(crate) already_live_bytes: u64,
    pub(crate) readbacks: &'a [ExactProviderReadback<'a>],
    pub(crate) produced_readback: Option<ExactProducedProviderReadback>,
    /// Provider-owned internal stages not represented by original readbacks
    /// (for example a second readback of an adapted, newly uploaded kernel).
    /// The typed provider operation contract must prove these stage peaks.
    pub(crate) internal_input_stages: Option<ProviderInputStageDemands>,
    pub(crate) output_queue_slots: Option<u32>,
    /// The public form permits a host result only after this provider hook
    /// returns Unsupported. This is one conditional route, not an equivalent
    /// host alternative offered to placement policy.
    pub(crate) conditional_host_fallback: Option<ResourceByteBound>,
    /// Fallback-only host workspace; includes its final host result when that
    /// result is published on host, unlike provider upload's exclusive scratch.
    pub(crate) conditional_host_scratch: ResourceByteBound,
    pub(crate) conditional_provider_upload: Option<ExactConditionalProviderUpload>,
    pub(crate) provider_host_mediated: bool,
}

/// Placement feedback groups physically equivalent requests. This is not a
/// stream reservation, executable grant, or replacement for `domain_facts`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProviderPhysicalCalibration(Digest);

impl ProviderPhysicalCalibration {
    pub(crate) fn from_typed(facts: &impl serde::Serialize) -> Self {
        Self(Digest::sha256_serialized(facts).expect("typed physical calibration serializes"))
    }

    pub(crate) fn digest(self) -> Digest {
        self.0
    }
}

/// An independently legal host-compute/provider-upload alternative. Its
/// physical upload query and output contract are kept with the candidate so
/// the selected receipt validates the actual publication, not just a cost.
pub(crate) struct ExactHostUploadAlternative<'a> {
    pub(crate) candidate: WholeWorkCandidate,
    pub(crate) query: &'a ProviderFeasibilityQuery,
    pub(crate) output: ProviderRepresentation,
    pub(crate) requested_bytes: u64,
}

pub(crate) enum ExactProviderOrHostSelection {
    Provider(SelectedProviderOperation),
    HostUpload(HostPresentationReceipt),
}

pub(crate) struct SelectedProviderOperation {
    pub(super) provider: &'static dyn AccelProvider,
    pub(super) receipt: HostPresentationReceipt,
    pub(super) direct_output: DirectProviderPublication,
    pub(super) conditional_upload: ConditionalUploadPublication,
    pub(super) produced_readback: Option<PreparedProducedReadback>,
}

pub(super) enum ConditionalUploadPublication {
    NotApplicable,
    Pending(u64),
    Active(u64),
    Validated(u64),
    Rejected,
}

impl ConditionalUploadPublication {
    pub(super) fn new(bytes: Option<u64>) -> Self {
        bytes.map_or(Self::NotApplicable, Self::Pending)
    }

    fn begin(&mut self) {
        *self = match *self {
            Self::Pending(bytes) => Self::Active(bytes),
            _ => Self::Rejected,
        };
    }

    fn validate(&mut self, bytes: u64, owner_valid: bool) -> bool {
        let expected = match *self {
            Self::Active(expected) | Self::Validated(expected) => expected,
            Self::NotApplicable | Self::Pending(_) | Self::Rejected => {
                *self = Self::Rejected;
                return false;
            }
        };
        if owner_valid && bytes == expected {
            *self = Self::Validated(expected);
            true
        } else {
            *self = Self::Rejected;
            false
        }
    }

    fn validated(&self) -> bool {
        matches!(self, Self::Validated(_))
    }

    fn conditional_attempted(&self) -> bool {
        matches!(self, Self::Active(_) | Self::Validated(_) | Self::Rejected)
    }
}

fn conditional_output_owner_valid(
    provider: &'static dyn AccelProvider,
    output: &GpuTensorHandle,
) -> bool {
    output.device_id() == provider.device_id()
        && output.allocation_lease().validate_live().is_ok()
        && provider_for_handle(output).is_some_and(|owner| same_provider_instance(owner, provider))
}

impl SelectedProviderOperation {
    pub(crate) fn validate_direct_provider_outputs(
        &mut self,
        outputs: &[&GpuTensorHandle],
    ) -> bool {
        self.direct_output.validate(self.provider, outputs)
    }

    pub(crate) fn validate_reused_direct_provider_outputs(
        &mut self,
        outputs: &[&GpuTensorHandle],
        borrowed: &[&GpuTensorHandle],
    ) -> bool {
        self.direct_output
            .validate_reused(self.provider, outputs, borrowed)
    }

    pub(crate) fn admits_produced_readback(&self, output: &GpuTensorHandle) -> bool {
        let Some(planned) = self.produced_readback.as_ref() else {
            return false;
        };
        if !conditional_output_owner_valid(self.provider, output)
            || output.physical().requested_bytes() != planned.evidence.source_requested_bytes
        {
            return false;
        }
        let Some(actual) = ProviderFeasibilityQuery::exact_numeric_download(output) else {
            return false;
        };
        actual.inputs.first() == Some(&planned.output)
            && self
                .provider
                .query_numeric_download_feasibility(output, &actual)
                .is_supported()
            && self.provider.download_numeric_host_workspace_bound(output)
                == planned.evidence.host_workspace_bytes
            && self
                .provider
                .download_numeric_provider_scratch_bound(output)
                == planned.evidence.provider_scratch_bytes
            && self.provider.download_numeric_queue_slots(output) == planned.evidence.queue_slots
    }
    pub(crate) fn validate_direct_provider_output(&mut self, output: &GpuTensorHandle) -> bool {
        self.direct_output.validate(self.provider, &[output])
    }

    pub(crate) fn begin_conditional_provider_upload(&mut self) {
        self.conditional_upload.begin();
        self.receipt.begin_conditional_provider_upload();
    }

    pub(crate) fn begin_preparation_provider_upload(&mut self) {
        self.conditional_upload.begin();
        self.receipt.begin_preparation_provider_upload();
    }

    pub(crate) fn recorder(&self) -> ProviderOperationRecorder {
        self.receipt.recorder()
    }

    pub(crate) fn provider(&self) -> &'static dyn AccelProvider {
        self.provider
    }

    pub(crate) fn admits_conditional_provider_upload(&mut self, output: &GpuTensorHandle) -> bool {
        self.conditional_upload.validate(
            output.physical().requested_bytes(),
            conditional_output_owner_valid(self.provider, output),
        )
    }

    pub(crate) fn finish(self, published: Result<bool, ProviderObservationFailure>) {
        if matches!(published, Ok(true))
            && (self.conditional_upload.conditional_attempted()
                || !self.direct_output.can_publish())
        {
            self.receipt
                .finish(Err(ProviderObservationFailure::PublicationFailed));
        } else {
            self.receipt.finish(published);
        }
    }

    pub(crate) fn finish_conditional_host_fallback(self) {
        if self.conditional_upload.conditional_attempted() {
            self.receipt
                .finish(Err(ProviderObservationFailure::PublicationFailed));
        } else {
            self.receipt.finish_conditional_host_fallback();
        }
    }

    pub(crate) fn finish_preparation_host_fallback(self) {
        if self.conditional_upload.conditional_attempted() {
            self.receipt
                .finish(Err(ProviderObservationFailure::PublicationFailed));
        } else {
            self.receipt.finish_preparation_host_fallback();
        }
    }

    pub(crate) fn finish_conditional_provider_upload(self) {
        if self.conditional_upload.validated() {
            self.receipt.finish_conditional_provider_upload();
        } else {
            self.receipt
                .finish(Err(ProviderObservationFailure::PublicationFailed));
        }
    }

    pub(crate) fn finish_preparation_provider_upload(self) {
        if self.conditional_upload.validated() {
            self.receipt.finish_preparation_provider_upload();
        } else {
            self.receipt
                .finish(Err(ProviderObservationFailure::PublicationFailed));
        }
    }
}

#[cfg(test)]
#[path = "types/tests.rs"]
mod tests;
