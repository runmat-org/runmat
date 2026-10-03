//! Typed provider-operation inputs and the selected completion owner.

use super::super::host_presentation::HostPresentationReceipt;
use runmat_accelerate_api::observation::{ProviderObservationFailure, ProviderOperationRecorder};
use runmat_accelerate_api::{
    provider_for_handle, same_provider_instance, AccelProvider, GpuTensorHandle,
    ProviderFeasibilityQuery,
};
use runmat_execution::{Digest, ProviderInputStageDemands, ResourceByteBound};

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

pub(crate) struct SelectedProviderOperation {
    pub(super) provider: &'static dyn AccelProvider,
    pub(super) receipt: HostPresentationReceipt,
    pub(super) direct_output: DirectProviderPublication,
    pub(super) conditional_upload: ConditionalUploadPublication,
}

pub(super) enum DirectProviderPublication {
    NoOutput,
    Pending {
        outputs: usize,
        requested_bytes: u64,
    },
    Validated {
        outputs: usize,
        requested_bytes: u64,
    },
    Rejected,
}

impl DirectProviderPublication {
    pub(super) fn new(outputs: usize, requested_bytes: Option<u64>) -> Option<Self> {
        if outputs == 0 {
            return Some(Self::NoOutput);
        }
        requested_bytes.map(|requested_bytes| Self::Pending {
            outputs,
            requested_bytes,
        })
    }

    fn validate(
        &mut self,
        provider: &'static dyn AccelProvider,
        handles: &[&GpuTensorHandle],
    ) -> bool {
        let (outputs, requested_bytes) = match *self {
            Self::Pending {
                outputs,
                requested_bytes,
            }
            | Self::Validated {
                outputs,
                requested_bytes,
            } => (outputs, requested_bytes),
            Self::NoOutput | Self::Rejected => {
                *self = Self::Rejected;
                return false;
            }
        };
        let actual_bytes = handles.iter().try_fold(0_u64, |total, handle| {
            conditional_output_owner_valid(provider, handle)
                .then(|| total.checked_add(handle.physical().requested_bytes()))
                .flatten()
        });
        if handles.len() == outputs && actual_bytes == Some(requested_bytes) {
            *self = Self::Validated {
                outputs,
                requested_bytes,
            };
            true
        } else {
            *self = Self::Rejected;
            false
        }
    }

    fn can_publish(&self) -> bool {
        matches!(self, Self::NoOutput | Self::Validated { .. })
    }
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
mod tests {
    use super::{
        conditional_output_owner_valid, ConditionalUploadPublication, DirectProviderPublication,
    };
    use crate::builtins::common::gpu_helpers;
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{
        same_provider_instance, AccelProvider, GpuTensorView, ThreadProviderGuard,
    };
    use runmat_value::Tensor;

    #[test]
    fn conditional_physical_publication_needs_exact_witness_and_rejection_is_terminal() {
        let mut pending = ConditionalUploadPublication::new(Some(8));
        assert!(!pending.validated());
        assert!(!pending.conditional_attempted());
        pending.begin();
        assert!(pending.conditional_attempted());
        assert!(pending.validate(8, true));
        assert!(pending.validated());
        assert!(!pending.validate(32, true));
        assert!(!pending.validate(8, true));
        assert!(!pending.validated());
        assert!(!ConditionalUploadPublication::new(None).validate(8, true));
        let mut invalid_owner = ConditionalUploadPublication::new(Some(8));
        invalid_owner.begin();
        assert!(!invalid_owner.validate(8, false));
        let mut premature = ConditionalUploadPublication::new(Some(8));
        assert!(!premature.validate(8, true));
        assert!(premature.conditional_attempted());
    }

    #[test]
    fn same_device_foreign_facade_is_not_a_conditional_output_owner() {
        let owner: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
        let foreign: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
        assert_eq!(owner.device_id(), foreign.device_id());
        assert!(!same_provider_instance(owner, foreign));
        let _selected_owner = ThreadProviderGuard::set(Some(owner));
        let output = gpu_helpers::upload_tensor(
            owner,
            &Tensor::new(vec![1.0], vec![1, 1]).expect("one-element result"),
        )
        .expect("owner result");
        assert!(!conditional_output_owner_valid(foreign, &output));
        let mut selected =
            ConditionalUploadPublication::new(Some(output.physical().requested_bytes()));
        selected.begin();
        assert!(!selected.validate(
            output.physical().requested_bytes(),
            conditional_output_owner_valid(foreign, &output),
        ));
        assert!(!selected.validated());
        assert!(output.allocation_lease().validate_live().is_ok());
    }

    #[test]
    fn direct_publication_uses_selected_physical_extent_and_exact_owner() {
        let owner: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
        let foreign: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
        let _selected_owner = ThreadProviderGuard::set(Some(owner));
        let exact = gpu_helpers::upload_tensor(
            owner,
            &Tensor::new(vec![1.0], vec![1, 1]).expect("exact tensor"),
        )
        .expect("exact output");
        let retained_caller = gpu_helpers::upload_tensor(
            owner,
            &Tensor::new(vec![2.0; 4], vec![2, 2]).expect("large tensor"),
        )
        .expect("caller-owned allocation");
        let larger_view = retained_caller
            .with_view(GpuTensorView::contiguous(vec![1, 1]))
            .expect("same view, larger backing allocation");
        let expected = exact.physical().requested_bytes();
        assert!(DirectProviderPublication::new(0, None)
            .expect("no-output form")
            .can_publish());
        assert!(DirectProviderPublication::new(1, None).is_none());
        let mut admitted = DirectProviderPublication::new(1, Some(expected)).expect("bound");
        assert!(!admitted.can_publish());
        assert!(admitted.validate(owner, &[&exact]));
        assert!(admitted.can_publish());
        let mut oversized = DirectProviderPublication::new(1, Some(expected)).expect("bound");
        assert!(!oversized.validate(owner, &[&larger_view]));
        assert!(!oversized.can_publish());
        assert!(!oversized.validate(owner, &[&exact]));
        let mut wrong_owner = DirectProviderPublication::new(1, Some(expected)).expect("bound");
        assert!(!wrong_owner.validate(foreign, &[&exact]));
        assert!(!wrong_owner.can_publish());
        let error = crate::build_runtime_error("invalid provider result")
            .with_identifier("RunMat:test")
            .build();
        let _ = gpu_helpers::reject_gpu_output(larger_view, error);
        assert!(retained_caller.allocation_lease().validate_live().is_ok());
        assert!(exact.allocation_lease().validate_live().is_ok());
    }
}
