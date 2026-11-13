//! Fresh and explicitly reused provider-output publication receipts.

use super::conditional_output_owner_valid;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderFeasibilityQuery};

#[derive(Clone, Copy)]
pub(in crate::context::provider_operation) enum DirectProviderOutputContract {
    Fresh,
    ReusedInputs,
}

pub(in crate::context::provider_operation) enum DirectProviderPublication {
    NoOutput,
    Pending {
        outputs: usize,
        requested_bytes: u64,
    },
    ReusedPending {
        outputs: usize,
        already_live_bytes: u64,
    },
    Validated {
        outputs: usize,
        requested_bytes: u64,
    },
    ReusedValidated {
        outputs: usize,
        already_live_bytes: u64,
    },
    Rejected,
}

impl DirectProviderPublication {
    pub(in crate::context::provider_operation) fn from_query(
        query: &ProviderFeasibilityQuery,
        requested_bytes: Option<u64>,
        already_live_bytes: u64,
        contract: DirectProviderOutputContract,
    ) -> Option<Self> {
        match contract {
            DirectProviderOutputContract::Fresh => {
                Self::fresh(query.outputs.len(), requested_bytes)
            }
            DirectProviderOutputContract::ReusedInputs => (!query.outputs.is_empty()
                && query.inputs == query.outputs
                && requested_bytes == Some(0))
            .then_some(Self::ReusedPending {
                outputs: query.outputs.len(),
                already_live_bytes,
            }),
        }
    }

    fn fresh(outputs: usize, requested_bytes: Option<u64>) -> Option<Self> {
        if outputs == 0 {
            return Some(Self::NoOutput);
        }
        requested_bytes.map(|requested_bytes| Self::Pending {
            outputs,
            requested_bytes,
        })
    }

    pub(super) fn validate(
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
            Self::NoOutput
            | Self::ReusedPending { .. }
            | Self::ReusedValidated { .. }
            | Self::Rejected => {
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

    pub(super) fn validate_reused(
        &mut self,
        provider: &'static dyn AccelProvider,
        handles: &[&GpuTensorHandle],
        borrowed: &[&GpuTensorHandle],
    ) -> bool {
        let (outputs, already_live_bytes) = match *self {
            Self::ReusedPending {
                outputs,
                already_live_bytes,
            }
            | Self::ReusedValidated {
                outputs,
                already_live_bytes,
            } => (outputs, already_live_bytes),
            Self::NoOutput | Self::Pending { .. } | Self::Validated { .. } | Self::Rejected => {
                *self = Self::Rejected;
                return false;
            }
        };
        let actual_live_bytes = borrowed.iter().try_fold(0_u64, |total, input| {
            conditional_output_owner_valid(provider, input)
                .then(|| total.checked_add(input.physical().requested_bytes()))
                .flatten()
        });
        let exact_reuse = handles.iter().zip(borrowed).all(|(output, input)| {
            conditional_output_owner_valid(provider, output)
                && output == input
                && output.view_identity() == input.view_identity()
        });
        if handles.len() == outputs
            && borrowed.len() == outputs
            && exact_reuse
            && actual_live_bytes == Some(already_live_bytes)
        {
            *self = Self::ReusedValidated {
                outputs,
                already_live_bytes,
            };
            true
        } else {
            *self = Self::Rejected;
            false
        }
    }

    pub(super) fn can_publish(&self) -> bool {
        matches!(
            self,
            Self::NoOutput | Self::Validated { .. } | Self::ReusedValidated { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{DirectProviderOutputContract, DirectProviderPublication};
    use crate::context::physical_test_support::OwnedPhysicalProvider;
    use runmat_accelerate_api::{
        ProviderElementType, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
        ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
        ProviderWorkload, ThreadProviderGuard,
    };

    #[test]
    fn direct_publication_uses_selected_physical_extent_and_exact_owner() {
        let owner: &'static OwnedPhysicalProvider =
            Box::leak(Box::new(OwnedPhysicalProvider::new()));
        let foreign: &'static OwnedPhysicalProvider =
            Box::leak(Box::new(OwnedPhysicalProvider::new()));
        let _selected_owner = ThreadProviderGuard::set(Some(owner));
        let exact = owner.allocate_f64(&[1, 1]);
        let (retained_caller, larger_view) = owner.allocate_padded_f64_view(&[2, 2], &[1, 1]);
        assert_eq!(larger_view.shape(), exact.handle.shape());
        assert_eq!(
            larger_view.physical().element_type(),
            exact.handle.physical().element_type()
        );
        let expected = exact.handle.physical().requested_bytes();
        assert!(larger_view.physical().requested_bytes() > expected);
        assert!(DirectProviderPublication::fresh(0, None)
            .expect("no-output form")
            .can_publish());
        assert!(DirectProviderPublication::fresh(1, None).is_none());
        let mut admitted = DirectProviderPublication::fresh(1, Some(expected)).expect("bound");
        assert!(!admitted.can_publish());
        assert!(admitted.validate(owner, &[&exact.handle]));
        assert!(admitted.can_publish());
        let mut oversized = DirectProviderPublication::fresh(1, Some(expected)).expect("bound");
        assert!(!oversized.validate(owner, &[&larger_view]));
        assert!(!oversized.can_publish());
        assert!(!oversized.validate(owner, &[&exact.handle]));
        let mut wrong_owner = DirectProviderPublication::fresh(1, Some(expected)).expect("bound");
        assert!(!wrong_owner.validate(foreign, &[&exact.handle]));
        assert!(!wrong_owner.can_publish());
        let error = crate::build_runtime_error("invalid provider result")
            .with_identifier("RunMat:test")
            .build();
        let _ = crate::builtins::common::gpu_helpers::reject_gpu_output(larger_view, error);
        assert!(retained_caller
            .handle
            .allocation_lease()
            .validate_live()
            .is_ok());
        assert!(exact.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(retained_caller.probe.retirements(), 0);
        assert_eq!(exact.probe.retirements(), 0);
    }

    #[test]
    fn reused_publication_requires_explicit_contract_and_exact_admitted_live_views() {
        let owner: &'static OwnedPhysicalProvider =
            Box::leak(Box::new(OwnedPhysicalProvider::new()));
        let foreign: &'static OwnedPhysicalProvider =
            Box::leak(Box::new(OwnedPhysicalProvider::new()));
        let _selected_owner = ThreadProviderGuard::set(Some(owner));
        let output = owner.allocate_f64(&[1, 1]);
        let accepted = owner.allocate_f64(&[1, 1]);
        let representation = ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 1],
            residency: ProviderResidency::Device,
        };
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("test.reused-publication"),
            family: ProviderOperationFamily::Library,
            inputs: vec![representation.clone(), representation.clone()],
            outputs: vec![representation.clone(), representation],
            workload: ProviderWorkload::default(),
        };
        let live_bytes = output
            .handle
            .physical()
            .requested_bytes()
            .checked_add(accepted.handle.physical().requested_bytes())
            .expect("fixture extent");
        let returned_output = output.handle.clone();
        let returned_accepted = accepted.handle.clone();

        let mut fresh = DirectProviderPublication::from_query(
            &query,
            Some(0),
            live_bytes,
            DirectProviderOutputContract::Fresh,
        )
        .expect("fresh zero-byte receipt remains distinct");
        assert!(!fresh.validate_reused(
            owner,
            &[&returned_output, &returned_accepted],
            &[&output.handle, &accepted.handle],
        ));

        let mut admitted = DirectProviderPublication::from_query(
            &query,
            Some(0),
            live_bytes,
            DirectProviderOutputContract::ReusedInputs,
        )
        .expect("explicit reuse");
        assert!(!admitted.can_publish());
        assert!(admitted.validate_reused(
            owner,
            &[&returned_output, &returned_accepted],
            &[&output.handle, &accepted.handle],
        ));
        assert!(admitted.can_publish());

        let wrong_view = owner.allocate_f64(&[1, 1]);
        let mut mismatched = DirectProviderPublication::from_query(
            &query,
            Some(0),
            live_bytes,
            DirectProviderOutputContract::ReusedInputs,
        )
        .expect("explicit reuse");
        assert!(!mismatched.validate_reused(
            owner,
            &[&wrong_view.handle, &returned_accepted],
            &[&output.handle, &accepted.handle],
        ));
        assert!(!mismatched.can_publish());

        let derived_view = output
            .handle
            .with_view(runmat_accelerate_api::GpuTensorView::contiguous(vec![1, 1]))
            .expect("distinct view record on the same allocation");
        assert_eq!(derived_view, output.handle);
        assert_ne!(derived_view.view_identity(), output.handle.view_identity());
        let mut new_metadata = DirectProviderPublication::from_query(
            &query,
            Some(0),
            live_bytes,
            DirectProviderOutputContract::ReusedInputs,
        )
        .expect("explicit reuse");
        assert!(!new_metadata.validate_reused(
            owner,
            &[&derived_view, &returned_accepted],
            &[&output.handle, &accepted.handle],
        ));
        assert!(!new_metadata.can_publish());

        let mut wrong_owner = DirectProviderPublication::from_query(
            &query,
            Some(0),
            live_bytes,
            DirectProviderOutputContract::ReusedInputs,
        )
        .expect("explicit reuse");
        assert!(!wrong_owner.validate_reused(
            foreign,
            &[&returned_output, &returned_accepted],
            &[&output.handle, &accepted.handle],
        ));
        assert!(!wrong_owner.can_publish());
    }
}
