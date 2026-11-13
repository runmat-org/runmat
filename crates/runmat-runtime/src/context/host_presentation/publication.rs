//! The selected upload's typed output and provider-owned physical evidence
//! remain attached to the receipt until the realized result is published.

use runmat_accelerate_api::{
    provider_for_handle, same_provider_instance, AccelProvider, GpuTensorHandle, GpuTensorLayout,
    GpuTensorStorage, ProviderElementType, ProviderLayout, ProviderPrecision,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};

pub(crate) struct AdmittedHostUploadOutputs {
    pub(crate) provider: &'static dyn AccelProvider,
    pub(crate) outputs: Vec<ProviderRepresentation>,
    /// A provider can certify an exact physical one-output route. Absence is
    /// unavailable evidence, not an inferred logical byte count.
    pub(crate) one_output_requested_bytes: Option<u64>,
}

pub(super) enum UploadPublication {
    NotApplicable,
    RejectedNoAdmission,
    Selected {
        admitted: AdmittedHostUploadOutputs,
        state: SelectedUploadPublication,
    },
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum SelectedUploadPublication {
    Pending,
    Validated,
    Rejected,
}

impl UploadPublication {
    pub(super) fn validate(&mut self, handles: &[&GpuTensorHandle]) -> bool {
        let (admitted, state) = match self {
            Self::Selected { admitted, state } => (admitted, state),
            Self::NotApplicable => {
                *self = Self::RejectedNoAdmission;
                return false;
            }
            Self::RejectedNoAdmission => return false,
        };
        if *state == SelectedUploadPublication::Rejected {
            return false;
        }
        if admitted.admits(handles) {
            *state = SelectedUploadPublication::Validated;
            true
        } else {
            *state = SelectedUploadPublication::Rejected;
            false
        }
    }

    pub(super) fn rejected(&self) -> bool {
        matches!(self, Self::RejectedNoAdmission)
            || matches!(
                self,
                Self::Selected {
                    state: SelectedUploadPublication::Rejected,
                    ..
                }
            )
    }

    pub(super) fn can_publish(&self) -> bool {
        matches!(self, Self::NotApplicable)
            || matches!(
                self,
                Self::Selected {
                    state: SelectedUploadPublication::Validated,
                    ..
                }
            )
    }
}

impl AdmittedHostUploadOutputs {
    pub(super) fn admits(&self, handles: &[&GpuTensorHandle]) -> bool {
        if handles.len() != self.outputs.len() || handles.is_empty() {
            return false;
        }
        if self.one_output_requested_bytes.is_some() && handles.len() != 1 {
            return false;
        }
        handles
            .iter()
            .zip(&self.outputs)
            .all(|(&handle, expected)| {
                handle.device_id() == self.provider.device_id()
                    && handle.allocation_lease().validate_live().is_ok()
                    && provider_for_handle(handle)
                        .is_some_and(|owner| same_provider_instance(owner, self.provider))
                    && expected.residency == ProviderResidency::Device
                    && expected.shape.len() == handle.shape().len()
                    && expected
                        .shape
                        .iter()
                        .zip(handle.shape())
                        .all(|(&expected, &actual)| u64::try_from(actual).ok() == Some(expected))
                    && matches_storage(expected.storage, handle)
                    && matches_layout(expected.layout, handle)
                    && matches_element(expected.element_type, handle)
                    && self
                        .one_output_requested_bytes
                        .is_none_or(|bytes| handle.physical().requested_bytes() == bytes)
            })
    }
}

fn matches_storage(expected: ProviderStorage, handle: &GpuTensorHandle) -> bool {
    matches!(
        (expected, runmat_accelerate_api::handle_storage(handle)),
        (ProviderStorage::DenseReal, GpuTensorStorage::Real)
            | (
                ProviderStorage::DenseComplexInterleaved,
                GpuTensorStorage::ComplexInterleaved
            )
    )
}

fn matches_layout(expected: ProviderLayout, handle: &GpuTensorHandle) -> bool {
    matches!(
        (expected, handle.view().layout()),
        (
            ProviderLayout::ColumnMajorContiguous,
            GpuTensorLayout::Contiguous
        ) | (ProviderLayout::Strided, GpuTensorLayout::Transpose(_))
    )
}

fn matches_element(expected: ProviderElementType, handle: &GpuTensorHandle) -> bool {
    use ProviderElementType::*;
    if expected == Logical {
        return runmat_accelerate_api::handle_is_logical(handle)
            && runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real;
    }
    if runmat_accelerate_api::handle_is_logical(handle) {
        return false;
    }
    if let Some(integer) = runmat_accelerate_api::handle_integer_type(handle) {
        return ProviderElementType::from(integer) == expected;
    }
    match expected {
        F32 | ComplexF32 => {
            runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F32)
        }
        F64 | ComplexF64 => {
            runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F64)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::physical_test_support::OwnedPhysicalProvider;
    use runmat_accelerate_api::ThreadProviderGuard;

    #[test]
    fn selected_physical_extent_rejects_larger_owner_valid_view_without_retiring_inputs() {
        let provider: &'static OwnedPhysicalProvider =
            Box::leak(Box::new(OwnedPhysicalProvider::new()));
        let _owner = ThreadProviderGuard::set(Some(provider));
        let input = provider.allocate_f64(&[1, 1]);
        let exact = provider.allocate_f64(&[1, 1]);
        let (larger, larger_view) = provider.allocate_padded_f64_view(&[2, 2], &[1, 1]);
        assert_eq!(exact.handle.shape(), larger_view.shape());
        assert_eq!(
            exact.handle.physical().element_type(),
            larger_view.physical().element_type()
        );
        assert!(
            larger_view.physical().requested_bytes() > exact.handle.physical().requested_bytes()
        );
        let admitted = AdmittedHostUploadOutputs {
            provider,
            outputs: vec![ProviderRepresentation {
                element_type: ProviderElementType::F64,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: vec![1, 1],
                residency: ProviderResidency::Device,
            }],
            one_output_requested_bytes: Some(exact.handle.physical().requested_bytes()),
        };
        assert!(admitted.admits(&[&exact.handle]));
        assert!(!admitted.admits(&[&larger_view]));
        assert!(!admitted.admits(&[&input.handle, &exact.handle]));
        drop(larger_view);
        drop(larger);
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        assert!(exact.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(input.probe.retirements(), 0);
        assert_eq!(exact.probe.retirements(), 0);
    }
}
