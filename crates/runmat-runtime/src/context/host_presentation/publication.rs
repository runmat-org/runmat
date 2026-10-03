//! The selected upload's typed output and provider-owned physical evidence
//! remain attached to the receipt until the realized result is published.

use runmat_accelerate_api::{
    provider_for_handle, same_provider_instance, AccelProvider, GpuTensorHandle, GpuTensorLayout,
    GpuTensorStorage, ProviderElementType, ProviderLayout, ProviderPrecision,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};

pub(super) struct AdmittedHostUploadOutputs {
    pub(super) provider: &'static dyn AccelProvider,
    pub(super) outputs: Vec<ProviderRepresentation>,
    /// A provider can certify an exact physical one-output route. Absence is
    /// unavailable evidence, not an inferred logical byte count.
    pub(super) one_output_requested_bytes: Option<u64>,
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
    use crate::builtins::common::gpu_helpers;
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{GpuTensorView, ThreadProviderGuard};
    use runmat_value::Tensor;

    #[test]
    fn selected_physical_extent_rejects_larger_owner_valid_view_without_retiring_inputs() {
        let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
        let _owner = ThreadProviderGuard::set(Some(provider));
        let input = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![3.0], vec![1, 1]).expect("input"),
        )
        .expect("resident input");
        let exact = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![6.0], vec![1, 1]).expect("exact output"),
        )
        .expect("one-element output");
        let larger = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![6.0; 4], vec![2, 2]).expect("large output"),
        )
        .expect("four-element output");
        let larger_view = larger
            .with_view(GpuTensorView::contiguous(vec![1, 1]))
            .expect("smaller view retains the larger physical allocation");
        assert_eq!(exact.shape(), larger_view.shape());
        assert!(larger_view.physical().requested_bytes() > exact.physical().requested_bytes());
        let admitted = AdmittedHostUploadOutputs {
            provider,
            outputs: vec![ProviderRepresentation {
                element_type: ProviderElementType::F64,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: vec![1, 1],
                residency: ProviderResidency::Device,
            }],
            one_output_requested_bytes: Some(exact.physical().requested_bytes()),
        };
        assert!(admitted.admits(&[&exact]));
        assert!(!admitted.admits(&[&larger_view]));
        assert!(!admitted.admits(&[&input, &exact]));
        drop(larger_view);
        drop(larger);
        assert!(input.allocation_lease().validate_live().is_ok());
        assert!(exact.allocation_lease().validate_live().is_ok());
    }
}
