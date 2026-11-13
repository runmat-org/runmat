//! Exact readback admission belongs to the registry's native payload, not this
//! provider's preferred F64 computation class. No payload is cloned or scanned.
mod staging;
use super::*;
use runmat_accelerate_api::{
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderRejectionCode as Rejection,
    ProviderResourceEstimate,
};

impl InProcessProvider {
    pub(super) fn native_numeric_download_feasibility(
        &self,
        source: &GpuTensorHandle,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderFeasibility {
        if !query.matches_numeric_download(source) || source.device_id() != self.device_id {
            return ProviderFeasibility::rejected(
                Rejection::InvalidShape,
                "in_process.download.exact_contract",
            );
        }
        // Floating readback follows the native payload. Integer admission
        // retains its existing generic contract.
        if !matches!(
            source.physical().element_type(),
            NumericElementType::F32 | NumericElementType::F64
        ) {
            return self.query_feasibility(query);
        }
        if source.allocation_lease().validate_live().is_err() {
            return ProviderFeasibility::rejected(
                Rejection::ProviderUnavailable,
                "in_process.download.source_unavailable",
            );
        }
        if source.view().validate(source.physical()).is_err() {
            return ProviderFeasibility::rejected(
                Rejection::InvalidShape,
                "in_process.download.source_view",
            );
        }
        // The current numeric download clones the full registry vector into
        // the host result. It does not implement a lazy transpose/subview.
        if source.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous {
            return ProviderFeasibility::rejected(
                Rejection::UnsupportedLayout,
                "in_process.download.native_layout",
            );
        }
        let Ok(registry) = numeric_registry().lock_for(self.device_id) else {
            return ProviderFeasibility::rejected(
                Rejection::ProviderUnavailable,
                "in_process.download.owner_unavailable",
            );
        };
        let Some(data) = registry.get(source) else {
            return ProviderFeasibility::rejected(
                Rejection::ProviderUnavailable,
                "in_process.download.source_owner",
            );
        };
        let element = match data {
            InProcessNumericData::F32(_) => NumericElementType::F32,
            InProcessNumericData::F64(_) => NumericElementType::F64,
            InProcessNumericData::Integer(_) => {
                return ProviderFeasibility::rejected(
                    Rejection::UnknownResourceEstimate,
                    "in_process.download.payload_class",
                );
            }
        };
        let count = if source.shape().contains(&0) {
            Some(0)
        } else {
            source
                .shape()
                .iter()
                .try_fold(1_usize, |count, &extent| count.checked_mul(extent))
        };
        if source.physical().element_type() != element
            || source.physical().storage() != GpuTensorStorage::Real
            || count != Some(data.len())
            || u64::try_from(data.len()).ok() != Some(source.physical().element_capacity())
            || data.byte_len() != source.physical().payload_bytes()
        {
            return ProviderFeasibility::rejected(
                Rejection::UnknownResourceEstimate,
                "in_process.download.physical_mismatch",
            );
        }
        ProviderFeasibility::supported(ProviderResourceEstimate {
            // The returned native vector is the registry clone itself; no
            // provider-side staging or queue submission exists on this owner.
            transient_bytes: Some(0),
            output_bytes: Some(data.byte_len()),
            dispatches: Some(0),
        })
    }
}
