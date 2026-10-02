//! Provider-audited physical bounds for late-shaped upload outputs.

use runmat_accelerate_api::{
    ProviderCapabilitySnapshot, ProviderElementType, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderLayout, ProviderOperationFamily, ProviderRejectionCode, ProviderResidency,
    ProviderResourceEstimate, ProviderStorage,
};

#[cfg(all(test, feature = "wgpu"))]
#[path = "bounded_transfer/tests.rs"]
mod tests;

#[cfg(feature = "wgpu")]
pub(super) fn wgpu_dense_feasibility(
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
    precision: runmat_accelerate_api::ProviderPrecision,
) -> ProviderFeasibility {
    // An empty WGPU numeric upload still owns one storage element. This is a
    // physical estimate; logical transfer volume remains zero.
    let empty_floor = ProviderElementType::from(precision).byte_width();
    super::dense_feasibility_with_bytes(snapshot, query, |representation| {
        representation.checked_byte_len().map(|bytes| {
            if bytes == 0 && query.operation.0 == "transfer.upload" {
                empty_floor
            } else {
                bytes
            }
        })
    })
}

pub(super) fn feasibility(
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let bounds = query.workload.transfer_output_bounds.as_ref()?;
    if query.operation.0 != "transfer.upload"
        || query.family != ProviderOperationFamily::Upload
        || !query.inputs.is_empty()
        || !query.outputs.is_empty()
        || bounds.is_empty()
        || query.workload.requested_outputs != u32::try_from(bounds.len()).ok()
    {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::InvalidShape,
            "provider.transfer.bound_contract",
        ));
    }
    let mut total = 0_u64;
    for bound in bounds {
        // This provider contract admits native float payloads only. Logical
        // and short integers can have wider physical WGPU representations.
        if !matches!(
            bound.element_type,
            ProviderElementType::F32 | ProviderElementType::F64
        ) || !snapshot.element_types.contains(&bound.element_type)
            || bound.storage != ProviderStorage::DenseReal
            || bound.layout != ProviderLayout::ColumnMajorContiguous
            || bound.residency != ProviderResidency::Device
        {
            return Some(ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedElementType,
                "provider.transfer.bound_class",
            ));
        }
        let Some(bytes) = bound.max_bytes() else {
            return Some(ProviderFeasibility::rejected(
                ProviderRejectionCode::ResourceLimit,
                "provider.transfer.bound_overflow",
            ));
        };
        if snapshot
            .max_allocation_bytes
            .is_some_and(|limit| bytes > limit)
        {
            return Some(ProviderFeasibility::rejected(
                ProviderRejectionCode::ResourceLimit,
                "provider.transfer.bound_allocation_limit",
            ));
        }
        let Some(next) = total.checked_add(bytes) else {
            return Some(ProviderFeasibility::rejected(
                ProviderRejectionCode::ResourceLimit,
                "provider.transfer.bound_overflow",
            ));
        };
        total = next;
    }
    Some(ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: None,
        output_bytes: Some(total),
        dispatches: Some(0),
    }))
}
