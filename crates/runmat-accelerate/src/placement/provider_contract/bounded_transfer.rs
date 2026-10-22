//! Provider-audited physical bounds for late-shaped upload outputs.

use runmat_accelerate_api::{
    ProviderCapabilitySnapshot, ProviderElementType, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderLayout, ProviderOperationFamily, ProviderRejectionCode, ProviderResidency,
    ProviderResourceEstimate, ProviderStorage, ProviderTransferOutputBound,
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
    if snapshot.supports_operation(&query.operation, query.family) {
        if let Some(result) = feasibility_with_bytes(snapshot, query, |bound| {
            crate::backend::wgpu_storage::checked_upload_bound_allocation_bytes(bound, precision)
        }) {
            return result;
        }
    }
    super::dense_feasibility_with_bytes(snapshot, query, |representation| {
        if query.operation.0 == "transfer.upload" {
            crate::backend::wgpu_storage::checked_upload_allocation_bytes(representation, precision)
        } else {
            representation.checked_byte_len()
        }
    })
}

pub(super) fn feasibility(
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    feasibility_with_bytes(snapshot, query, ProviderTransferOutputBound::max_bytes)
}

fn feasibility_with_bytes(
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
    allocation_bytes: impl Fn(&ProviderTransferOutputBound) -> Option<u64>,
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
        let Some(bytes) = allocation_bytes(bound) else {
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

/// An exact upload query represents one host payload per device result. A
/// multi-result query is a sequence of those exact uploads, never one input
/// that magically produces additional outputs. This runs before generic dense
/// byte estimation for both native and non-native classes.
pub(super) fn exact_contract(query: &ProviderFeasibilityQuery) -> Option<ProviderFeasibility> {
    if query.operation.0 != "transfer.upload" || query.family != ProviderOperationFamily::Upload {
        return None;
    }
    let count = u32::try_from(query.outputs.len()).ok();
    let valid_count = !query.inputs.is_empty()
        && query.inputs.len() == query.outputs.len()
        && count.is_some_and(|count| {
            query.workload.requested_outputs == Some(count)
                || (count == 1 && query.workload.requested_outputs.is_none())
        });
    let valid_pairs = query
        .inputs
        .iter()
        .zip(&query.outputs)
        .all(|(input, output)| {
            let expected_storage = match input.element_type {
                ProviderElementType::ComplexF32 | ProviderElementType::ComplexF64 => {
                    ProviderStorage::DenseComplexInterleaved
                }
                _ => ProviderStorage::DenseReal,
            };
            input.element_type == output.element_type
                && input.storage == expected_storage
                && output.storage == expected_storage
                && input.layout == ProviderLayout::ColumnMajorContiguous
                && output.layout == ProviderLayout::ColumnMajorContiguous
                && input.shape == output.shape
                && input.residency == ProviderResidency::Host
                && output.residency == ProviderResidency::Device
        });
    if valid_count && valid_pairs {
        None
    } else {
        Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::InvalidShape,
            "provider.transfer.exact_contract",
        ))
    }
}
