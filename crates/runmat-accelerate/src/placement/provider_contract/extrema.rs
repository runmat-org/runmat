//! Physical facts for the existing two-output dimensional min/max hook.
//! The caller has already parsed options and selected an in-bounds axis.
use super::*;
use runmat_accelerate_api::{ProviderExtremaDemand, ProviderExtremaKind};

#[cfg(test)]
mod tests;

fn identity(kind: ProviderExtremaKind) -> &'static str {
    match kind {
        ProviderExtremaKind::Minimum => runmat_builtins::MIN_CATALOG_ENTRY.identity.name,
        ProviderExtremaKind::Maximum => runmat_builtins::MAX_CATALOG_ENTRY.identity.name,
    }
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot, wgpu: bool) {
    if !wgpu {
        return;
    }
    for kind in [ProviderExtremaKind::Minimum, ProviderExtremaKind::Maximum] {
        snapshot.operations.push(ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new(identity(kind)),
            family: ProviderOperationFamily::Reduction,
        });
    }
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let kind = [ProviderExtremaKind::Minimum, ProviderExtremaKind::Maximum]
        .into_iter()
        .find(|kind| query.operation.0 == identity(*kind))?;
    Some(work(provider, wgpu, query, kind))
}

#[cfg(feature = "wgpu")]
pub(crate) fn observation_capability(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> runmat_accelerate_api::observation::ProviderObservationCapability {
    use runmat_accelerate_api::observation::{
        ProviderObservationCapability, ProviderObservationOverhead,
    };
    let Some(ProviderFeasibility::Supported { estimate }) = validate(provider, true, query) else {
        return ProviderObservationCapability::Unsupported;
    };
    let produced_readback = query
        .workload
        .extrema
        .as_ref()
        .is_some_and(|route| route.demand == ProviderExtremaDemand::ValuesAndIndices)
        && provider.precision() == ProviderPrecision::F32;
    let Some(compute_encoders) = estimate.dispatches else {
        return ProviderObservationCapability::Unsupported;
    };
    let Some(submissions) = compute_encoders
        .checked_add(u32::from(produced_readback))
        .filter(|count| *count != 0)
    else {
        return ProviderObservationCapability::Unsupported;
    };
    let Some(requested_peak_buffer_bytes) = u64::from(submissions)
        .checked_mul(crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES)
    else {
        return ProviderObservationCapability::Unsupported;
    };
    ProviderObservationCapability::Supported(ProviderObservationOverhead {
        submissions,
        requested_peak_buffer_bytes,
        clear_commands: submissions,
        map_callbacks: submissions,
        wgpu_compute_query: crate::backend::wgpu::dispatch::observation::prospective_encoder_query(
            compute_encoders,
            submissions,
        ),
    })
}

fn reject(code: ProviderRejectionCode, detail: &'static str) -> ProviderFeasibility {
    ProviderFeasibility::rejected(code, detail)
}

fn work(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
    kind: ProviderExtremaKind,
) -> ProviderFeasibility {
    if !wgpu {
        return reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "extrema.in_process_recipe_unresolved",
        );
    }
    let Some(route) = query.workload.extrema.as_ref() else {
        return reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "extrema.missing_admitted_route",
        );
    };
    if route.kind != kind
        || query.family != ProviderOperationFamily::Reduction
        || query.inputs.len() != 1
        || query.outputs.len()
            != match route.demand {
                ProviderExtremaDemand::ValuesOnly => 1,
                ProviderExtremaDemand::ValuesAndIndices => 2,
            }
        || query.workload.requested_outputs != u32::try_from(query.outputs.len()).ok()
    {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "extrema.identity_or_output_demand",
        );
    }
    let input = &query.inputs[0];
    if input.residency != ProviderResidency::Device
        || input.layout != ProviderLayout::ColumnMajorContiguous
        || input.storage != ProviderStorage::DenseReal
    {
        return reject(
            ProviderRejectionCode::UnsupportedLayout,
            "extrema.input_residency_layout_or_storage",
        );
    }
    let Some(backing_bytes) = route
        .input_backing
        .filter(|backing| backing.matches_representation(input))
        .and_then(|backing| backing.live_requested_bytes())
    else {
        return reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "extrema.input_backing_unresolved",
        );
    };
    let Some(geometry) = provider.extrema_geometry() else {
        return reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "extrema.provider_geometry_unresolved",
        );
    };
    if geometry.workgroup_size == 0 || input.shape.len() != 2 || route.axis > 1 {
        return reject(
            ProviderRejectionCode::UnsupportedRank,
            "extrema.rank_or_axis",
        );
    }
    let Some(elements) = input.checked_element_count() else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "extrema.input_overflow",
        );
    };
    let axis = route.axis as usize;
    if elements > u64::from(u32::MAX)
        || (provider.precision() == ProviderPrecision::F32 && input.shape[axis] > (1 << 24))
    {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "extrema.dispatch_or_index_limit",
        );
    }
    if input.shape.contains(&0) {
        // Empty WGPU buffers use backend-minimum physical extents that this
        // route does not yet authenticate as exact retained bytes. Keep the
        // existing direct hook available without a measured placement claim.
        return reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "extrema.empty_physical_extent_unresolved",
        );
    }
    let output_shape = if axis == 0 {
        vec![1, input.shape[1]]
    } else {
        vec![input.shape[0], 1]
    };
    let output_elements = input.shape[1 - axis];
    let groups = output_elements.div_ceil(u64::from(geometry.workgroup_size));
    if groups > u64::from(geometry.max_workgroups_per_dimension) {
        return reject(ProviderRejectionCode::ResourceLimit, "extrema.workgroups");
    }
    let index_type = ProviderElementType::from(provider.precision());
    let value_bytes = match input.element_type {
        ProviderElementType::F32 if provider.precision() == ProviderPrecision::F32 => {
            output_elements.checked_mul(4)
        }
        ProviderElementType::F64 if provider.precision() == ProviderPrecision::F64 => {
            output_elements.checked_mul(8)
        }
        ProviderElementType::I8
        | ProviderElementType::I16
        | ProviderElementType::I32
        | ProviderElementType::U8
        | ProviderElementType::U16
        | ProviderElementType::U32 => output_elements.checked_mul(4),
        ProviderElementType::I64 | ProviderElementType::U64 => output_elements.checked_mul(8),
        _ => {
            return reject(
                ProviderRejectionCode::UnsupportedElementType,
                "extrema.numeric_class",
            )
        }
    };
    let Some((value_bytes, index_bytes)) =
        value_bytes.zip(output_elements.checked_mul(index_type.byte_width()))
    else {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "extrema.output_overflow",
        );
    };
    if query.outputs[0].element_type != input.element_type
        || query.outputs[0].shape != output_shape
        || query.outputs[0].layout != ProviderLayout::ColumnMajorContiguous
        || query.outputs[0].storage != ProviderStorage::DenseReal
        || query.outputs[0].residency != ProviderResidency::Device
        || query.outputs.get(1).is_some_and(|index| {
            index.element_type != index_type
                || index.shape != output_shape
                || index.layout != ProviderLayout::ColumnMajorContiguous
                || index.storage != ProviderStorage::DenseReal
                || index.residency != ProviderResidency::Device
        })
    {
        return reject(ProviderRejectionCode::InvalidShape, "extrema.output_roles");
    }
    if [backing_bytes, value_bytes, index_bytes]
        .into_iter()
        .any(|bytes| bytes > geometry.max_buffer_bytes || bytes > geometry.max_binding_bytes)
    {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "extrema.buffer_limits",
        );
    }
    let Some(two_outputs) = value_bytes.checked_add(index_bytes) else {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "extrema.peak_overflow",
        );
    };
    let retained = match route.demand {
        ProviderExtremaDemand::ValuesOnly => value_bytes,
        ProviderExtremaDemand::ValuesAndIndices => two_outputs,
    };
    // The borrowed source remains live throughout the nonempty hook, which
    // allocates both outputs and a 16/32-byte uniform. ProviderResourceEstimate
    // is the complete simultaneous peak; Runtime subtracts the authenticated
    // already-live backing exactly once when testing incremental capacity.
    let uniform = if matches!(
        input.element_type,
        ProviderElementType::F32 | ProviderElementType::F64
    ) {
        16
    } else {
        32
    };
    let Some(peak) = complete_peak(backing_bytes, two_outputs, uniform) else {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "extrema.peak_overflow",
        );
    };
    ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: Some(peak),
        output_bytes: Some(retained),
        dispatches: Some(1),
    })
}

fn complete_peak(borrowed_bytes: u64, two_outputs: u64, uniform_bytes: u64) -> Option<u64> {
    borrowed_bytes
        .checked_add(two_outputs)?
        .checked_add(uniform_bytes)
}
