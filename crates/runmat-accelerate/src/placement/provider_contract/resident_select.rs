//! Exact feasibility of the three-input resident selection primitive.
use super::*;
use runmat_accelerate_api::RESIDENT_SELECT_OPERATION_IDENTITY;

#[cfg(test)]
mod tests;

pub(crate) struct ResidentSelectWork {
    pub estimate: ProviderResourceEstimate,
    #[cfg(feature = "wgpu")]
    pub uniform_bytes: u64,
    #[cfg(feature = "wgpu")]
    pub queue_submissions: u32,
    #[cfg(feature = "wgpu")]
    pub scalar_visits: u64,
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.push(ProviderCapabilityOperation {
        identity: ProviderOperationIdentity::new(RESIDENT_SELECT_OPERATION_IDENTITY),
        family: ProviderOperationFamily::Elementwise,
    });
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    analyze(provider, wgpu, query).map(|work| match work {
        Ok(work) => ProviderFeasibility::supported(work.estimate),
        Err(rejection) => rejection,
    })
}

#[cfg(feature = "wgpu")]
pub(crate) fn physical_work(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Option<ResidentSelectWork> {
    analyze(provider, true, query)?.ok()
}

fn analyze(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<Result<ResidentSelectWork, ProviderFeasibility>> {
    if query.operation.0 != RESIDENT_SELECT_OPERATION_IDENTITY {
        return None;
    }
    Some(evaluate(provider, wgpu, query))
}

fn evaluate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Result<ResidentSelectWork, ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    if query.family != ProviderOperationFamily::Elementwise
        || query.inputs.len() != 3
        || query.outputs.len() != 1
    {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.arity_or_family",
        ));
    }
    let mask = &query.inputs[0];
    let then_value = &query.inputs[1];
    let else_value = &query.inputs[2];
    let output = &query.outputs[0];
    if query.inputs.iter().chain(&query.outputs).any(|value| {
        value.storage != ProviderStorage::DenseReal || value.residency != ProviderResidency::Device
    }) {
        return Err(reject(
            ProviderRejectionCode::UnsupportedStorage,
            "resident_select.resident_real_required",
        ));
    }
    let native = ProviderElementType::from(provider.precision());
    let branch = then_value.element_type;
    if mask.element_type != ProviderElementType::Logical
        || !matches!(branch, ProviderElementType::F32 | ProviderElementType::F64)
        || branch != else_value.element_type
        || (wgpu && branch != native)
        || output.element_type != branch
    {
        return Err(reject(
            ProviderRejectionCode::UnsupportedElementType,
            "resident_select.logical_mask_matching_real_branches",
        ));
    }
    for input in &query.inputs {
        let contiguous = input.layout == ProviderLayout::ColumnMajorContiguous;
        let supported_strided =
            wgpu && input.layout == ProviderLayout::Strided && input.shape.len() == 2;
        if !contiguous && !supported_strided {
            return Err(reject(
                ProviderRejectionCode::UnsupportedLayout,
                "resident_select.input_layout",
            ));
        }
    }
    if output.layout != ProviderLayout::ColumnMajorContiguous {
        return Err(reject(
            ProviderRejectionCode::UnsupportedLayout,
            "resident_select.output_layout",
        ));
    }
    let mut shapes = Vec::new();
    for representation in &query.inputs {
        let Ok(shape) = representation
            .shape
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
        else {
            return Err(reject(
                ProviderRejectionCode::InvalidShape,
                "resident_select.dimension_overflow",
            ));
        };
        shapes.push(shape);
    }
    let Ok(shape) = runmat_types::geometry::broadcast_concrete_shape(&shapes[0], &shapes[1])
        .and_then(|shape| runmat_types::geometry::broadcast_concrete_shape(&shape, &shapes[2]))
    else {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.broadcast_shape",
        ));
    };
    let Ok(shape_u64) = shape
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
    else {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.dimension_overflow",
        ));
    };
    if output.shape != shape_u64 {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.output_shape",
        ));
    }
    let Some(elements) = shape_u64
        .iter()
        .try_fold(1_u64, |count, &extent| count.checked_mul(extent))
    else {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.element_count_overflow",
        ));
    };
    if query
        .workload
        .elements
        .is_some_and(|declared| declared != elements)
    {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "resident_select.workload_geometry",
        ));
    }
    let (uniform_bytes, dispatches) = if wgpu {
        #[cfg(feature = "wgpu")]
        {
            if shape.len() > crate::backend::wgpu::params::BCAST_MAX_RANK
                || shape.iter().any(|&extent| extent > u32::MAX as usize)
            {
                return Err(reject(
                    ProviderRejectionCode::UnsupportedRank,
                    "resident_select.broadcast_rank_or_extent",
                ));
            }
            if elements > u64::from(u32::MAX) {
                return Err(reject(
                    ProviderRejectionCode::ResourceLimit,
                    "resident_select.dispatch_length",
                ));
            }
            let Some(geometry) = provider.elementwise_geometry() else {
                return Err(reject(
                    ProviderRejectionCode::UnknownResourceEstimate,
                    "resident_select.dispatch_geometry_unknown",
                ));
            };
            let Some(capacity) =
                crate::backend::wgpu_geometry::elementwise_chunk_capacity(geometry.workgroup_size)
            else {
                return Err(reject(
                    ProviderRejectionCode::UnknownResourceEstimate,
                    "resident_select.dispatch_geometry_unknown",
                ));
            };
            let Some(uniform_bytes) = crate::backend::wgpu_geometry::fusion_broadcast_uniform_bytes(
                3,
                crate::backend::wgpu::params::BCAST_MAX_RANK,
            ) else {
                return Err(reject(
                    ProviderRejectionCode::ResourceLimit,
                    "resident_select.uniform_overflow",
                ));
            };
            (
                uniform_bytes,
                u32::try_from(elements.div_ceil(capacity)).map_err(|_| {
                    reject(
                        ProviderRejectionCode::ResourceLimit,
                        "resident_select.dispatch_overflow",
                    )
                })?,
            )
        }
        #[cfg(not(feature = "wgpu"))]
        {
            return Err(reject(
                ProviderRejectionCode::UnsupportedOperation,
                "resident_select.wgpu_unavailable",
            ));
        }
    } else {
        (0, u32::from(elements != 0))
    };
    let Some(output_bytes) = elements
        .max(u64::from(wgpu))
        .checked_mul(branch.byte_width())
    else {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "resident_select.output_bytes_overflow",
        ));
    };
    let Some(_incremental_bytes) = output_bytes.checked_add(uniform_bytes) else {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "resident_select.incremental_bytes_overflow",
        ));
    };
    if wgpu
        && provider
            .device_info_struct()
            .memory_bytes
            .is_some_and(|maximum| output_bytes > maximum)
    {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "resident_select.output_buffer_limit",
        ));
    }
    // A representation records logical shape, not a resident allocation's
    // possibly larger physical backing. Its total live footprint cannot be
    // certified from this query; leave total transient bytes unknown rather
    // than present the payload minimum as an upper bound.
    Ok(ResidentSelectWork {
        estimate: ProviderResourceEstimate {
            transient_bytes: None,
            output_bytes: Some(output_bytes),
            dispatches: Some(dispatches),
        },
        #[cfg(feature = "wgpu")]
        uniform_bytes,
        #[cfg(feature = "wgpu")]
        queue_submissions: dispatches.saturating_add(u32::from(wgpu)),
        #[cfg(feature = "wgpu")]
        scalar_visits: elements,
    })
}
