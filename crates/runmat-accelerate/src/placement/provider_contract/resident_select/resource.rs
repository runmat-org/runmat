//! Physical selection work and resource admission shared by feasibility and cost.
use super::*;

pub(super) fn estimate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    branch: ProviderElementType,
    elements: u64,
    shape: &[usize],
) -> Result<ResidentSelectWork, ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    #[cfg(not(feature = "wgpu"))]
    let _ = (provider, shape);
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
