//! Physical pass geometry of the existing real-F64 global sum kernel.
use super::*;
use runmat_accelerate_api::ProviderSumGeometry;

pub(super) fn work(
    input_bytes: u64,
    elements: u64,
    output_bytes: u64,
    geometry: ProviderSumGeometry,
) -> Result<SumWork, ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    if input_bytes > geometry.max_buffer_bytes || input_bytes > geometry.max_binding_bytes {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.global_input_binding_limit",
        ));
    }
    if geometry.global_group_elements < 2
        || geometry.global_chunk_elements < geometry.global_group_elements
        || !geometry
            .global_chunk_elements
            .is_multiple_of(geometry.global_group_elements)
    {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.global_geometry_unavailable",
        ));
    }
    if output_bytes > geometry.max_buffer_bytes || output_bytes > geometry.max_binding_bytes {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.global_output_binding_limit",
        ));
    }
    let mut current = elements;
    let mut dispatches = 0_u32;
    // Admitted host data is uploaded once; each pass owns a result buffer and
    // each dispatched chunk owns its uniform buffer.
    let mut allocations = 1_u32;
    let mut scalar_visits = 0_u64;
    let mut uniform_bytes = 0_u64;
    while current > 1 {
        let partial = current.div_ceil(geometry.global_group_elements);
        let partial_bytes = partial.checked_mul(8).ok_or_else(|| {
            reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.global_partial_bytes_overflow",
            )
        })?;
        if partial_bytes > geometry.max_buffer_bytes || partial_bytes > geometry.max_binding_bytes {
            return Err(reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.global_partial_binding_limit",
            ));
        }
        let chunks = current.div_ceil(geometry.global_chunk_elements);
        let chunks = u32::try_from(chunks).map_err(|_| {
            reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.global_dispatch_count",
            )
        })?;
        dispatches = dispatches.checked_add(chunks).ok_or_else(|| {
            reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.global_dispatch_count",
            )
        })?;
        allocations = allocations
            .checked_add(chunks)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| {
                reject(
                    ProviderRejectionCode::ResourceLimit,
                    "sum.global_allocation_count",
                )
            })?;
        scalar_visits = scalar_visits
            .checked_add(current)
            .and_then(|count| count.checked_add(partial))
            .ok_or_else(|| {
                reject(
                    ProviderRejectionCode::ResourceLimit,
                    "sum.global_scalar_work_overflow",
                )
            })?;
        uniform_bytes = uniform_bytes
            .checked_add(u64::from(chunks) * 16)
            .ok_or_else(|| {
                reject(
                    ProviderRejectionCode::ResourceLimit,
                    "sum.global_uniform_bytes_overflow",
                )
            })?;
        current = partial;
    }
    #[cfg(not(feature = "wgpu"))]
    let _ = (allocations, scalar_visits, uniform_bytes);
    Ok(SumWork {
        estimate: ProviderResourceEstimate {
            output_bytes: Some(output_bytes.max(8)),
            transient_bytes: None,
            dispatches: Some(dispatches),
        },
        #[cfg(feature = "wgpu")]
        kernel: SumKernelRoute::Global,
        #[cfg(feature = "wgpu")]
        input_cast: false,
        #[cfg(feature = "wgpu")]
        integer_input_cast: false,
        #[cfg(feature = "wgpu")]
        output_cast: false,
        #[cfg(feature = "wgpu")]
        nan_map: false,
        #[cfg(feature = "wgpu")]
        scalar_visits,
        #[cfg(feature = "wgpu")]
        queue_submissions: dispatches,
        #[cfg(feature = "wgpu")]
        allocations,
        #[cfg(feature = "wgpu")]
        uniform_bytes,
    })
}
