//! Real-F64 omission runs a fused NaN map, then one dimensional reduction per
//! selected axis. It does not use the global reduction shortcut.
use super::*;

pub(super) fn work(
    provider: &(impl AccelProvider + ?Sized),
    input: &ProviderRepresentation,
    route: &ProviderSumWorkload,
    elements: u64,
    output_elements: u64,
    publication_bytes: u64,
) -> Result<SumWork, ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    // The owning runtime widens Single observations to the exact F64
    // accumulator before invoking the provider's native F64 NaN-map hook.
    if !matches!(
        route.input_class,
        NumericClass::Single | NumericClass::Double
    ) || !matches!(
        route.publication_class,
        NumericClass::Single | NumericClass::Double
    ) || route.axes.is_empty()
    {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.omit_physical_route_unresolved",
        ));
    }
    if elements > u64::from(u32::MAX)
        || input
            .shape
            .iter()
            .any(|&extent| extent > u64::from(u32::MAX))
    {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.omit_dispatch_extent",
        ));
    }
    let Some(geometry) = provider.sum_geometry() else {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.omit_geometry_unknown",
        ));
    };
    if geometry.cast_chunk_elements == 0 {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.omit_map_chunk_unknown",
        ));
    }
    let input_bytes = input.checked_byte_len().ok_or_else(|| {
        reject(
            ProviderRejectionCode::InvalidShape,
            "sum.omit_input_bytes_overflow",
        )
    })?;
    check_buffer(input_bytes, geometry)?;
    let mapped_bytes = elements.checked_mul(8).ok_or_else(|| {
        reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.omit_mapped_bytes_overflow",
        )
    })?;
    check_buffer(mapped_bytes, geometry)?;
    let map_chunks =
        u32::try_from(elements.div_ceil(geometry.cast_chunk_elements)).map_err(|_| {
            reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.omit_map_dispatches",
            )
        })?;
    let mut shape = input.shape.clone();
    let mut current_elements = elements;
    let mut reduction = None;
    for &axis in &route.axes {
        let selected = usize::try_from(axis).map_err(|_| {
            reject(
                ProviderRejectionCode::InvalidShape,
                "sum.omit_axis_overflow",
            )
        })?;
        let Some(extent) = shape.get_mut(selected) else {
            return Err(reject(
                ProviderRejectionCode::InvalidShape,
                "sum.omit_axis_out_of_bounds",
            ));
        };
        *extent = 1;
        let next_elements = shape
            .iter()
            .try_fold(1_u64, |n, extent| n.checked_mul(*extent))
            .ok_or_else(|| {
                reject(
                    ProviderRejectionCode::ResourceLimit,
                    "sum.omit_intermediate_shape",
                )
            })?;
        let next_bytes = next_elements.checked_mul(8).ok_or_else(|| {
            reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.omit_intermediate_bytes",
            )
        })?;
        check_buffer(next_bytes, geometry)?;
        let stage = floating::dimension_work(
            shape.len(),
            1,
            current_elements,
            next_elements,
            next_bytes,
            geometry,
        )?;
        reduction = Some(match reduction {
            None => stage,
            Some(mut prior) => {
                merge_stage(&mut prior, stage)?;
                prior
            }
        });
        current_elements = next_elements;
    }
    if current_elements != output_elements {
        return Err(reject(
            ProviderRejectionCode::InvalidShape,
            "sum.omit_output_geometry_mismatch",
        ));
    }
    let mut work = reduction
        .ok_or_else(|| reject(ProviderRejectionCode::InvalidShape, "sum.omit_missing_axis"))?;
    add_map(&mut work, elements, map_chunks)?;
    if route.input_class == NumericClass::Single {
        floating::add_input_cast(&mut work, elements, geometry)?;
    }
    if route.publication_class == NumericClass::Single {
        floating::add_output_cast(&mut work, output_elements, geometry)?;
    }
    work.estimate.output_bytes = Some(publication_bytes.max(8));
    Ok(work)
}

fn check_buffer(
    bytes: u64,
    geometry: runmat_accelerate_api::ProviderSumGeometry,
) -> Result<(), ProviderFeasibility> {
    if bytes > geometry.max_buffer_bytes || bytes > geometry.max_binding_bytes {
        return Err(ProviderFeasibility::rejected(
            ProviderRejectionCode::ResourceLimit,
            "sum.omit_buffer_or_binding_limit",
        ));
    }
    Ok(())
}

fn merge_stage(prior: &mut SumWork, next: SumWork) -> Result<(), ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    prior.estimate.dispatches = prior
        .estimate
        .dispatches
        .zip(next.estimate.dispatches)
        .and_then(|(left, right)| left.checked_add(right));
    if prior.estimate.dispatches.is_none() {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.omit_dispatch_count",
        ));
    }
    #[cfg(feature = "wgpu")]
    {
        prior.scalar_visits = prior
            .scalar_visits
            .checked_add(next.scalar_visits)
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.omit_scalar_work"))?;
        prior.queue_submissions = prior
            .queue_submissions
            .checked_add(next.queue_submissions)
            .ok_or_else(|| {
                reject(
                    ProviderRejectionCode::ResourceLimit,
                    "sum.omit_queue_submissions",
                )
            })?;
        let incremental_allocations = next.allocations.checked_sub(1).ok_or_else(|| {
            reject(
                ProviderRejectionCode::InvalidShape,
                "sum.omit_stage_allocation",
            )
        })?;
        prior.allocations = prior
            .allocations
            .checked_add(incremental_allocations)
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.omit_allocations"))?;
        prior.uniform_bytes = prior
            .uniform_bytes
            .checked_add(next.uniform_bytes)
            .ok_or_else(|| {
                reject(
                    ProviderRejectionCode::ResourceLimit,
                    "sum.omit_uniform_bytes",
                )
            })?;
    }
    Ok(())
}

fn add_map(work: &mut SumWork, elements: u64, chunks: u32) -> Result<(), ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    work.estimate.dispatches = work.estimate.dispatches.and_then(|n| n.checked_add(chunks));
    if work.estimate.dispatches.is_none() {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.omit_total_dispatches",
        ));
    }
    #[cfg(not(feature = "wgpu"))]
    let _ = elements;
    #[cfg(feature = "wgpu")]
    {
        work.nan_map = true;
        // The fused-map owner always submits its pipeline warmup no-op before
        // one actual dispatch per chunk, including for a cached pipeline.
        work.queue_submissions = work
            .queue_submissions
            .checked_add(chunks)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.omit_map_queue"))?;
        work.allocations = work.allocations.checked_add(2).ok_or_else(|| {
            reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.omit_map_allocations",
            )
        })?;
        work.uniform_bytes = work
            .uniform_bytes
            .checked_add(16)
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.omit_map_uniform"))?;
        work.scalar_visits = work
            .scalar_visits
            .checked_add(elements)
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.omit_map_work"))?;
    }
    Ok(())
}
