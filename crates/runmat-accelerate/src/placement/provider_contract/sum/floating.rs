//! Complete host-origin real floating sum stages: upload, optional F64 input
//! conversion, the existing F64 reduction kernel and public Single cast.
use super::*;
use runmat_accelerate_api::ProviderSumGeometry;

pub(super) fn work(
    provider: &(impl AccelProvider + ?Sized),
    input: &ProviderRepresentation,
    route: &ProviderSumWorkload,
    elements: u64,
    output_elements: u64,
    publication_bytes: u64,
) -> Result<SumWork, ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    if !matches!(
        route.input_class,
        NumericClass::Single | NumericClass::Double
    ) || !matches!(
        route.publication_class,
        NumericClass::Single | NumericClass::Double
    ) {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.floating_class_route_unresolved",
        ));
    }
    if input
        .shape
        .iter()
        .any(|&extent| extent > u64::from(u32::MAX))
        || elements > u64::from(u32::MAX)
        || output_elements > u64::from(u32::MAX)
    {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.dispatch_extent",
        ));
    }
    let Some(geometry) = provider.sum_geometry() else {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.kernel_geometry_unknown",
        ));
    };
    let input_bytes = input.checked_byte_len().ok_or_else(|| {
        reject(
            ProviderRejectionCode::InvalidShape,
            "sum.input_bytes_overflow",
        )
    })?;
    let accumulated_input_bytes = elements.checked_mul(8).ok_or_else(|| {
        reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.accumulator_bytes_overflow",
        )
    })?;
    let accumulated_output_bytes = output_elements.checked_mul(8).ok_or_else(|| {
        reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.accumulated_output_bytes_overflow",
        )
    })?;
    for bytes in [
        input_bytes,
        accumulated_input_bytes,
        accumulated_output_bytes,
        publication_bytes,
    ] {
        if bytes > geometry.max_buffer_bytes || bytes > geometry.max_binding_bytes {
            return Err(reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.floating_buffer_or_binding_limit",
            ));
        }
    }
    let mut reduction = if route.axes.len() == input.shape.len() && elements > 1 {
        global::work(
            accumulated_input_bytes,
            elements,
            accumulated_output_bytes,
            geometry,
        )?
    } else {
        dimension_work(
            input.shape.len(),
            route.axes.len(),
            elements,
            output_elements,
            accumulated_output_bytes,
            geometry,
        )?
    };
    if route.input_class == NumericClass::Single {
        add_cast(&mut reduction, elements, geometry, CastStage::Input)?;
    }
    if route.publication_class == NumericClass::Single {
        add_cast(&mut reduction, output_elements, geometry, CastStage::Output)?;
    }
    reduction.estimate.output_bytes = Some(publication_bytes.max(4));
    Ok(reduction)
}

pub(super) fn dimension_work(
    rank: usize,
    axes: usize,
    elements: u64,
    output_elements: u64,
    output_bytes: u64,
    geometry: ProviderSumGeometry,
) -> Result<SumWork, ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    if rank < 2 || axes != 1 {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.physical_recipe_unresolved",
        ));
    }
    let (dispatches, uniform_bytes) = if rank == 2 {
        if output_elements > geometry.dimension_2_max_output_elements {
            return Err(reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.dimension_2_dispatch_limit",
            ));
        }
        (1_u32, 16_u64)
    } else {
        if geometry.nd_chunk_elements == 0 {
            return Err(reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "sum.nd_chunk_geometry_unknown",
            ));
        }
        let chunks = output_elements.div_ceil(geometry.nd_chunk_elements);
        let dispatches = u32::try_from(chunks).map_err(|_| {
            reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.nd_dispatch_count",
            )
        })?;
        let uniform_bytes = chunks
            .checked_mul(16)
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.nd_uniform_bytes"))?;
        (dispatches, uniform_bytes)
    };
    #[cfg(not(feature = "wgpu"))]
    let _ = (elements, uniform_bytes);
    Ok(SumWork {
        estimate: ProviderResourceEstimate {
            output_bytes: Some(output_bytes.max(8)),
            transient_bytes: None,
            dispatches: Some(dispatches),
        },
        #[cfg(feature = "wgpu")]
        kernel: if rank == 2 {
            SumKernelRoute::Dimension2
        } else {
            SumKernelRoute::DimensionN
        },
        #[cfg(feature = "wgpu")]
        input_cast: false,
        #[cfg(feature = "wgpu")]
        integer_input_cast: false,
        #[cfg(feature = "wgpu")]
        output_cast: false,
        #[cfg(feature = "wgpu")]
        nan_map: false,
        #[cfg(feature = "wgpu")]
        scalar_visits: elements.checked_add(output_elements).ok_or_else(|| {
            reject(
                ProviderRejectionCode::ResourceLimit,
                "sum.scalar_work_overflow",
            )
        })?,
        #[cfg(feature = "wgpu")]
        queue_submissions: dispatches,
        // One host input upload, one reduction result, one uniform per chunk.
        #[cfg(feature = "wgpu")]
        allocations: dispatches
            .checked_add(2)
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.allocation_count"))?,
        #[cfg(feature = "wgpu")]
        uniform_bytes,
    })
}

#[derive(Clone, Copy)]
enum CastStage {
    Input,
    IntegerInput,
    Output,
}

pub(super) fn add_input_cast(
    work: &mut SumWork,
    elements: u64,
    geometry: ProviderSumGeometry,
) -> Result<(), ProviderFeasibility> {
    add_cast(work, elements, geometry, CastStage::Input)
}

pub(super) fn add_integer_input_cast(
    work: &mut SumWork,
    elements: u64,
    geometry: ProviderSumGeometry,
) -> Result<(), ProviderFeasibility> {
    add_cast(work, elements, geometry, CastStage::IntegerInput)
}

pub(super) fn add_output_cast(
    work: &mut SumWork,
    elements: u64,
    geometry: ProviderSumGeometry,
) -> Result<(), ProviderFeasibility> {
    add_cast(work, elements, geometry, CastStage::Output)
}

fn add_cast(
    work: &mut SumWork,
    elements: u64,
    geometry: ProviderSumGeometry,
    stage: CastStage,
) -> Result<(), ProviderFeasibility> {
    let reject = ProviderFeasibility::rejected;
    let chunk_elements = match stage {
        CastStage::IntegerInput => geometry.integer_cast_chunk_elements,
        CastStage::Input | CastStage::Output => geometry.cast_chunk_elements,
    };
    if chunk_elements == 0 {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.cast_chunk_geometry_unknown",
        ));
    }
    let chunks = u32::try_from(elements.div_ceil(chunk_elements)).map_err(|_| {
        reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.cast_dispatch_count",
        )
    })?;
    work.estimate.dispatches = work
        .estimate
        .dispatches
        .and_then(|old| old.checked_add(chunks));
    if work.estimate.dispatches.is_none() {
        return Err(reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.composite_dispatch_count",
        ));
    }
    #[cfg(not(feature = "wgpu"))]
    let _ = stage;
    #[cfg(feature = "wgpu")]
    {
        // A composite stage may already include a queue submission that did
        // not dispatch work (the fused NaN-map pipeline warmup). Add only this
        // cast's chunks; assigning total dispatches would erase that cost.
        work.queue_submissions = work
            .queue_submissions
            .checked_add(chunks)
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.cast_queue"))?;
        work.allocations = work
            .allocations
            .checked_add(chunks)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.cast_allocations"))?;
        work.uniform_bytes = work
            .uniform_bytes
            .checked_add(u64::from(chunks) * 16)
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.cast_uniforms"))?;
        work.scalar_visits = work
            .scalar_visits
            .checked_add(elements)
            .ok_or_else(|| reject(ProviderRejectionCode::ResourceLimit, "sum.cast_scalar_work"))?;
        match stage {
            CastStage::Input => work.input_cast = true,
            CastStage::IntegerInput => work.integer_input_cast = true,
            CastStage::Output => work.output_cast = true,
        }
    }
    Ok(())
}
