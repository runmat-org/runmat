//! Host-origin integer observations converted by the WGPU sum shader to the
//! ordinary F64 accumulator. Native integer reduction has a different kernel
//! and is deliberately not represented by this recipe.
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
    if route.accumulation_class != NumericClass::Double
        || route.publication_class != NumericClass::Double
        || route.input_class.integer_class().is_none()
    {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.integer_conversion_route_unresolved",
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
            "sum.integer_conversion_dispatch_extent",
        ));
    }
    let Some(geometry) = provider.sum_geometry() else {
        return Err(reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "sum.integer_conversion_geometry_unknown",
        ));
    };
    let input_bytes =
        crate::backend::wgpu_storage::checked_allocation_bytes(input).ok_or_else(|| {
            reject(
                ProviderRejectionCode::InvalidShape,
                "sum.integer_word_storage_overflow",
            )
        })?;
    let accumulated_input_bytes = elements.checked_mul(8).ok_or_else(|| {
        reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.integer_accumulator_overflow",
        )
    })?;
    let accumulated_output_bytes = output_elements.checked_mul(8).ok_or_else(|| {
        reject(
            ProviderRejectionCode::ResourceLimit,
            "sum.integer_output_overflow",
        )
    })?;
    for bytes in [
        input_bytes,
        accumulated_input_bytes,
        accumulated_output_bytes,
        publication_bytes,
    ] {
        check_buffer(bytes, geometry)?;
    }
    let mut reduction = if route.axes.len() == input.shape.len() && elements > 1 {
        global::work(
            accumulated_input_bytes,
            elements,
            accumulated_output_bytes,
            geometry,
        )?
    } else {
        floating::dimension_work(
            input.shape.len(),
            route.axes.len(),
            elements,
            output_elements,
            accumulated_output_bytes,
            geometry,
        )?
    };
    // The conversion owns a distinct F64 buffer and one 16-byte uniform per
    // chunk. A device input would need authenticated backing/view geometry;
    // the enclosing admission currently limits this recipe to host origins.
    floating::add_integer_input_cast(&mut reduction, elements, geometry)?;
    reduction.estimate.output_bytes = Some(publication_bytes.max(8));
    Ok(reduction)
}

fn check_buffer(bytes: u64, geometry: ProviderSumGeometry) -> Result<(), ProviderFeasibility> {
    if bytes > geometry.max_buffer_bytes || bytes > geometry.max_binding_bytes {
        return Err(ProviderFeasibility::rejected(
            ProviderRejectionCode::ResourceLimit,
            "sum.integer_buffer_or_binding_limit",
        ));
    }
    Ok(())
}
