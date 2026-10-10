//! Physical stages of Double arithmetic with one final Single publication.
//! The cast owner supplies both cast recipes; this module only composes their
//! simultaneously live outputs with the native fill and power stages.

use super::*;
use runmat_accelerate_api::{ProviderPreparationRoute, ProviderWorkload};

pub(super) struct MixedPhysicalWork {
    pub temporary_bytes: u64,
    pub dispatches: u32,
    pub unknown_peak: bool,
}

/// The synchronous provider uses the same typed cast admission as direct
/// floating conversion. Its fill and paired-domain power hooks allocate
/// native Double vectors; no shader geometry, uniforms, or queue dispatch are
/// attributed to this route.
pub(super) fn in_process_physical_work(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
    public_output: &ProviderRepresentation,
    native_words: u64,
) -> Result<MixedPhysicalWork, (ProviderRejectionCode, &'static str)> {
    use ProviderRejectionCode as Reject;
    let resident = query
        .inputs
        .iter()
        .find(|input| input.residency == ProviderResidency::Device)
        .ok_or((Reject::InvalidShape, "power.mixed.resident.required"))?;
    let elements = resident
        .checked_element_count()
        .ok_or((Reject::InvalidShape, "power.mixed.resident.shape"))?;
    let widen = if resident.element_type == ProviderElementType::F32 {
        Some(
            super::super::floating_conversion::in_process_physical_work(
                provider,
                &cast_query(resident.clone(), ProviderPrecision::F64),
            )
            .ok_or((
                Reject::UnknownResourceEstimate,
                "power.mixed.widen.unproven",
            ))?,
        )
    } else {
        None
    };
    let native_output = ProviderRepresentation {
        element_type: if public_output.storage == ProviderStorage::DenseComplexInterleaved {
            ProviderElementType::ComplexF64
        } else {
            ProviderElementType::F64
        },
        storage: public_output.storage,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: resident.shape.clone(),
        residency: ProviderResidency::Device,
    };
    let final_cast = super::super::floating_conversion::in_process_physical_work(
        provider,
        &cast_query(native_output, ProviderPrecision::F32),
    )
    .ok_or((
        Reject::UnknownResourceEstimate,
        "power.mixed.round.unproven",
    ))?;
    let public_bytes = public_output
        .checked_byte_len()
        .ok_or((Reject::ResourceLimit, "power.mixed.public.bytes"))?;
    if final_cast.output_bytes != Some(public_bytes) || final_cast.dispatches != Some(0) {
        return Err((Reject::UnknownResourceEstimate, "power.mixed.round.recipe"));
    }
    let mut temporary_bytes = elements
        .checked_mul(8)
        .and_then(|fill| {
            native_words
                .checked_mul(8)
                .and_then(|native| fill.checked_add(native))
        })
        .ok_or((Reject::ResourceLimit, "power.mixed.native.bytes"))?;
    let mut unknown_peak = elements == 0 || final_cast.transient_bytes.is_none();
    if let Some(widen) = widen {
        temporary_bytes = temporary_bytes
            .checked_add(
                widen
                    .output_bytes
                    .ok_or((Reject::UnknownResourceEstimate, "power.mixed.widen.output"))?,
            )
            .ok_or((Reject::ResourceLimit, "power.mixed.widen.bytes"))?;
        if widen.dispatches != Some(0) {
            return Err((
                Reject::UnknownResourceEstimate,
                "power.mixed.widen.dispatch",
            ));
        }
        unknown_peak |= widen.transient_bytes.is_none();
    }
    Ok(MixedPhysicalWork {
        temporary_bytes,
        dispatches: 0,
        unknown_peak,
    })
}

fn cast_query(
    input: ProviderRepresentation,
    target: ProviderPrecision,
) -> ProviderFeasibilityQuery {
    let identity = match target {
        ProviderPrecision::F32 => runmat_builtins::SINGLE_CATALOG_ENTRY.identity,
        ProviderPrecision::F64 => runmat_builtins::DOUBLE_CATALOG_ENTRY.identity,
    };
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(identity.name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![input],
        outputs: vec![],
        workload: ProviderWorkload {
            preparation_route: Some(ProviderPreparationRoute::FloatingCast { target }),
            ..ProviderWorkload::default()
        },
    }
}

#[cfg(feature = "wgpu")]
pub(super) fn physical_work(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
    public_output: &ProviderRepresentation,
    native_words: u64,
) -> Result<MixedPhysicalWork, (ProviderRejectionCode, &'static str)> {
    use ProviderRejectionCode as Reject;
    let resident = query
        .inputs
        .iter()
        .find(|input| input.residency == ProviderResidency::Device)
        .ok_or((Reject::InvalidShape, "power.mixed.resident.required"))?;
    let elements = resident
        .checked_element_count()
        .ok_or((Reject::InvalidShape, "power.mixed.resident.shape"))?;
    let resident_is_single = resident.element_type == ProviderElementType::F32;
    let widen = if resident_is_single {
        Some(
            super::super::floating_conversion::physical_work(
                provider,
                &cast_query(resident.clone(), ProviderPrecision::F64),
            )
            .ok_or((
                Reject::UnknownResourceEstimate,
                "power.mixed.widen.unproven",
            ))?,
        )
    } else {
        None
    };
    let native_output = ProviderRepresentation {
        element_type: if public_output.storage == ProviderStorage::DenseComplexInterleaved {
            ProviderElementType::ComplexF64
        } else {
            ProviderElementType::F64
        },
        storage: public_output.storage,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: resident.shape.clone(),
        residency: ProviderResidency::Device,
    };
    let final_cast = super::super::floating_conversion::physical_work(
        provider,
        &cast_query(native_output, ProviderPrecision::F32),
    )
    .ok_or((
        Reject::UnknownResourceEstimate,
        "power.mixed.round.unproven",
    ))?;
    if final_cast.scalar_visits != native_words {
        return Err((Reject::UnknownResourceEstimate, "power.mixed.round.visits"));
    }
    let width = provider
        .elementwise_geometry()
        .map(|geometry| geometry.workgroup_size)
        .ok_or((Reject::UnknownResourceEstimate, "power.mixed.fill.geometry"))?;
    // `fill_buffer_demand` uses the same capacity and FillParamsF64 ABI. The
    // query has no scalar bits, so this budgets the shader even for +0, whose
    // actual constructor can use a cheaper clear path; -0 must use the shader.
    let fill =
        crate::backend::wgpu_geometry::fill_physical_work(elements, width, ProviderPrecision::F64)
            .ok_or((Reject::ResourceLimit, "power.mixed.fill.physical_work"))?;
    let native_output_bytes = crate::backend::wgpu_storage::raw_word_requested_bytes(
        native_words
            .checked_mul(2)
            .ok_or((Reject::ResourceLimit, "power.mixed.native.words"))?,
    )
    .ok_or((Reject::ResourceLimit, "power.mixed.native.output"))?;
    let cast_output = final_cast
        .estimate
        .output_bytes
        .ok_or((Reject::UnknownResourceEstimate, "power.mixed.round.output"))?;
    let public_logical = public_output
        .checked_byte_len()
        .ok_or((Reject::ResourceLimit, "power.mixed.public.output"))?;
    let mut extra = fill
        .total_bytes()
        .and_then(|bytes| bytes.checked_add(native_output_bytes))
        .and_then(|bytes| bytes.checked_add(final_cast.uniform_bytes))
        .and_then(|bytes| bytes.checked_add(cast_output.saturating_sub(public_logical)))
        .ok_or((Reject::ResourceLimit, "power.mixed.bytes.overflow"))?;
    let mut dispatches = fill
        .dispatches
        .checked_add(final_cast.estimate.dispatches.unwrap_or(0))
        .ok_or((Reject::ResourceLimit, "power.mixed.dispatch.overflow"))?;
    let mut unknown_peak = final_cast.estimate.transient_bytes.is_none();
    if let Some(widen) = widen {
        extra = extra
            .checked_add(
                widen
                    .estimate
                    .output_bytes
                    .ok_or((Reject::UnknownResourceEstimate, "power.mixed.widen.output"))?,
            )
            .and_then(|bytes| bytes.checked_add(widen.uniform_bytes))
            .ok_or((Reject::ResourceLimit, "power.mixed.widen.bytes"))?;
        dispatches = dispatches
            .checked_add(widen.estimate.dispatches.unwrap_or(0))
            .ok_or((Reject::ResourceLimit, "power.mixed.widen.dispatch"))?;
        unknown_peak |= widen.estimate.transient_bytes.is_none();
    }
    // The native input and the filled scalar retain the original resident's
    // logical shape. Empty resident views can refer to a larger allocation;
    // shape alone cannot certify its peak lifetime.
    unknown_peak |= elements == 0;
    Ok(MixedPhysicalWork {
        temporary_bytes: extra,
        dispatches,
        unknown_peak,
    })
}
