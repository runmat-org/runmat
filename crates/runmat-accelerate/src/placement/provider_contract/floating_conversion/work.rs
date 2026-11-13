//! Physical route of the existing typed cast, not a separate cost scheduler.
use super::*;

pub(crate) struct FloatingCastWork {
    pub estimate: ProviderResourceEstimate,
    #[cfg(any(feature = "wgpu", test))]
    pub source: ProviderPrecision,
    #[cfg(any(feature = "wgpu", test))]
    pub target: ProviderPrecision,
    #[cfg(any(feature = "wgpu", test))]
    pub workgroup: u32,
    #[cfg(any(feature = "wgpu", test))]
    pub allocations: u32,
    #[cfg(any(feature = "wgpu", test))]
    pub uniform_bytes: u64,
    #[cfg(any(feature = "wgpu", test))]
    pub scratch_bytes: u64,
    #[cfg(any(feature = "wgpu", test))]
    pub scalar_visits: u64,
}

pub(super) fn build(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
    target: ProviderElementType,
) -> Option<FloatingCastWork> {
    let input = query.inputs.first()?;
    build_for_input(provider, input, target)
}

/// Shared ordinary cast allocator/chunk work; semantic feasibility remains
/// the caller's separate typed admission obligation.
pub(crate) fn build_for_input(
    provider: &(impl AccelProvider + ?Sized),
    input: &ProviderRepresentation,
    target: ProviderElementType,
) -> Option<FloatingCastWork> {
    let elements = input.checked_element_count()?;
    let scalar_width = if input.storage == ProviderStorage::DenseComplexInterleaved {
        2u64
    } else {
        1
    };
    let scalar_visits = elements.checked_mul(scalar_width)?;
    let output_type = match (input.storage, target) {
        (ProviderStorage::DenseReal, target) => target,
        (ProviderStorage::DenseComplexInterleaved, ProviderElementType::F32) => {
            ProviderElementType::ComplexF32
        }
        (ProviderStorage::DenseComplexInterleaved, ProviderElementType::F64) => {
            ProviderElementType::ComplexF64
        }
        _ => return None,
    };
    let workgroup = provider.elementwise_geometry()?.workgroup_size;
    let capacity = crate::backend::wgpu_geometry::elementwise_chunk_capacity(workgroup)?;
    let same = input.element_type == output_type;
    let dispatches = if same || scalar_visits == 0 {
        0
    } else {
        u32::try_from(scalar_visits.div_ceil(capacity)).ok()?
    };
    let source_bytes = input.checked_byte_len()?;
    let output_payload = elements.checked_mul(output_type.byte_width())?;
    let output_requested = if same {
        0
    } else {
        crate::backend::wgpu_storage::raw_word_requested_bytes(output_payload.checked_div(4)?)?
    };
    let uniform_bytes = u64::from(dispatches).checked_mul(16)?;
    // A known host tensor upload is a mapped initialization, not a queue
    // submission. Empty host values never reach promotion in native_auto.
    #[cfg(any(feature = "wgpu", test))]
    let uploaded = input.residency == ProviderResidency::Host && elements != 0;
    #[cfg(any(feature = "wgpu", test))]
    let scratch_bytes =
        uniform_bytes.checked_add(if uploaded && !same { source_bytes } else { 0 })?;
    // Shape cannot reveal an empty resident view's backing allocation. Leave
    // total footprint unknown; fresh requested output/params remain explicit.
    let transient_bytes = if input.residency == ProviderResidency::Device && elements == 0 {
        None
    } else {
        source_bytes
            .checked_add(output_requested)?
            .checked_add(uniform_bytes)
    };
    Some(FloatingCastWork {
        estimate: ProviderResourceEstimate {
            transient_bytes,
            output_bytes: Some(if same { source_bytes } else { output_requested }),
            dispatches: Some(dispatches),
        },
        #[cfg(any(feature = "wgpu", test))]
        source: match input.element_type {
            ProviderElementType::F32 | ProviderElementType::ComplexF32 => ProviderPrecision::F32,
            ProviderElementType::F64 | ProviderElementType::ComplexF64 => ProviderPrecision::F64,
            _ => return None,
        },
        #[cfg(any(feature = "wgpu", test))]
        target: match target {
            ProviderElementType::F32 => ProviderPrecision::F32,
            ProviderElementType::F64 => ProviderPrecision::F64,
            _ => return None,
        },
        #[cfg(any(feature = "wgpu", test))]
        workgroup,
        #[cfg(any(feature = "wgpu", test))]
        allocations: u32::from(uploaded)
            .checked_add(u32::from(!same))?
            .checked_add(dispatches)?,
        #[cfg(any(feature = "wgpu", test))]
        uniform_bytes,
        #[cfg(any(feature = "wgpu", test))]
        scratch_bytes,
        #[cfg(any(feature = "wgpu", test))]
        scalar_visits: if dispatches == 0 { 0 } else { scalar_visits },
    })
}

#[cfg(feature = "wgpu")]
pub(crate) fn physical_work(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Option<FloatingCastWork> {
    let target = super::target(query)?;
    if !super::has_exact_route(query, target) {
        return None;
    }
    if !super::validate_target(provider, true, query, target)?.is_supported() {
        return None;
    }
    build(provider, query, target)
}

#[cfg(feature = "wgpu")]
pub(crate) fn observation_capability(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> runmat_accelerate_api::observation::ProviderObservationCapability {
    use runmat_accelerate_api::observation::{
        ProviderObservationCapability, ProviderObservationOverhead,
    };
    // Only the reviewed single target is enabled. This is the same typed target
    // authority as numerical feasibility, not a second builtin spelling table.
    if super::target(query) != Some(ProviderElementType::F32) {
        return ProviderObservationCapability::Unsupported;
    }
    let Some(work) = physical_work(provider, query) else {
        return ProviderObservationCapability::Unsupported;
    };
    let submissions = work.estimate.dispatches.unwrap_or(0);
    ProviderObservationCapability::Supported(ProviderObservationOverhead {
        submissions,
        requested_peak_buffer_bytes: u64::from(submissions)
            * crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES,
        clear_commands: submissions,
        map_callbacks: submissions,
        wgpu_compute_query: crate::backend::wgpu::dispatch::observation::prospective_encoder_query(
            submissions,
            submissions,
        ),
    })
}
