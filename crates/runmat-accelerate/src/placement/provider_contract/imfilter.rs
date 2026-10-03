//! Provider-owned direct imfilter shader preflight. It is intentionally one
//! physical hook for both public filter2 and imfilter semantics; the runtime
//! owns their input preparation, error mapping and fallback publication.

use super::*;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{
    observation::{ProviderObservationCapability, ProviderObservationOverhead},
    ProviderImfilterPhysicalEvidence, ProviderImfilterPhysicalRoute, ProviderWorkload,
};
#[cfg(feature = "wgpu")]
use runmat_execution::{ProviderInputStageDemand, ProviderInputStageDemands, ResourceByteBound};

#[cfg(feature = "wgpu")]
mod host;

#[cfg(feature = "wgpu")]
struct HookPlan {
    estimate: ProviderResourceEstimate,
    physical: ProviderImfilterPhysicalEvidence,
    submissions: u32,
}

#[cfg(feature = "wgpu")]
struct ValidatedImfilterQuery<'a> {
    workload: &'a runmat_accelerate_api::ProviderImfilterWorkload,
    image: &'a ProviderRepresentation,
    kernel: &'a ProviderRepresentation,
    output: &'a ProviderRepresentation,
    image_shape: Vec<usize>,
    kernel_shape: Vec<usize>,
    borrowed_bytes: u64,
    planned_kernel: bool,
}

#[cfg(feature = "wgpu")]
fn validated_query<'a>(
    provider: &(impl AccelProvider + ?Sized),
    query: &'a ProviderFeasibilityQuery,
) -> Option<ValidatedImfilterQuery<'a>> {
    if query.family != ProviderOperationFamily::Library
        || query.inputs.len() != 2
        || query.outputs.len() != 1
        || query.workload.requested_outputs != Some(1)
    {
        return None;
    }
    let workload = query.workload.imfilter.as_ref()?;
    let mut expected_workload = ProviderWorkload::default();
    expected_workload.imfilter = Some(workload.clone());
    expected_workload.requested_outputs = Some(1);
    if query.workload != expected_workload {
        return None;
    }
    let (borrowed_bytes, planned_kernel) = match (
        workload.input_backings.as_ref(),
        workload.prepared_kernel.as_ref(),
    ) {
        (Some(backings), None)
            if backings.live_device_ids()? == [provider.device_id(); 2]
                && backings.live_representations()?.as_slice() == query.inputs.as_slice() =>
        {
            (backings.live_unique_requested_bytes()?, false)
        }
        (None, Some(prepared))
            if prepared.live_device_id()? == provider.device_id()
                && prepared.live_representations()?.as_slice() == query.inputs.as_slice() =>
        {
            (prepared.live_image_requested_bytes()?, true)
        }
        _ => return None,
    };
    let [image, kernel] = query.inputs.as_slice() else {
        return None;
    };
    for input in [image, kernel] {
        if input.residency != ProviderResidency::Device
            || input.storage != ProviderStorage::DenseReal
            || !matches!(
                input.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
            || !matches!(
                input.layout,
                ProviderLayout::ColumnMajorContiguous | ProviderLayout::Strided
            )
        {
            return None;
        }
    }
    let shape = |representation: &ProviderRepresentation| {
        representation
            .shape
            .iter()
            .map(|&extent| usize::try_from(extent).ok())
            .collect::<Option<Vec<_>>>()
    };
    let image_shape = shape(image)?;
    let kernel_shape = shape(kernel)?;
    let expected_shape = runmat_runtime::builtins::image::filters::imfilter::filter_output_shape(
        &image_shape,
        &kernel_shape,
        workload.shape,
        "imfilter",
    )
    .ok()?
    .into_iter()
    .map(|extent| u64::try_from(extent).ok())
    .collect::<Option<Vec<_>>>()?;
    let output = &query.outputs[0];
    if output.element_type != image.element_type
        || output.storage != ProviderStorage::DenseReal
        || output.layout != ProviderLayout::ColumnMajorContiguous
        || output.residency != ProviderResidency::Device
        || output.shape != expected_shape
    {
        return None;
    }
    Some(ValidatedImfilterQuery {
        workload,
        image,
        kernel,
        output,
        image_shape,
        kernel_shape,
        borrowed_bytes,
        planned_kernel,
    })
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot, wgpu: bool) {
    if wgpu {
        snapshot.operations.push(ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new("imfilter"),
            family: ProviderOperationFamily::Library,
        });
    }
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.operation.0 != "imfilter" {
        return None;
    }
    if !wgpu {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::UnknownResourceEstimate,
            "imfilter.in_process_unobserved",
        ));
    }
    #[cfg(feature = "wgpu")]
    {
        return Some(match planned_route(provider, query) {
            Some(plan) => ProviderFeasibility::supported(plan.estimate),
            None => ProviderFeasibility::rejected(
                ProviderRejectionCode::UnknownResourceEstimate,
                "imfilter.physical_route_unproven",
            ),
        });
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = provider;
        Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "imfilter.no_wgpu_recipe",
        ))
    }
}

#[cfg(feature = "wgpu")]
pub(crate) fn physical_evidence(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderImfilterPhysicalEvidence> {
    planned_route(provider, query).map(|plan| plan.physical)
}

#[cfg(feature = "wgpu")]
pub(crate) fn observation_capability(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> ProviderObservationCapability {
    let Some(plan) = planned_route(provider, query) else {
        return ProviderObservationCapability::Unsupported;
    };
    let Some(requested_peak_buffer_bytes) = u64::from(plan.submissions)
        .checked_mul(crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES)
    else {
        return ProviderObservationCapability::Unsupported;
    };
    ProviderObservationCapability::Supported(ProviderObservationOverhead {
        submissions: plan.submissions,
        requested_peak_buffer_bytes,
        clear_commands: plan.submissions,
        map_callbacks: plan.submissions,
    })
}

#[cfg(feature = "wgpu")]
fn planned_route(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Option<HookPlan> {
    direct_plan(provider, query).or_else(|| host::plan(provider, query))
}

#[cfg(feature = "wgpu")]
pub(crate) fn shader_disabled() -> bool {
    std::env::var("RUNMAT_WGPU_DISABLE_IMFILTER")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
}

#[cfg(feature = "wgpu")]
fn direct_plan(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Option<HookPlan> {
    let ValidatedImfilterQuery {
        workload,
        image,
        kernel,
        output,
        image_shape,
        kernel_shape,
        borrowed_bytes,
        planned_kernel,
    } = validated_query(provider, query)?;
    let geometry = provider.imfilter_geometry()?;
    if geometry.max_rank == 0
        || geometry.workgroup_size == 0
        || geometry.max_workgroups_per_dispatch == 0
        || shader_disabled()
    {
        return None;
    }
    let rank = image_shape.len().max(kernel_shape.len()).max(2);
    let image_len = image.checked_element_count()?;
    let kernel_len = kernel.checked_element_count()?;
    let output_len = output.checked_element_count()?;
    let prepared_kernel_bytes = if planned_kernel {
        kernel_len
            .max(1)
            .checked_mul(kernel.element_type.byte_width())?
    } else {
        0
    };
    if rank > geometry.max_rank
        || image_len > u64::from(u32::MAX)
        || kernel_len > u64::from(u32::MAX)
        || output_len > u64::from(u32::MAX)
        || kernel_shape
            .iter()
            .any(|&extent| extent > i32::MAX as usize)
    {
        return None;
    }
    let accumulation_f64 =
        image.element_type == ProviderElementType::F64 || !workload.explicit_image;
    if accumulation_f64 && !geometry.shader_f64 {
        return None;
    }
    let element_width = image.element_type.byte_width();
    let accumulation_width = if accumulation_f64 { 8_u64 } else { 4_u64 };
    let output_bytes = output_len.max(1).checked_mul(element_width)?;
    let offsets_bytes = kernel_len
        .max(1)
        .checked_mul(u64::try_from(rank).ok()?)?
        .checked_mul(4)?;
    let values_bytes = kernel_len.max(1).checked_mul(accumulation_width)?;
    let kernel_readback_bytes = if planned_kernel {
        prepared_kernel_bytes
    } else {
        workload.input_backings.as_ref()?.live_requested_bytes()?[1]
    };
    let lazy_image_bytes = if image.layout == ProviderLayout::Strided {
        match workload.input_backings.as_ref() {
            Some(backings) => backings.live_requested_bytes()?[0],
            None => workload
                .prepared_kernel
                .as_ref()?
                .live_image_requested_bytes()?,
        }
    } else {
        0
    };
    let chunk_capacity = u64::from(geometry.max_workgroups_per_dispatch)
        .checked_mul(u64::from(geometry.workgroup_size))?;
    let chunks = if output_len == 0 {
        0
    } else {
        output_len.div_ceil(chunk_capacity)
    };
    let uniform_width = if accumulation_f64 {
        std::mem::size_of::<crate::backend::wgpu::params::ImfilterParamsF64>() as u64
    } else {
        std::mem::size_of::<crate::backend::wgpu::params::ImfilterParamsF32>() as u64
    };
    let uniform_bytes = chunks.checked_mul(uniform_width)?;
    for bytes in [
        output_bytes,
        offsets_bytes,
        values_bytes,
        uniform_width,
        prepared_kernel_bytes,
    ] {
        if bytes > geometry.max_buffer_bytes {
            return None;
        }
    }
    for bytes in [output_bytes, offsets_bytes, values_bytes] {
        if bytes > geometry.max_binding_bytes {
            return None;
        }
    }
    let peak = [
        borrowed_bytes,
        prepared_kernel_bytes,
        output_bytes,
        offsets_bytes,
        values_bytes,
        uniform_bytes,
        lazy_image_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| total.checked_add(bytes))?;
    let native_kernel_bytes = kernel_len.checked_mul(kernel.element_type.byte_width())?;
    let rank_bytes = u64::try_from(rank).ok()?.checked_mul(32)?.checked_mul(8)?;
    let point_bytes = kernel_len.checked_mul(
        u64::try_from(std::mem::size_of::<
            runmat_runtime::builtins::image::filters::imfilter::ImfilterKernelPoint,
        >())
        .ok()?
        .checked_add(u64::try_from(rank).ok()?.checked_mul(8)?)?,
    )?;
    let host_workspace = [
        native_kernel_bytes.checked_mul(if kernel.layout == ProviderLayout::Strided {
            2
        } else {
            1
        })?,
        if kernel.element_type == ProviderElementType::F32 {
            kernel_len.checked_mul(8)?
        } else {
            0
        },
        rank_bytes,
        point_bytes,
        kernel_len
            .checked_mul(u64::try_from(rank).ok()?)?
            .checked_mul(4)?,
        kernel_len.checked_mul(8)?,
        if accumulation_f64 {
            0
        } else {
            kernel_len.checked_mul(4)?
        },
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| total.checked_add(bytes))?;
    let readback_peak = prepared_kernel_bytes.checked_add(kernel_readback_bytes)?;
    let input_stages = ProviderInputStageDemands::sequential([ProviderInputStageDemand {
        device_id: provider.device_id(),
        incremental_peak_bytes: ResourceByteBound::Known(readback_peak),
        queue_slots: 1,
    }])?;
    let output_queue_slots =
        u32::try_from(chunks.checked_add(u64::from(image.layout == ProviderLayout::Strided))?)
            .ok()?;
    let submissions = output_queue_slots.checked_add(1)?;
    Some(HookPlan {
        estimate: ProviderResourceEstimate {
            transient_bytes: Some(peak),
            output_bytes: Some(output_bytes),
            dispatches: Some(submissions),
        },
        physical: ProviderImfilterPhysicalEvidence {
            route: ProviderImfilterPhysicalRoute::Direct,
            host_workspace: ResourceByteBound::Known(host_workspace),
            input_stages,
            output_queue_slots,
            prepared_kernel_requested_bytes: planned_kernel.then_some(prepared_kernel_bytes),
        },
        submissions,
    })
}
