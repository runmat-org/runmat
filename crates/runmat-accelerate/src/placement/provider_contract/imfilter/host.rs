//! Provider-owned physical route for WGPU imfilter branches that actually
//! execute native readback/host filtering/resident upload inside the hook.
//! A rejected shader estimate alone is not a fallback witness: only the
//! source-backed branch triggers below qualify this route.

use super::*;

pub(super) fn plan(
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
    let image_len = image.checked_element_count()?;
    let kernel_len = kernel.checked_element_count()?;
    let output_len = output.checked_element_count()?;
    let rank = image_shape.len().max(kernel_shape.len()).max(2);
    let accumulation_f64 =
        image.element_type == ProviderElementType::F64 || !workload.explicit_image;
    // The actual hook returns Unsupported before host mediation when shader
    // f64 is required but absent. This is the runtime's external fallback,
    // never a successful provider-host route.
    if accumulation_f64 && !geometry.shader_f64 {
        return None;
    }
    let disabled_early = shader_disabled();
    let rank_u64 = u64::try_from(rank).ok()?;
    let after_kernel_readback = rank > geometry.max_rank
        || image_len > u64::from(u32::MAX)
        || output_len > u64::from(u32::MAX)
        || kernel_len > u64::from(u32::MAX)
        || kernel_len.checked_mul(rank_u64)? > u64::from(u32::MAX)
        || kernel_shape
            .iter()
            .any(|&extent| extent > i32::MAX as usize);
    if !disabled_early && !after_kernel_readback {
        return None;
    }

    let image_width = image.element_type.byte_width();
    let kernel_width = kernel.element_type.byte_width();
    let image_payload = image_len.checked_mul(image_width)?;
    let kernel_payload = kernel_len.checked_mul(kernel_width)?;
    let output_payload = output_len.checked_mul(image_width)?;
    let output_bytes = output_len.max(1).checked_mul(image_width)?;
    if output_bytes > geometry.max_buffer_bytes {
        return None;
    }
    let prepared_kernel_bytes = if planned_kernel {
        kernel_len.max(1).checked_mul(kernel_width)?
    } else {
        0
    };
    if prepared_kernel_bytes > geometry.max_buffer_bytes {
        return None;
    }
    let [image_requested, kernel_requested] = match workload.input_backings.as_ref() {
        Some(backings) => backings.live_requested_bytes()?,
        None => [
            workload
                .prepared_kernel
                .as_ref()?
                .live_image_requested_bytes()?,
            prepared_kernel_bytes,
        ],
    };
    // Empty native downloads return an empty payload without a GPU staging
    // buffer or queue submission. Nonempty readbacks stage the source's real
    // physical extent. Prepared H remains allocated during later stages.
    let stage = |requested: u64, nonempty: bool| {
        prepared_kernel_bytes
            .checked_add(if nonempty { requested } else { 0 })?
            .checked_add(if nonempty {
                crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES
            } else {
                0
            })
    };
    let image_stage = stage(image_requested, image_len != 0)?;
    let kernel_stage = stage(kernel_requested, kernel_len != 0)?;
    let mut stages = Vec::with_capacity(3);
    if !disabled_early {
        stages.push(ProviderInputStageDemand {
            device_id: provider.device_id(),
            incremental_peak_bytes: ResourceByteBound::Known(kernel_stage),
            queue_slots: u32::from(kernel_len != 0),
        });
    }
    if image_len != 0 {
        stages.push(ProviderInputStageDemand {
            device_id: provider.device_id(),
            incremental_peak_bytes: ResourceByteBound::Known(image_stage),
            queue_slots: 1,
        });
    }
    if kernel_len != 0 {
        stages.push(ProviderInputStageDemand {
            device_id: provider.device_id(),
            incremental_peak_bytes: ResourceByteBound::Known(kernel_stage),
            queue_slots: 1,
        });
    }
    let input_stages = ProviderInputStageDemands::sequential(stages)?;
    let output_stage = prepared_kernel_bytes.checked_add(output_bytes)?;
    let transient_bytes =
        borrowed_bytes.checked_add(image_stage.max(kernel_stage).max(output_stage))?;

    // `download_numeric_exec` returns native A/H vectors; lazy 2-D views may
    // clone each once for transpose. A late fallback retains the first H
    // readback while downloading A and H again. The shared native evaluator
    // then owns rank vectors, kernel points, f64 arithmetic and the final
    // native output concurrently at its peak. All terms are checked bounds.
    let lazy_image = if image.layout == ProviderLayout::Strided {
        image_payload
    } else {
        0
    };
    let lazy_kernel = if kernel.layout == ProviderLayout::Strided {
        kernel_payload
    } else {
        0
    };
    let early_kernel = if disabled_early { 0 } else { kernel_payload };
    let point_width = u64::try_from(std::mem::size_of::<
        runmat_runtime::builtins::image::filters::kernel::ImfilterKernelPoint,
    >())
    .ok()?
    .checked_add(rank_u64.checked_mul(8)?)?;
    let rank_vectors = rank_u64.checked_mul(16)?.checked_mul(8)?;
    let host_workspace = [
        image_payload,
        kernel_payload,
        lazy_image,
        lazy_kernel,
        early_kernel,
        if image.element_type == ProviderElementType::F32 {
            image_len.checked_mul(8)?
        } else {
            0
        },
        if kernel.element_type == ProviderElementType::F32 {
            kernel_len.checked_mul(8)?
        } else {
            0
        },
        kernel_len.checked_mul(point_width)?,
        rank_vectors,
        output_len.checked_mul(8)?,
        if image.element_type == ProviderElementType::F32 {
            output_payload
        } else {
            0
        },
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| total.checked_add(bytes))?;
    let submissions = 1_u32
        .checked_add(u32::from(image_len != 0))?
        .checked_add(u32::from(kernel_len != 0))?
        .checked_add(u32::from(!disabled_early && kernel_len != 0))?;
    Some(HookPlan {
        estimate: ProviderResourceEstimate {
            transient_bytes: Some(transient_bytes),
            output_bytes: Some(output_bytes),
            dispatches: Some(submissions),
        },
        physical: ProviderImfilterPhysicalEvidence {
            route: if disabled_early {
                ProviderImfilterPhysicalRoute::HostMediatedEarly
            } else {
                ProviderImfilterPhysicalRoute::HostMediatedAfterKernelReadback
            },
            host_workspace: ResourceByteBound::Known(host_workspace),
            input_stages,
            output_queue_slots: 1,
            prepared_kernel_requested_bytes: planned_kernel.then_some(prepared_kernel_bytes),
        },
        submissions,
        compute_encoders: 0,
    })
}
