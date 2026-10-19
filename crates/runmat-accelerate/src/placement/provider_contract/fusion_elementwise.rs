//! Physical execution geometry for the typed floating fusion route. Source
//! representations remain source facts; staging buffers and output adapters
//! are counted separately instead of pretending the kernel writes F32 in an
//! F64 allocation.
use super::*;
#[cfg(test)]
pub(crate) mod fixture;

/// Shared costing consumes the same physical plan as feasibility. This is a
/// prior for the complete route, not timing evidence from one kernel enqueue.
#[cfg(any(feature = "wgpu", test))]
pub(crate) fn physical_work(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> Option<PhysicalWork> {
    analyze(provider, true, query)?.ok()
}

pub(crate) struct PhysicalWork {
    pub estimate: ProviderResourceEstimate,
    #[cfg(any(feature = "wgpu", test))]
    pub scratch_bytes: u64,
    #[cfg(any(feature = "wgpu", test))]
    pub queue_submissions: u32,
    /// Numerical/adapter encoders plus actual no-op compute warmups; excludes
    /// the one noncompute flush in each materialization setup pair.
    #[cfg(any(feature = "wgpu", test))]
    pub compute_encoders: u32,
    /// Logical element visits in host staging and device representation
    /// adapters, excluding the primary fused program and transfer bytes.
    #[cfg(any(feature = "wgpu", test))]
    pub adapter_element_visits: u64,
}

/// This route's complete submission geometry includes staging and output casts,
/// chunks and multioutput kernels. Other provider operations are not inferred to
/// have observation coverage from sharing an operation family.
#[cfg(feature = "wgpu")]
pub(crate) fn observation_capability(
    provider: &(impl AccelProvider + ?Sized),
    query: &ProviderFeasibilityQuery,
) -> runmat_accelerate_api::observation::ProviderObservationCapability {
    use runmat_accelerate_api::observation::{
        ProviderObservationCapability, ProviderObservationOverhead,
    };
    let Some(work) = physical_work(provider, query) else {
        return ProviderObservationCapability::Unsupported;
    };
    ProviderObservationCapability::Supported(ProviderObservationOverhead {
        submissions: work.queue_submissions,
        requested_peak_buffer_bytes: u64::from(work.queue_submissions)
            * crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES,
        clear_commands: work.queue_submissions,
        map_callbacks: work.queue_submissions,
        wgpu_compute_query: crate::backend::wgpu::dispatch::observation::prospective_encoder_query(
            work.compute_encoders,
            work.queue_submissions,
        ),
    })
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    analyze(provider, wgpu, query).map(|work| match work {
        Ok(work) => ProviderFeasibility::supported(work.estimate),
        Err(rejected) => rejected,
    })
}

fn analyze(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<Result<PhysicalWork, ProviderFeasibility>> {
    if query.operation.0 != fusion_operation_token(&FusionKind::ElementwiseChain) {
        return None;
    }
    let reject = |code, reason| Some(Err(ProviderFeasibility::rejected(code, reason)));
    if !wgpu {
        // InProcessProvider's actual fused_elementwise hook is unsupported.
        // Its ordinary numeric kernels do not imply a WGSL interpreter.
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "fusion.elementwise.shader_backend_required",
        );
    }
    let Some(geometry) = provider
        .elementwise_geometry()
        .filter(|g| g.workgroup_size != 0)
    else {
        return reject(
            ProviderRejectionCode::UnknownResourceEstimate,
            "fusion.elementwise.unknown_dispatch_geometry",
        );
    };
    let dispatches = |elements| dispatches(elements, geometry.workgroup_size);
    if query.family != ProviderOperationFamily::Fusion
        || query.inputs.is_empty()
        || query.outputs.is_empty()
    {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "fusion.elementwise.arity",
        );
    }
    let Ok(output_shape) = query.outputs[0]
        .shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<Vec<_>, _>>()
    else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "fusion.elementwise.output_overflow",
        );
    };
    #[cfg(feature = "wgpu")]
    if output_shape.len() > crate::backend::wgpu::params::BCAST_MAX_RANK {
        return reject(
            ProviderRejectionCode::UnsupportedRank,
            "fusion.elementwise.broadcast_rank",
        );
    }
    for input in &query.inputs {
        if !matches!(
            input.residency,
            ProviderResidency::Host | ProviderResidency::Device
        ) {
            return reject(
                ProviderRejectionCode::UnsupportedOperation,
                "fusion.elementwise.unresolved_residency",
            );
        }
        let Ok(input_shape) = input
            .shape
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
        else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "fusion.elementwise.input_overflow",
            );
        };
        let broadcast = runmat_runtime::builtins::common::broadcast::broadcast_shapes(
            "fusion.elementwise",
            &input_shape,
            &output_shape,
        );
        if !broadcast.is_ok_and(|shape| shape == output_shape) {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "fusion.elementwise.broadcast_shape",
            );
        }
    }
    let typed = query.workload.fused_elementwise;
    if typed.is_some_and(|work| !provider.supports_fused_elementwise_abi(work.abi)) {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "fusion.elementwise.typed_abi_unsupported",
        );
    }
    let native =
        ProviderElementType::from(typed.map_or(provider.precision(), |work| work.abi.precision()));
    let host_elements = query
        .inputs
        .iter()
        .filter(|input| input.residency == ProviderResidency::Host)
        .try_fold(0_u64, |count, input| {
            count.checked_add(input.checked_element_count()?)
        });
    if typed.is_some_and(|work| {
        work.checked_host_workspace_bytes().is_none()
            || host_elements.is_none_or(|elements| work.host_materialization_elements > elements)
            || query
                .inputs
                .iter()
                .chain(&query.outputs)
                .any(|value| value.element_type != native)
    }) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "fusion.elementwise.typed_preparation_invalid",
        );
    }
    let mut physical = query.clone();
    physical.inputs.clear();
    physical.outputs.clear();
    let mut adapter_dispatches = 0u64;
    let mut adapter_element_visits = typed.map_or(0, |work| work.host_materialization_elements);
    let mut final_output_bytes = 0u64;
    // Cost scratch includes host workspace as well as device adapters; the
    // device feasibility estimate below still comes from actual GPU buffers.
    let mut scratch_bytes = typed.map_or(0, |work| work.host_workspace_bytes);
    let mut materialization_setup_submissions = 0u32;
    let mut shape = None;
    for input in &query.inputs {
        let strided =
            input.residency == ProviderResidency::Device && input.layout == ProviderLayout::Strided;
        let materialize = strided && input.element_type != native;
        if strided && input.shape.len() != 2 {
            return reject(
                ProviderRejectionCode::UnsupportedRank,
                "fusion.elementwise.strided_matrix_required",
            );
        }
        if input.layout != ProviderLayout::ColumnMajorContiguous && !strided {
            return reject(
                ProviderRejectionCode::UnsupportedLayout,
                "fusion.elementwise.layout",
            );
        }
        if input.storage != ProviderStorage::DenseReal
            || !matches!(
                input.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
        {
            return reject(
                ProviderRejectionCode::UnsupportedStorage,
                "fusion.elementwise.real_float_input",
            );
        }
        let cast = input.residency == ProviderResidency::Device && input.element_type != native;
        let Some(elements) = input.checked_element_count() else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "fusion.elementwise.input_overflow",
            );
        };
        // Host tensors materialize an f64 staging vector; F32 providers then
        // convert that vector to f32 before upload. Resident inputs instead
        // execute one pass per selected materialization/widening adapter.
        let passes = if typed.is_some() && input.residency == ProviderResidency::Host {
            0 // Already described by the immutable host view projection above.
        } else if input.residency == ProviderResidency::Host {
            1 + u64::from(native == ProviderElementType::F32)
        } else {
            u64::from(materialize) + u64::from(cast)
        };
        let Some(total) = elements
            .checked_mul(passes)
            .and_then(|visits| adapter_element_visits.checked_add(visits))
        else {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "fusion.elementwise.adapter_work_overflow",
            );
        };
        adapter_element_visits = total;
        let mut physical_input = input.clone();
        physical_input.layout = ProviderLayout::ColumnMajorContiguous;
        if materialize {
            // The underlying original allocation stays live; the logical
            // strided view is materialized into a separate physical buffer.
            physical.inputs.push(physical_input.clone());
            let Some(bytes) =
                crate::backend::wgpu_storage::checked_allocation_bytes(&physical_input)
            else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.materialize_bytes",
                );
            };
            let Some(total) = scratch_bytes.checked_add(bytes) else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.scratch_overflow",
                );
            };
            scratch_bytes = total;
            let Some(count) = input.checked_element_count().and_then(dispatches) else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.materialize_dispatches",
                );
            };
            let Some(total) = adapter_dispatches.checked_add(count) else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.dispatch_overflow",
                );
            };
            adapter_dispatches = total;
            if count != 0 {
                // Existing permute submits a no-op pass and a flush before
                // its chunked copy dispatches. These are queue work, not
                // additional numerical kernel invocations.
                let Some(total) = materialization_setup_submissions.checked_add(2) else {
                    return reject(
                        ProviderRejectionCode::ResourceLimit,
                        "fusion.elementwise.queue_overflow",
                    );
                };
                materialization_setup_submissions = total;
            }
        }
        if cast {
            if provider.precision() != ProviderPrecision::F64
                || input.element_type != ProviderElementType::F32
            {
                return reject(
                    ProviderRejectionCode::UnsupportedElementType,
                    "fusion.elementwise.resident_cast",
                );
            }
            // The original allocation remains borrowed and live while its
            // widened staging allocation is used by the native-precision ABI.
            physical.inputs.push(physical_input.clone());
            let Some(count) = input.checked_element_count() else {
                return reject(
                    ProviderRejectionCode::InvalidShape,
                    "fusion.elementwise.input_overflow",
                );
            };
            let Some(dispatches) = dispatches(count) else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.dispatch_overflow",
                );
            };
            adapter_dispatches = match adapter_dispatches.checked_add(dispatches) {
                Some(value) => value,
                None => {
                    return reject(
                        ProviderRejectionCode::ResourceLimit,
                        "fusion.elementwise.dispatch_overflow",
                    )
                }
            };
        }
        let mut staged = physical_input;
        staged.element_type = native;
        staged.residency = ProviderResidency::Device;
        if input.residency == ProviderResidency::Host || cast {
            let Some(bytes) = crate::backend::wgpu_storage::checked_allocation_bytes(&staged)
            else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.staging_bytes",
                );
            };
            let Some(total) = scratch_bytes.checked_add(bytes) else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.scratch_overflow",
                );
            };
            scratch_bytes = total;
        }
        physical.inputs.push(staged);
    }
    for output in &query.outputs {
        if output.storage != ProviderStorage::DenseReal
            || output.layout != ProviderLayout::ColumnMajorContiguous
            || output.residency != ProviderResidency::Device
            || !matches!(
                output.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
            || (native == ProviderElementType::F32
                && output.element_type != ProviderElementType::F32)
        {
            return reject(
                ProviderRejectionCode::UnsupportedElementType,
                "fusion.elementwise.output_representation",
            );
        }
        if shape.is_some_and(|shape| shape != &output.shape) {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "fusion.elementwise.output_shapes",
            );
        }
        shape = Some(&output.shape);
        let Some(bytes) = crate::backend::wgpu_storage::checked_allocation_bytes(output) else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "fusion.elementwise.output_overflow",
            );
        };
        final_output_bytes = match final_output_bytes.checked_add(bytes) {
            Some(value) => value,
            None => {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.bytes_overflow",
                )
            }
        };
        let mut raw = output.clone();
        raw.element_type = native;
        physical.outputs.push(raw);
        if output.element_type != native {
            let Some(bytes) = crate::backend::wgpu_storage::checked_allocation_bytes(
                physical.outputs.last().expect("raw output added"),
            ) else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.raw_output_bytes",
                );
            };
            let Some(total) = scratch_bytes.checked_add(bytes) else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.scratch_overflow",
                );
            };
            scratch_bytes = total;
            // Transactional publication keeps every raw output alive until
            // every final adapter has completed; peak includes both sets.
            physical.outputs.push(output.clone());
            let Some(count) = output.checked_element_count() else {
                return reject(
                    ProviderRejectionCode::InvalidShape,
                    "fusion.elementwise.output_overflow",
                );
            };
            let Some(total) = adapter_element_visits.checked_add(count) else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.adapter_work_overflow",
                );
            };
            adapter_element_visits = total;
            let Some(dispatches) = dispatches(count) else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.dispatch_overflow",
                );
            };
            adapter_dispatches = match adapter_dispatches.checked_add(dispatches) {
                Some(value) => value,
                None => {
                    return reject(
                        ProviderRejectionCode::ResourceLimit,
                        "fusion.elementwise.dispatch_overflow",
                    )
                }
            };
        }
    }
    // The execution owner exits to the shared runtime before any preparation
    // for empty output shapes, including broadcasts with nonempty scalars.
    if output_shape.contains(&0) {
        return Some(Ok(PhysicalWork {
            estimate: ProviderResourceEstimate {
                transient_bytes: Some(0),
                output_bytes: Some(0),
                dispatches: Some(0),
            },
            #[cfg(any(feature = "wgpu", test))]
            scratch_bytes: 0,
            #[cfg(any(feature = "wgpu", test))]
            queue_submissions: 0,
            #[cfg(any(feature = "wgpu", test))]
            compute_encoders: 0,
            #[cfg(any(feature = "wgpu", test))]
            adapter_element_visits: 0,
        }));
    }
    let mut snapshot = dense_capability_snapshot(
        provider,
        1,
        &[(
            fusion_operation_token(&FusionKind::ElementwiseChain),
            ProviderOperationFamily::Fusion,
        )],
    );
    if !snapshot.element_types.contains(&ProviderElementType::F32) {
        snapshot.element_types.push(ProviderElementType::F32);
    }
    let mut result = dense_feasibility_with_bytes(
        &snapshot,
        &physical,
        crate::backend::wgpu_storage::checked_allocation_bytes,
    );
    if let ProviderFeasibility::Supported { estimate } = &mut result {
        let Some(elements) = query.outputs[0].checked_element_count() else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "fusion.elementwise.output_overflow",
            );
        };
        let Some(dispatches) = dispatches(elements)
            .and_then(|kernel| kernel.checked_add(adapter_dispatches))
            .and_then(|total| u32::try_from(total).ok())
        else {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "fusion.elementwise.dispatch_overflow",
            );
        };
        estimate.output_bytes = Some(final_output_bytes);
        estimate.dispatches = Some(dispatches);
    }
    Some(match result {
        ProviderFeasibility::Supported { estimate } => {
            let Some(_queue_submissions) = estimate
                .dispatches
                .and_then(|dispatches| dispatches.checked_add(materialization_setup_submissions))
                // Both fused entrypoints submit one pipeline warmup pass even
                // on a cache hit, before the numerical chunk submissions.
                .and_then(|submissions| submissions.checked_add(1))
            else {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "fusion.elementwise.queue_overflow",
                );
            };
            // Each nonempty materialization contributes exactly a no-op
            // compute pass followed by a noncompute flush. Keep the warmup
            // compute pass, but subtract only those flushes from query demand.
            let _compute_encoders = _queue_submissions - materialization_setup_submissions / 2;
            Ok(PhysicalWork {
                estimate,
                #[cfg(any(feature = "wgpu", test))]
                scratch_bytes,
                #[cfg(any(feature = "wgpu", test))]
                queue_submissions: _queue_submissions,
                #[cfg(any(feature = "wgpu", test))]
                compute_encoders: _compute_encoders,
                #[cfg(any(feature = "wgpu", test))]
                adapter_element_visits,
            })
        }
        rejected => Err(rejected),
    })
}

fn dispatches(elements: u64, workgroup: u32) -> Option<u64> {
    if elements > u64::from(u32::MAX) {
        return None;
    }
    let capacity = crate::backend::wgpu_geometry::elementwise_chunk_capacity(workgroup)?;
    Some(elements.div_ceil(capacity))
}

#[cfg(test)]
mod tests;
