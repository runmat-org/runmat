//! Physical execution geometry for the typed floating fusion route. Source
//! representations remain source facts; staging buffers and output adapters
//! are counted separately instead of pretending the kernel writes F32 in an
//! F64 allocation.
use super::*;

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.operation.0 != "fusion.elementwise" {
        return None;
    }
    let reject = |code, reason| Some(ProviderFeasibility::rejected(code, reason));
    if !wgpu {
        // InProcessProvider's actual fused_elementwise hook is unsupported.
        // Its ordinary numeric kernels do not imply a WGSL interpreter.
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "fusion.elementwise.shader_backend_required",
        );
    }
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
    let native = ProviderElementType::from(provider.precision());
    let mut physical = query.clone();
    physical.inputs.clear();
    physical.outputs.clear();
    let mut adapter_dispatches = 0u64;
    let mut final_output_bytes = 0u64;
    let mut shape = None;
    for input in &query.inputs {
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
        if input.residency == ProviderResidency::Device && input.element_type != native {
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
            physical.inputs.push(input.clone());
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
        let mut staged = input.clone();
        staged.element_type = native;
        staged.residency = ProviderResidency::Device;
        physical.inputs.push(staged);
    }
    for output in &query.outputs {
        if output.storage != ProviderStorage::DenseReal
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
            // Transactional publication keeps every raw output alive until
            // every final adapter has completed; peak includes both sets.
            physical.outputs.push(output.clone());
            let Some(count) = output.checked_element_count() else {
                return reject(
                    ProviderRejectionCode::InvalidShape,
                    "fusion.elementwise.output_overflow",
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
    }
    let mut snapshot = dense_capability_snapshot(
        provider,
        1,
        &[("fusion.elementwise", ProviderOperationFamily::Fusion)],
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
    Some(result)
}

fn dispatches(elements: u64) -> Option<u64> {
    if elements > u64::from(u32::MAX) {
        return None;
    }
    #[cfg(feature = "wgpu")]
    let capacity = u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS)
        * u64::from(crate::backend::wgpu::config::effective_workgroup_size());
    #[cfg(not(feature = "wgpu"))]
    let capacity = u64::from(u32::MAX);
    Some(elements.div_ceil(capacity))
}

#[cfg(test)]
mod tests;
