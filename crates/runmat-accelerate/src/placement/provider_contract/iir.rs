//! Side-effect-free feasibility for one prepared resident IIR pass.
use runmat_accelerate_api::{
    AccelProvider, ProviderElementType, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderIirRoute, ProviderLayout, ProviderOperationFamily, ProviderRejectionCode,
    ProviderResidency, ProviderResourceEstimate, ProviderStorage,
    PROVIDER_IIR_CAUSAL_PASS_OPERATION, PROVIDER_IIR_PREPARED_PASS_OPERATION,
    PROVIDER_IIR_ZERO_PHASE_OPERATION,
};

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.operation.0 == PROVIDER_IIR_CAUSAL_PASS_OPERATION
        || query
            .workload
            .iir
            .is_some_and(|workload| workload.route == ProviderIirRoute::CausalPass)
    {
        return Some(validate_causal(provider, wgpu, query));
    }
    if query.operation.0 == PROVIDER_IIR_ZERO_PHASE_OPERATION
        || query
            .workload
            .iir
            .is_some_and(|workload| workload.route == ProviderIirRoute::ZeroPhase)
    {
        return Some(validate_zero_phase(provider, wgpu, query));
    }
    let Some(workload) = query.workload.iir else {
        return (query.operation.0 == PROVIDER_IIR_PREPARED_PASS_OPERATION).then(|| {
            ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedOperation,
                "iir.prepared_pass.missing_typed_route",
            )
        });
    };
    if workload.route != ProviderIirRoute::PreparedPass
        || query.family != ProviderOperationFamily::Library
        || query.operation.0 != PROVIDER_IIR_PREPARED_PASS_OPERATION
        || query.inputs.len() != 4
        || query.outputs.len() != 2
    {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "iir.prepared_pass.invalid_query",
        ));
    }
    let expected_class = ProviderElementType::from(provider.precision());
    for representation in query.inputs.iter().chain(&query.outputs) {
        if representation.element_type != expected_class
            || representation.storage != ProviderStorage::DenseReal
        {
            return Some(ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedElementType,
                "iir.prepared_pass.class",
            ));
        }
        if representation.layout != ProviderLayout::ColumnMajorContiguous {
            return Some(ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedLayout,
                "iir.prepared_pass.layout",
            ));
        }
        if representation.residency != ProviderResidency::Device {
            return Some(ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedOperation,
                "iir.prepared_pass.residency",
            ));
        }
        if representation.checked_byte_len().is_none() {
            return Some(ProviderFeasibility::rejected(
                ProviderRejectionCode::InvalidShape,
                "iir.prepared_pass.geometry_overflow",
            ));
        }
    }
    if provider
        .capability_snapshot()
        .max_allocation_bytes
        .is_some_and(|limit| {
            query
                .inputs
                .iter()
                .chain(&query.outputs)
                .any(|representation| representation.checked_byte_len().is_some_and(|n| n > limit))
        })
    {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::ResourceLimit,
            "iir.prepared_pass.allocation_limit",
        ));
    }
    let [numerator, denominator, signal, state] = query.inputs.as_slice() else {
        unreachable!("input arity checked above")
    };
    let Some(coefficient_len) = numerator.checked_element_count() else {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::InvalidShape,
            "iir.prepared_pass.coefficient_overflow",
        ));
    };
    if coefficient_len == 0
        || denominator.checked_element_count() != Some(coefficient_len)
        || numerator.shape.len() != 2
        || numerator.shape[1] != 1
        || denominator.shape != numerator.shape
        || signal.shape.len() < 2
        || signal.shape != query.outputs[0].shape
        || state.shape != query.outputs[1].shape
        || workload.state_len != coefficient_len.saturating_sub(1)
        || workload.reflection_len != 0
        || u64::from(workload.dim) >= signal.shape.len() as u64
        || state.checked_element_count() != Some(workload.state_len)
        || signal.shape[workload.dim as usize] == 0
        || signal
            .shape
            .iter()
            .enumerate()
            .any(|(axis, &extent)| axis != workload.dim as usize && extent != 1)
    {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::InvalidShape,
            "iir.prepared_pass.geometry",
        ));
    }
    #[cfg(feature = "wgpu")]
    if wgpu
        && (coefficient_len > u64::from(u32::MAX)
            || signal
                .checked_element_count()
                .is_none_or(|count| count > u64::from(u32::MAX))
            || signal
                .shape
                .iter()
                .any(|&extent| extent > u64::from(u32::MAX)))
    {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::ResourceLimit,
            "iir.prepared_pass.addressable_range",
        ));
    }
    #[cfg(feature = "wgpu")]
    let unsupported_rank = wgpu
        && (signal.shape.len() > crate::backend::wgpu::params::FILTER_MAX_RANK
            || numerator.shape.len() > crate::backend::wgpu::params::FILTER_MAX_RANK);
    #[cfg(not(feature = "wgpu"))]
    let unsupported_rank = wgpu;
    if unsupported_rank {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::UnsupportedRank,
            "iir.prepared_pass.rank",
        ));
    }
    Some(ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: None,
        output_bytes: None,
        dispatches: None,
    }))
}

fn validate_causal(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> ProviderFeasibility {
    let reject = |code, detail| ProviderFeasibility::rejected(code, detail);
    let Some(workload) = query.workload.iir else {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "iir.causal.missing_typed_route",
        );
    };
    if workload.route != ProviderIirRoute::CausalPass
        || query.operation.0 != PROVIDER_IIR_CAUSAL_PASS_OPERATION
        || query.family != ProviderOperationFamily::Library
        || query.inputs.len() != 4
        || query.outputs.len() != 2
        || workload.reflection_len != 0
    {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "iir.causal.invalid_query",
        );
    }
    let [numerator, denominator, signal, state] = query.inputs.as_slice() else {
        unreachable!("input arity checked above")
    };
    let [output, final_state] = query.outputs.as_slice() else {
        unreachable!("output arity checked above")
    };
    let expected_class = ProviderElementType::from(provider.precision());
    let max_allocation = provider.capability_snapshot().max_allocation_bytes;
    for representation in query.inputs.iter().chain(query.outputs.iter()) {
        if !matches!(
            representation.element_type,
            ProviderElementType::F32 | ProviderElementType::F64
        ) || (wgpu && representation.element_type != expected_class)
            || representation.element_type != signal.element_type
            || representation.storage != ProviderStorage::DenseReal
        {
            return reject(
                ProviderRejectionCode::UnsupportedElementType,
                "iir.causal.class",
            );
        }
        if representation.layout != ProviderLayout::ColumnMajorContiguous {
            return reject(
                ProviderRejectionCode::UnsupportedLayout,
                "iir.causal.layout",
            );
        }
        if representation.residency != ProviderResidency::Device {
            return reject(
                ProviderRejectionCode::UnsupportedOperation,
                "iir.causal.residency",
            );
        }
        let Some(bytes) = representation.checked_byte_len() else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "iir.causal.byte_overflow",
            );
        };
        if max_allocation.is_some_and(|limit| bytes > limit) {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "iir.causal.allocation_limit",
            );
        }
    }
    let Some(coefficient_len) = numerator.checked_element_count() else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "iir.causal.coefficient_overflow",
        );
    };
    let Ok(axis) = usize::try_from(workload.dim) else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "iir.causal.axis_overflow",
        );
    };
    let Ok(state_len) = usize::try_from(workload.state_len) else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "iir.causal.state_overflow",
        );
    };
    let Some(state_shape) = signal
        .shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()
        .and_then(|shape| {
            runmat_accelerate_api::provider_iir_state_shape(&shape, axis, state_len).ok()
        })
    else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "iir.causal.axis_geometry",
        );
    };
    let state_shape: Vec<u64> = state_shape
        .into_iter()
        .map(|extent| extent as u64)
        .collect();
    if coefficient_len == 0
        || numerator.shape != [coefficient_len, 1]
        || denominator.shape != numerator.shape
        || workload.state_len != coefficient_len.saturating_sub(1)
        || signal.shape.len() < 2
        || output.shape != signal.shape
        || state.shape != state_shape
        || final_state.shape != state_shape
    {
        return reject(ProviderRejectionCode::InvalidShape, "iir.causal.geometry");
    }
    #[cfg(feature = "wgpu")]
    if wgpu
        && (signal.shape.len() > crate::backend::wgpu::params::FILTER_MAX_RANK
            || coefficient_len > u64::from(u32::MAX)
            || query
                .inputs
                .iter()
                .chain(query.outputs.iter())
                .any(|representation| {
                    representation
                        .checked_element_count()
                        .is_none_or(|count| count > u64::from(u32::MAX))
                        || representation
                            .shape
                            .iter()
                            .any(|&extent| extent > u64::from(u32::MAX))
                }))
    {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "iir.causal.addressable_range",
        );
    }
    #[cfg(not(feature = "wgpu"))]
    if wgpu {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "iir.causal.no_wgpu",
        );
    }
    let output_bytes = output.checked_byte_len().and_then(|signal_bytes| {
        final_state
            .checked_byte_len()
            .and_then(|state_bytes| signal_bytes.checked_add(state_bytes))
    });
    ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: None,
        output_bytes,
        dispatches: None,
    })
}

fn validate_zero_phase(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> ProviderFeasibility {
    let reject = |code, detail| ProviderFeasibility::rejected(code, detail);
    let Some(workload) = query.workload.iir else {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "iir.zero_phase.missing_typed_route",
        );
    };
    if workload.route != ProviderIirRoute::ZeroPhase
        || query.operation.0 != PROVIDER_IIR_ZERO_PHASE_OPERATION
        || query.family != ProviderOperationFamily::Library
        || query.inputs.len() != 4
        || query.outputs.len() != 1
    {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "iir.zero_phase.invalid_query",
        );
    }
    let [numerator, denominator, signal, state] = query.inputs.as_slice() else {
        unreachable!("input arity checked above")
    };
    let output = &query.outputs[0];
    let expected_class = ProviderElementType::from(provider.precision());
    let max_allocation_bytes = provider.capability_snapshot().max_allocation_bytes;
    for (representation, residency) in [
        (numerator, ProviderResidency::Host),
        (denominator, ProviderResidency::Host),
        (signal, ProviderResidency::Device),
        (state, ProviderResidency::Host),
        (output, ProviderResidency::Device),
    ] {
        if representation.element_type != expected_class
            || representation.storage != ProviderStorage::DenseReal
        {
            return reject(
                ProviderRejectionCode::UnsupportedElementType,
                "iir.zero_phase.class",
            );
        }
        if representation.layout != ProviderLayout::ColumnMajorContiguous {
            return reject(
                ProviderRejectionCode::UnsupportedLayout,
                "iir.zero_phase.layout",
            );
        }
        if representation.residency != residency {
            return reject(
                ProviderRejectionCode::UnsupportedOperation,
                "iir.zero_phase.residency",
            );
        }
        let Some(bytes) = representation.checked_byte_len() else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "iir.zero_phase.byte_overflow",
            );
        };
        if max_allocation_bytes.is_some_and(|limit| bytes > limit) {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "iir.zero_phase.allocation_limit",
            );
        }
    }
    let Some(coefficient_len) = numerator.checked_element_count() else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "iir.zero_phase.coefficient_overflow",
        );
    };
    let Some(signal_len) = signal.checked_element_count() else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "iir.zero_phase.signal_overflow",
        );
    };
    let Some(extended_len) = workload
        .reflection_len
        .checked_mul(2)
        .and_then(|padding| signal_len.checked_add(padding))
    else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "iir.zero_phase.reflection_overflow",
        );
    };
    let Some(extended_bytes) = extended_len.checked_mul(expected_class.byte_width()) else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "iir.zero_phase.reflection_byte_overflow",
        );
    };
    if max_allocation_bytes.is_some_and(|limit| extended_bytes > limit) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "iir.zero_phase.reflection_allocation_limit",
        );
    }
    if numerator.shape != denominator.shape
        || numerator.shape != [coefficient_len, 1]
        || coefficient_len == 0
        || workload.state_len != coefficient_len - 1
        || state.shape != [workload.state_len, 1]
        || workload.reflection_len != workload.state_len.saturating_mul(3)
        || signal_len <= workload.reflection_len
        || signal.shape.len() != 2
        || signal.shape != output.shape
        || workload.dim > 1
        || signal.shape[workload.dim as usize] != signal_len
        || signal.shape[1 - workload.dim as usize] != 1
    {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "iir.zero_phase.geometry",
        );
    }
    if wgpu
        && (extended_len > u64::from(u32::MAX)
            || coefficient_len > u64::from(u32::MAX)
            || workload.state_len > u64::from(u32::MAX))
    {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "iir.zero_phase.addressable_range",
        );
    }
    ProviderFeasibility::supported(ProviderResourceEstimate {
        // The complete staged peak and dispatch inventory must be measured by
        // the backend before automatic placement may price this route.
        transient_bytes: None,
        output_bytes: output.checked_byte_len(),
        dispatches: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        ProviderIirWorkload, ProviderOperationIdentity, ProviderRepresentation, ProviderWorkload,
    };

    fn representation(shape: &[u64]) -> ProviderRepresentation {
        ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape.to_vec(),
            residency: ProviderResidency::Device,
        }
    }

    fn prepared_pass() -> ProviderFeasibilityQuery {
        ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(PROVIDER_IIR_PREPARED_PASS_OPERATION),
            family: ProviderOperationFamily::Library,
            inputs: vec![
                representation(&[2, 1]),
                representation(&[2, 1]),
                representation(&[1, 14]),
                representation(&[1, 1]),
            ],
            outputs: vec![representation(&[1, 14]), representation(&[1, 1])],
            workload: ProviderWorkload {
                iir: Some(ProviderIirWorkload {
                    route: ProviderIirRoute::PreparedPass,
                    dim: 1,
                    state_len: 1,
                    reflection_len: 0,
                }),
                ..Default::default()
            },
        }
    }

    fn causal_pass() -> ProviderFeasibilityQuery {
        ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(PROVIDER_IIR_CAUSAL_PASS_OPERATION),
            family: ProviderOperationFamily::Library,
            inputs: vec![
                representation(&[2, 1]),
                representation(&[2, 1]),
                representation(&[3, 4]),
                representation(&[1, 3]),
            ],
            outputs: vec![representation(&[3, 4]), representation(&[1, 3])],
            workload: ProviderWorkload {
                iir: Some(ProviderIirWorkload {
                    route: ProviderIirRoute::CausalPass,
                    dim: 1,
                    state_len: 1,
                    reflection_len: 0,
                }),
                ..Default::default()
            },
        }
    }

    #[test]
    fn causal_pass_requires_order_first_state_and_exact_output_geometry() {
        let provider = crate::simple_provider::InProcessProvider::new();
        let mut query = causal_pass();
        assert!(validate(&provider, false, &query).unwrap().is_supported());
        query.inputs[3].shape = vec![3, 1];
        assert!(!validate(&provider, false, &query).unwrap().is_supported());
        query = causal_pass();
        query.outputs[0].shape = vec![4, 3];
        assert!(!validate(&provider, false, &query).unwrap().is_supported());
        query = causal_pass();
        query.inputs[2].element_type = ProviderElementType::I64;
        assert!(!validate(&provider, false, &query).unwrap().is_supported());
        query = causal_pass();
        query.workload.iir = None;
        assert!(!validate(&provider, false, &query).unwrap().is_supported());
    }

    #[test]
    fn prepared_pass_requires_typed_route_and_exact_physical_geometry() {
        let provider = crate::simple_provider::InProcessProvider::new();
        let mut query = prepared_pass();
        assert!(validate(&provider, false, &query).unwrap().is_supported());
        query.workload.iir = None;
        assert!(!validate(&provider, false, &query).unwrap().is_supported());
        query = prepared_pass();
        query.outputs[0].shape = vec![14, 1];
        assert!(!validate(&provider, false, &query).unwrap().is_supported());
        query = prepared_pass();
        query.inputs[0].element_type = ProviderElementType::F32;
        assert!(!validate(&provider, false, &query).unwrap().is_supported());
    }
}
