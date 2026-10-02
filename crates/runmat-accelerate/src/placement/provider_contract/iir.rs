//! Side-effect-free feasibility for one prepared resident IIR pass.
use runmat_accelerate_api::{
    AccelProvider, ProviderElementType, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderIirRoute, ProviderLayout, ProviderOperationFamily, ProviderRejectionCode,
    ProviderResidency, ProviderResourceEstimate, ProviderStorage,
    PROVIDER_IIR_PREPARED_PASS_OPERATION,
};

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
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
                }),
                ..Default::default()
            },
        }
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
