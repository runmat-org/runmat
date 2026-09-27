//! Power capability is conditional on request-owned numerical-domain evidence.
use super::*;
use runmat_accelerate_api::{NumericalDomainEvidence, PowerResultDomain};

#[derive(Clone, Copy)]
pub(crate) enum HookProfile {
    InProcess,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

pub(crate) struct PowerWork {
    #[cfg(feature = "wgpu")]
    pub(crate) elements: u64,
    pub(crate) temporary_bytes: u64,
    #[cfg(feature = "wgpu")]
    pub(crate) readback_bytes: u64,
    pub(crate) dispatches: u32,
    output: ProviderRepresentation,
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.push(ProviderCapabilityOperation {
        identity: ProviderOperationIdentity::new(
            runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
        ),
        family: ProviderOperationFamily::Elementwise,
    });
}

pub(super) fn feasibility(
    profile: HookProfile,
    precision: ProviderPrecision,
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.operation.0 != runmat_builtins::POWER_CATALOG_ENTRY.identity.name {
        return None;
    }
    let work = match work(profile, precision, query) {
        Ok(work) => work,
        Err((code, detail)) => return Some(ProviderFeasibility::rejected(code, detail)),
    };
    let mut accounted = query.clone();
    if accounted.outputs.is_empty() {
        accounted.outputs.push(work.output.clone());
    }
    let result = match dense_feasibility(snapshot, &accounted) {
        ProviderFeasibility::Supported { mut estimate } => {
            let Some(total) = estimate
                .transient_bytes
                .and_then(|bytes| bytes.checked_add(work.temporary_bytes))
            else {
                return Some(ProviderFeasibility::rejected(
                    ProviderRejectionCode::ResourceLimit,
                    "power.bytes.overflow",
                ));
            };
            if snapshot
                .max_allocation_bytes
                .is_some_and(|maximum| total > maximum)
            {
                return Some(ProviderFeasibility::rejected(
                    ProviderRejectionCode::ResourceLimit,
                    "power.probe.memory_limit",
                ));
            }
            estimate.transient_bytes = Some(total);
            estimate.dispatches = Some(work.dispatches);
            ProviderFeasibility::Supported { estimate }
        }
        rejected => rejected,
    };
    Some(result)
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_work(
    precision: ProviderPrecision,
    query: &ProviderFeasibilityQuery,
) -> Option<PowerWork> {
    if query.operation.0 != runmat_builtins::POWER_CATALOG_ENTRY.identity.name {
        return None;
    }
    work(HookProfile::Wgpu, precision, query).ok()
}

fn work(
    profile: HookProfile,
    precision: ProviderPrecision,
    query: &ProviderFeasibilityQuery,
) -> Result<PowerWork, (ProviderRejectionCode, &'static str)> {
    use ProviderRejectionCode as Rejection;
    let NumericalDomainEvidence::RealPower(domain) = query.workload.numerical_domain else {
        return Err((
            Rejection::UnsupportedOperation,
            "power.resolved_domain.required",
        ));
    };
    if query.family != ProviderOperationFamily::Elementwise
        || query.inputs.len() != 2
        || query.outputs.len() > 1
    {
        return Err((Rejection::InvalidShape, "power.operand_roles.invalid"));
    }
    let (element, complex, bytes) = match precision {
        ProviderPrecision::F32 => (
            ProviderElementType::F32,
            ProviderElementType::ComplexF32,
            4u64,
        ),
        ProviderPrecision::F64 => (
            ProviderElementType::F64,
            ProviderElementType::ComplexF64,
            8u64,
        ),
    };
    if matches!(profile, HookProfile::InProcess) && precision != ProviderPrecision::F64 {
        return Err((
            Rejection::UnsupportedElementType,
            "power.native_double.required",
        ));
    }
    for input in &query.inputs {
        if input.element_type != element {
            return Err((
                Rejection::UnsupportedElementType,
                "power.precision.mismatch",
            ));
        }
        if input.storage != ProviderStorage::DenseReal {
            return Err((Rejection::UnsupportedStorage, "power.real_storage.required"));
        }
    }
    let rank = query.inputs[0].shape.len().max(query.inputs[1].shape.len());
    let align = |shape: &[u64]| {
        let mut shape = if rank >= 2 && shape.len() == 1 {
            vec![1, shape[0]]
        } else {
            shape.to_vec()
        };
        shape.resize(rank, 1);
        shape
    };
    let left = align(&query.inputs[0].shape);
    let right = align(&query.inputs[1].shape);
    let mut shape = Vec::with_capacity(rank);
    for (&a, &b) in left.iter().zip(&right) {
        shape.push(if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return Err((Rejection::InvalidShape, "power.broadcast.invalid"));
        });
    }
    let elements = shape
        .iter()
        .try_fold(1u64, |n, d| n.checked_mul(*d))
        .ok_or((Rejection::InvalidShape, "power.shape.overflow"))?;
    let width = if domain == PowerResultDomain::Complex {
        2u64
    } else {
        1
    };
    let words = elements
        .checked_mul(width)
        .ok_or((Rejection::ResourceLimit, "power.output.overflow"))?;
    let output = ProviderRepresentation {
        element_type: if width == 2 { complex } else { element },
        storage: if width == 2 {
            ProviderStorage::DenseComplexInterleaved
        } else {
            ProviderStorage::DenseReal
        },
        shape,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Host,
    };
    if let Some(declared) = query.outputs.first() {
        if declared.shape != output.shape
            || declared.element_type != output.element_type
            || declared.storage != output.storage
        {
            return Err((Rejection::InvalidShape, "power.output_contract.mismatch"));
        }
    }
    let result = PowerWork {
        #[cfg(feature = "wgpu")]
        elements,
        temporary_bytes: 0,
        #[cfg(feature = "wgpu")]
        readback_bytes: 0,
        dispatches: 0,
        output,
    };
    let _ = (profile, words, bytes);
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu) {
        let mut result = result;
        if rank > crate::backend::wgpu::params::BCAST_MAX_RANK
            || words > u64::from(u32::MAX)
            || result.output.shape.iter().any(|d| *d > u64::from(u32::MAX))
        {
            return Err((Rejection::ResourceLimit, "power.gpu_index_limit"));
        }
        if elements != 0 {
            // Paired flags, two scalar full-reduction stages, then power.
            result.temporary_bytes = elements
                .checked_add(2)
                .and_then(|n| n.checked_mul(bytes))
                .ok_or((Rejection::ResourceLimit, "power.probe.overflow"))?;
            result.readback_bytes = bytes;
            result.dispatches = 4;
        }
        return Ok(result);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_domain_is_not_real_and_complex_output_bytes_are_explicit() {
        let provider = crate::simple_provider::InProcessProvider::new();
        let mut snapshot = provider.capability_snapshot();
        append(&mut snapshot);
        let input = ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 3],
            residency: ProviderResidency::Host,
        };
        let mut query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(
                runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
            ),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![input.clone(), input],
            outputs: vec![],
            workload: Default::default(),
        };
        assert!(!feasibility(
            HookProfile::InProcess,
            ProviderPrecision::F64,
            &snapshot,
            &query
        )
        .unwrap()
        .is_supported());
        query.workload.numerical_domain =
            NumericalDomainEvidence::RealPower(PowerResultDomain::Complex);
        let ProviderFeasibility::Supported { estimate } = feasibility(
            HookProfile::InProcess,
            ProviderPrecision::F64,
            &snapshot,
            &query,
        )
        .unwrap() else {
            panic!("resolved support")
        };
        assert_eq!(estimate.output_bytes, Some(48));
        assert_eq!(estimate.transient_bytes, Some(96));
        #[cfg(feature = "wgpu")]
        {
            let work = wgpu_work(ProviderPrecision::F64, &query).unwrap();
            assert_eq!(work.temporary_bytes, 40);
            assert_eq!(work.readback_bytes, 8);
            assert_eq!(work.dispatches, 4);
            for input in &mut query.inputs {
                input.shape = vec![0, 3];
            }
            let empty = wgpu_work(ProviderPrecision::F64, &query).unwrap();
            assert_eq!(empty.temporary_bytes, 0);
            assert_eq!(empty.readback_bytes, 0);
            assert_eq!(empty.dispatches, 0);
            query.inputs[0].shape = vec![u64::MAX, 2];
            query.inputs[1].shape = vec![1, 1];
            assert!(wgpu_work(ProviderPrecision::F64, &query).is_none());
        }
    }
}
