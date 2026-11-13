//! Shared resource contract for a real-domain probe followed by a real unary hook.
//! Exact operation identities and hook availability remain with family owners.
use super::*;
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_types::{RealUnaryClassification, RealUnaryPredicate};

#[derive(Clone, Copy)]
pub(crate) enum HookProfile {
    InProcess,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

pub(crate) struct UnaryWork {
    #[cfg(feature = "wgpu")]
    pub(crate) probe_elements: u64,
    /// Conservative sum of numeric intermediate, uniform and readback storage.
    /// This is an allocation bound, not a claim that every buffer overlaps.
    pub(crate) temporary_bytes: u64,
    #[cfg(feature = "wgpu")]
    pub(crate) readback_bytes: u64,
    #[cfg(feature = "wgpu")]
    pub(crate) readbacks: u32,
    pub(crate) dispatches: u32,
    #[cfg(feature = "wgpu")]
    pub(crate) queue_submissions: u32,
    pub(crate) output: ProviderRepresentation,
}

type Rejection = (ProviderRejectionCode, &'static str);

pub(crate) fn work(
    profile: HookProfile,
    precision: ProviderPrecision,
    query: &ProviderFeasibilityQuery,
    predicate: RealUnaryPredicate,
) -> Result<UnaryWork, Rejection> {
    use ProviderRejectionCode as Code;
    if query.workload.numerical_domain
        != (NumericalDomainEvidence::RealUnary {
            predicate,
            classification: RealUnaryClassification::Inside,
        })
    {
        return Err((
            Code::UnsupportedOperation,
            "real_unary.inside_evidence.required",
        ));
    }
    if query.family != ProviderOperationFamily::Elementwise
        || query.inputs.len() != 1
        || query.outputs.len() > 1
    {
        return Err((Code::InvalidShape, "real_unary.unary.required"));
    }
    let input = &query.inputs[0];
    if input.storage != ProviderStorage::DenseReal {
        return Err((Code::UnsupportedStorage, "real_unary.real_dense.required"));
    }
    if input.element_type != ProviderElementType::from(precision) {
        return Err((
            Code::UnsupportedElementType,
            "real_unary.exact_precision.required",
        ));
    }
    if input.layout != ProviderLayout::ColumnMajorContiguous {
        return Err((Code::UnsupportedLayout, "real_unary.contiguous.required"));
    }
    let elements = input
        .checked_element_count()
        .ok_or((Code::ResourceLimit, "real_unary.shape.overflow"))?;
    if elements == 0 {
        // The runtime skips the probe and computes/restores the empty host result.
        // It is not the nonempty compound hook route represented here.
        return Err((Code::UnsupportedOperation, "real_unary.empty.host_route"));
    }
    let mut output = input.clone();
    output.residency = ProviderResidency::Device;
    if query.outputs.iter().any(|declared| declared != &output) {
        return Err((Code::InvalidShape, "real_unary.output.mismatch"));
    }
    let reductions = if predicate == RealUnaryPredicate::ClosedUnitInterval {
        2
    } else {
        1
    };
    let scalar_bytes = match precision {
        ProviderPrecision::F32 => 4,
        ProviderPrecision::F64 => 8,
    };
    let work = UnaryWork {
        #[cfg(feature = "wgpu")]
        probe_elements: elements
            .checked_mul(u64::from(reductions))
            .ok_or((Code::ResourceLimit, "real_unary.work.overflow"))?,
        temporary_bytes: scalar_bytes * u64::from(reductions) * 2,
        #[cfg(feature = "wgpu")]
        readback_bytes: scalar_bytes * u64::from(reductions),
        #[cfg(feature = "wgpu")]
        readbacks: reductions,
        dispatches: reductions + 1,
        #[cfg(feature = "wgpu")]
        queue_submissions: reductions + 1,
        output,
    };
    match profile {
        HookProfile::InProcess => Ok(work),
        #[cfg(feature = "wgpu")]
        HookProfile::Wgpu => {
            let mut work = work;
            if elements > u64::from(u32::MAX) {
                return Err((Code::ResourceLimit, "real_unary.dispatch.length"));
            }
            let groups = u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS);
            let chunk = groups * u64::from(crate::backend::wgpu::config::WORKGROUP_SIZE);
            let effective = u64::from(crate::backend::wgpu::config::effective_workgroup_size());
            if effective == 0 || elements.min(chunk).div_ceil(effective) > groups {
                return Err((Code::ResourceLimit, "real_unary.unary.dispatch_groups"));
            }
            if [
                "RUNMAT_DISABLE_REDUCE_GLOBAL",
                "RUNMAT_DISABLE_UNARY",
                "RUNMAT_PROVIDER_REDUCTION_SNAPSHOT",
                "RUNMAT_FORCE_REDUCE_SNAPSHOT",
            ]
            .iter()
            .any(|name| std::env::var(name).is_ok())
            {
                return Err((
                    Code::UnsupportedOperation,
                    "real_unary.nondefault_execution_route",
                ));
            }
            let resources = wgpu_resources(elements, scalar_bytes, reductions);
            work.probe_elements = resources.probe_elements;
            work.temporary_bytes = resources.temporary_bytes;
            work.dispatches = resources.dispatches;
            work.queue_submissions = resources.queue_submissions;
            Ok(work)
        }
    }
}

pub(crate) fn feasibility(
    profile: HookProfile,
    precision: ProviderPrecision,
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
    predicate: RealUnaryPredicate,
) -> ProviderFeasibility {
    let work = match work(profile, precision, query, predicate) {
        Ok(work) => work,
        Err((code, detail)) => return ProviderFeasibility::rejected(code, detail),
    };
    let mut accounted = query.clone();
    accounted.outputs = vec![work.output];
    match dense_feasibility(snapshot, &accounted) {
        ProviderFeasibility::Supported { mut estimate } => {
            let Some(total) = estimate
                .transient_bytes
                .and_then(|bytes| bytes.checked_add(work.temporary_bytes))
            else {
                return ProviderFeasibility::rejected(
                    ProviderRejectionCode::ResourceLimit,
                    "real_unary.bytes.overflow",
                );
            };
            if snapshot
                .max_allocation_bytes
                .is_some_and(|limit| total > limit)
            {
                return ProviderFeasibility::rejected(
                    ProviderRejectionCode::ResourceLimit,
                    "real_unary.probe.memory_limit",
                );
            }
            estimate.transient_bytes = Some(total);
            estimate.dispatches = Some(work.dispatches);
            ProviderFeasibility::Supported { estimate }
        }
        rejected => rejected,
    }
}

#[cfg(feature = "wgpu")]
struct WgpuResources {
    probe_elements: u64,
    temporary_bytes: u64,
    dispatches: u32,
    queue_submissions: u32,
}

#[cfg(feature = "wgpu")]
fn wgpu_resources(elements: u64, scalar_bytes: u64, reductions: u32) -> WgpuResources {
    use crate::backend::wgpu::config::{
        MAX_DISPATCH_WORKGROUPS, REDUCE_WORKGROUP_SIZE, WORKGROUP_SIZE,
    };
    let per_group = 2 * u64::from(REDUCE_WORKGROUP_SIZE);
    let reduce_chunk = per_group * u64::from(MAX_DISPATCH_WORKGROUPS);
    let unary_chunk = u64::from(WORKGROUP_SIZE) * u64::from(MAX_DISPATCH_WORKGROUPS);
    let mut current = elements;
    let mut probe_elements = 0;
    let mut intermediate_elements = 0;
    let mut reduction_dispatches = 0;
    while current > 1 {
        probe_elements += current;
        reduction_dispatches += current.div_ceil(reduce_chunk);
        current = current.div_ceil(per_group);
        intermediate_elements += current;
    }
    let reductions = u64::from(reductions);
    let reduction_dispatches = reduction_dispatches * reductions;
    let unary_dispatches = elements.div_ceil(unary_chunk);
    let dispatches = (reduction_dispatches + unary_dispatches) as u32;
    WgpuResources {
        probe_elements: probe_elements * reductions,
        // Each scalar download has a staging allocation; length-one reductions
        // alias backing storage under distinct handles and allocate no numeric buffer.
        temporary_bytes: (intermediate_elements + 1) * reductions * scalar_bytes
            + reduction_dispatches * 16
            + 16,
        dispatches,
        // Unary execution also submits its no-op warmup; each download submits
        // a copy and waits. Neither is a kernel dispatch or completion sample.
        queue_submissions: dispatches + 1 + reductions as u32,
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;
    #[test]
    fn real_unary_resources_count_each_pass_chunk_and_readback() {
        let scalar = wgpu_resources(1, 8, 2);
        assert_eq!(scalar.probe_elements, 0);
        assert_eq!(scalar.dispatches, 1);
        assert_eq!(scalar.queue_submissions, 4);
        assert_eq!(scalar.temporary_bytes, 32);
        let two_pass = wgpu_resources(1025, 4, 2);
        assert_eq!(two_pass.probe_elements, (1025 + 2) * 2);
        assert_eq!(two_pass.dispatches, 5);
        assert_eq!(two_pass.temporary_bytes, (2 + 1 + 1) * 2 * 4 + 4 * 16 + 16);
        let chunked = wgpu_resources(65535 * 1024 + 1, 8, 1);
        assert_eq!(chunked.dispatches, 7); // 2+1+1 reductions; 3 unary chunks
        assert_eq!(chunked.queue_submissions, 9);
    }
}
