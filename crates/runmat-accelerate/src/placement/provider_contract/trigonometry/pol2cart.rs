//! Physical compound coordinate work; no builtin redispatch or placement policy.
use super::super::*;

mod work;
#[cfg(test)]
mod tests;
#[cfg(feature = "wgpu")]
pub(crate) use work::CoordinateWork;

pub(super) fn operation() -> ProviderCapabilityOperation {
    ProviderCapabilityOperation {
        identity: ProviderOperationIdentity::new(
            runmat_builtins::POL2CART_CATALOG_ENTRY.identity.name,
        ),
        family: ProviderOperationFamily::Elementwise,
    }
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.operation != operation().identity {
        return None;
    }
    let work = match work::derive(provider.precision(), wgpu, query) {
        Ok(work) => work,
        Err((code, detail)) => return Some(ProviderFeasibility::rejected(code, detail)),
    };
    let mut complete = query.clone();
    complete.outputs = work.outputs;
    let mut snapshot = dense_capability_snapshot(provider, 1, &[]);
    snapshot.operations.push(operation());
    let dense = dense_feasibility_with_bytes(&snapshot, &complete, |value| {
        let count = value.checked_element_count()?;
        (if wgpu { count.max(1) } else { count }).checked_mul(value.element_type.byte_width())
    });
    Some(match dense {
        ProviderFeasibility::Supported { mut estimate } => {
            let Some(bytes) = estimate
                .transient_bytes
                .and_then(|bytes| bytes.checked_add(work.scratch_bytes))
            else {
                return Some(ProviderFeasibility::rejected(
                    ProviderRejectionCode::ResourceLimit,
                    "pol2cart.bytes.overflow",
                ));
            };
            if snapshot
                .max_allocation_bytes
                .is_some_and(|limit| bytes > limit)
            {
                return Some(ProviderFeasibility::rejected(
                    ProviderRejectionCode::ResourceLimit,
                    "pol2cart.compound.memory_limit",
                ));
            }
            estimate.transient_bytes = Some(bytes);
            estimate.dispatches = Some(work.dispatches);
            ProviderFeasibility::Supported { estimate }
        }
        rejected => rejected,
    })
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_work(
    precision: ProviderPrecision,
    query: &ProviderFeasibilityQuery,
) -> Option<CoordinateWork> {
    if query.operation != operation().identity {
        return None;
    }
    work::derive(precision, true, query).ok()
}
