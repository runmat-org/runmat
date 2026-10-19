//! Exact floating remainder hook admission and native allocation geometry.
use super::*;
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_types::FloatingRemainderKind;

#[cfg(feature = "wgpu")]
mod work;
#[cfg(feature = "wgpu")]
pub(crate) use work::wgpu_work;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(super) enum HookProfile {
    InProcess,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

fn identity(kind: FloatingRemainderKind) -> &'static str {
    match kind {
        FloatingRemainderKind::Modulus => runmat_builtins::MOD_CATALOG_ENTRY.identity.name,
        FloatingRemainderKind::Remainder => runmat_builtins::REM_CATALOG_ENTRY.identity.name,
    }
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.extend(
        [
            FloatingRemainderKind::Modulus,
            FloatingRemainderKind::Remainder,
        ]
        .map(|kind| ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new(identity(kind)),
            family: ProviderOperationFamily::Elementwise,
        }),
    );
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    profile: HookProfile,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let kind = [
        FloatingRemainderKind::Modulus,
        FloatingRemainderKind::Remainder,
    ]
    .into_iter()
    .find(|kind| identity(*kind) == query.operation.0)?;
    Some(evaluate(provider, profile, query, kind))
}

fn evaluate(
    provider: &(impl AccelProvider + ?Sized),
    profile: HookProfile,
    query: &ProviderFeasibilityQuery,
    kind: FloatingRemainderKind,
) -> ProviderFeasibility {
    let reject = ProviderFeasibility::rejected;
    if query.family != ProviderOperationFamily::Elementwise
        || query.workload.numerical_domain != NumericalDomainEvidence::FloatingRemainder(kind)
    {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "remainder.strategy.required",
        );
    }
    if query.inputs.len() != 2 || query.outputs.len() > 1 {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "remainder.operand_count",
        );
    }
    let precision = match profile {
        HookProfile::InProcess => ProviderPrecision::F64,
        #[cfg(feature = "wgpu")]
        HookProfile::Wgpu => provider.precision(),
    };
    let mut physical = query.clone();
    for input in &mut physical.inputs {
        if input.element_type != ProviderElementType::from(precision) {
            return reject(
                ProviderRejectionCode::UnsupportedElementType,
                "remainder.floating_precision",
            );
        }
        if input.storage != ProviderStorage::DenseReal {
            return reject(
                ProviderRejectionCode::UnsupportedStorage,
                "remainder.real_storage",
            );
        }
        let native_stride = match profile {
            HookProfile::InProcess => false,
            #[cfg(feature = "wgpu")]
            HookProfile::Wgpu => {
                input.residency == ProviderResidency::Device
                    && input.layout == ProviderLayout::Strided
                    && input.shape.len() == 2
            }
        };
        if input.layout != ProviderLayout::ColumnMajorContiguous && !native_stride {
            return reject(
                ProviderRejectionCode::UnsupportedLayout,
                "remainder.native_layout",
            );
        }
        // Only allocation sizing uses contiguous layout here. The execution
        // receives the original handle and validates its physical strides.
        input.layout = ProviderLayout::ColumnMajorContiguous;
    }
    let shapes: Option<Vec<Vec<usize>>> = physical
        .inputs
        .iter()
        .map(|input| {
            input
                .shape
                .iter()
                .map(|extent| usize::try_from(*extent).ok())
                .collect()
        })
        .collect();
    let Some(shapes) = shapes else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "remainder.shape.overflow",
        );
    };
    let Ok(shape) = runmat_runtime::builtins::common::broadcast::broadcast_shapes(
        "remainder",
        &shapes[0],
        &shapes[1],
    ) else {
        return reject(ProviderRejectionCode::InvalidShape, "remainder.broadcast");
    };
    let mut output = physical.inputs[0].clone();
    output.shape = shape.iter().map(|extent| *extent as u64).collect();
    output.residency = ProviderResidency::Device;
    let Some(count) = output.checked_element_count() else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "remainder.count.overflow",
        );
    };
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu)
        && (shape.len() > crate::backend::wgpu::params::BCAST_MAX_RANK
            || count > u64::from(u32::MAX)
            || shape.iter().any(|extent| *extent > u32::MAX as usize))
    {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "remainder.index_limit",
        );
    }
    let _ = count;
    if query.outputs.iter().any(|declared| declared != &output) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "remainder.output_contract",
        );
    }
    physical.outputs = vec![output];
    let mut snapshot = dense_capability_snapshot(provider, 1, &[]);
    append(&mut snapshot);
    let mut feasibility = dense_feasibility(&snapshot, &physical);
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu) {
        let Some(work) = wgpu_work(precision, &physical) else {
            return reject(
                ProviderRejectionCode::ResourceLimit,
                "remainder.work.overflow",
            );
        };
        if let ProviderFeasibility::Supported { estimate } = &mut feasibility {
            estimate.dispatches = Some(work.dispatches);
            estimate.transient_bytes = estimate
                .transient_bytes
                .and_then(|bytes| bytes.checked_add(work.scratch_bytes));
            if estimate.transient_bytes.is_none_or(|bytes| {
                snapshot
                    .max_allocation_bytes
                    .is_some_and(|limit| bytes > limit)
            }) {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "remainder.memory.limit",
                );
            }
        }
    }
    #[cfg(not(feature = "wgpu"))]
    let _ = &mut feasibility;
    feasibility
}
