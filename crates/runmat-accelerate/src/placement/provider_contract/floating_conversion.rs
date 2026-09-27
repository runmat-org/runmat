//! Exact floating-conversion hooks; optional output templates remain runtime-owned.
use super::*;
use runmat_builtins::{DOUBLE_CATALOG_ENTRY, SINGLE_CATALOG_ENTRY};

#[cfg(test)]
mod tests;
mod work;
#[cfg(feature = "wgpu")]
pub(crate) use work::{observation_capability, physical_work};

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.extend(
        [DOUBLE_CATALOG_ENTRY.identity, SINGLE_CATALOG_ENTRY.identity]
            .into_iter()
            .map(|identity| ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(identity.name),
                family: ProviderOperationFamily::Elementwise,
            }),
    );
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let target = target(query)?;
    validate_target(provider, wgpu, query, target)
}

fn target(query: &ProviderFeasibilityQuery) -> Option<ProviderElementType> {
    if query.operation.0 == DOUBLE_CATALOG_ENTRY.identity.name {
        Some(ProviderElementType::F64)
    } else if query.operation.0 == SINGLE_CATALOG_ENTRY.identity.name {
        Some(ProviderElementType::F32)
    } else {
        None
    }
}

fn validate_target(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
    target: ProviderElementType,
) -> Option<ProviderFeasibility> {
    let reject = |code, reason| Some(ProviderFeasibility::rejected(code, reason));
    if query.family != ProviderOperationFamily::Elementwise {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "floating_cast.family",
        );
    }
    if query.inputs.len() != 1 || query.outputs.len() > 1 {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "floating_cast.operand_count",
        );
    }
    let input = &query.inputs[0];
    if input.storage != ProviderStorage::DenseReal {
        return reject(
            ProviderRejectionCode::UnsupportedStorage,
            "floating_cast.real.required",
        );
    }
    let input_supported = match (input.element_type, target) {
        (ProviderElementType::F32, ProviderElementType::F32) => true,
        (ProviderElementType::F64, _) => provider.precision() == ProviderPrecision::F64,
        _ => false,
    };
    if !input_supported {
        return reject(
            ProviderRejectionCode::UnsupportedElementType,
            "floating_cast.input_type",
        );
    }
    let output = ProviderRepresentation {
        element_type: target,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: input.shape.clone(),
        residency: ProviderResidency::Device,
    };
    if query.outputs.iter().any(|declared| declared != &output) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "floating_cast.output_representation",
        );
    }
    let Some(elements) = input
        .shape
        .iter()
        .try_fold(1_u64, |n, dim| n.checked_mul(*dim))
    else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "floating_cast.shape.overflow",
        );
    };
    if wgpu && elements > u64::from(u32::MAX) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "floating_cast.dispatch.length",
        );
    }
    let mut complete = query.clone();
    // Same-type WGPU casts publish an alias registration, not a second physical
    // payload. Output descriptor consistency was checked above.
    complete.outputs = if wgpu && input.element_type == target {
        vec![]
    } else {
        vec![output]
    };
    let mut snapshot = dense_capability_snapshot(provider, 1, &[]);
    append(&mut snapshot);
    // This operation has genuinely typed F32 output on an F64 owner. Do not
    // infer support for unrelated native-precision legacy operations.
    if !snapshot.element_types.contains(&ProviderElementType::F32) {
        snapshot.element_types.push(ProviderElementType::F32);
    }
    let allocation_bytes: fn(&ProviderRepresentation) -> Option<u64> = if wgpu {
        crate::backend::wgpu_storage::checked_allocation_bytes
    } else {
        ProviderRepresentation::checked_byte_len
    };
    let mut feasibility = dense_feasibility_with_bytes(&snapshot, &complete, allocation_bytes);
    if let ProviderFeasibility::Supported {
        estimate: resources,
    } = &mut feasibility
    {
        // InProcess executes synchronously on CPU; WGPU same-type casts only
        // register a fresh handle sharing the original allocation.
        if wgpu {
            let Some(work) = work::build(provider, query, target) else {
                return reject(
                    ProviderRejectionCode::UnknownResourceEstimate,
                    "floating_cast.physical_work.unknown",
                );
            };
            *resources = work.estimate;
            if snapshot
                .max_allocation_bytes
                .is_some_and(|limit| resources.transient_bytes.is_some_and(|bytes| bytes > limit))
            {
                return reject(
                    ProviderRejectionCode::ResourceLimit,
                    "floating_cast.physical_work.limit",
                );
            }
        } else {
            resources.dispatches = Some(0);
        }
    }
    Some(feasibility)
}
