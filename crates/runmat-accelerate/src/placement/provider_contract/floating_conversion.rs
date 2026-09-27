//! Exact floating-conversion hooks; optional output templates remain runtime-owned.
use super::*;
use runmat_builtins::{DOUBLE_CATALOG_ENTRY, SINGLE_CATALOG_ENTRY};

#[cfg(test)]
mod tests;

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
    let target = if query.operation.0 == DOUBLE_CATALOG_ENTRY.identity.name {
        ProviderElementType::F64
    } else if query.operation.0 == SINGLE_CATALOG_ENTRY.identity.name {
        ProviderElementType::F32
    } else {
        return None;
    };
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
    complete.outputs = vec![output];
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
        resources.dispatches = Some(if !wgpu || elements == 0 || input.element_type == target {
            0
        } else {
            #[cfg(feature = "wgpu")]
            {
                let capacity = u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS)
                    * u64::from(crate::backend::wgpu::config::effective_workgroup_size());
                elements.div_ceil(capacity) as u32
            }
            #[cfg(not(feature = "wgpu"))]
            {
                1
            }
        });
        if wgpu && input.element_type == target {
            resources.transient_bytes = resources.output_bytes;
        }
    }
    Some(feasibility)
}
