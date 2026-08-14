use runmat_accelerate_api::{
    AccelProvider, ProviderCapabilityOperation, ProviderCapabilitySnapshot,
    ProviderConcurrencyCapabilities, ProviderElementType, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderRejectionCode, ProviderResourceEstimate, ProviderStorage,
    PROVIDER_CAPABILITY_SCHEMA_VERSION,
};

const IN_PROCESS_OPERATIONS: &[(&str, ProviderOperationFamily)] = &[
    ("transfer.upload", ProviderOperationFamily::Upload),
    ("transfer.download", ProviderOperationFamily::Download),
    ("legacy.elementwise", ProviderOperationFamily::Elementwise),
    ("legacy.reduction", ProviderOperationFamily::Reduction),
    ("legacy.matmul", ProviderOperationFamily::MatrixMultiply),
    ("fusion.elementwise", ProviderOperationFamily::Fusion),
    ("fusion.matmul_epilogue", ProviderOperationFamily::Fusion),
    ("fusion.centered_gram", ProviderOperationFamily::Fusion),
    (
        "fusion.power_step_normalize",
        ProviderOperationFamily::Fusion,
    ),
    ("fusion.explained_variance", ProviderOperationFamily::Fusion),
    ("fusion.image_normalize", ProviderOperationFamily::Fusion),
];

#[cfg(feature = "wgpu")]
const WGPU_OPERATIONS: &[(&str, ProviderOperationFamily)] = &[
    ("transfer.upload", ProviderOperationFamily::Upload),
    ("transfer.download", ProviderOperationFamily::Download),
    ("legacy.elementwise", ProviderOperationFamily::Elementwise),
    ("legacy.reduction", ProviderOperationFamily::Reduction),
    ("legacy.matmul", ProviderOperationFamily::MatrixMultiply),
    ("fusion.elementwise", ProviderOperationFamily::Fusion),
    ("fusion.reduction", ProviderOperationFamily::Fusion),
    ("fusion.matmul_epilogue", ProviderOperationFamily::Fusion),
    ("fusion.centered_gram", ProviderOperationFamily::Fusion),
    (
        "fusion.power_step_normalize",
        ProviderOperationFamily::Fusion,
    ),
    ("fusion.explained_variance", ProviderOperationFamily::Fusion),
    ("fusion.image_normalize", ProviderOperationFamily::Fusion),
];

pub(crate) fn in_process_capability_snapshot(
    provider: &(impl AccelProvider + ?Sized),
) -> ProviderCapabilitySnapshot {
    dense_capability_snapshot(provider, 1, IN_PROCESS_OPERATIONS)
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_capability_snapshot(
    provider: &(impl AccelProvider + ?Sized),
) -> ProviderCapabilitySnapshot {
    dense_capability_snapshot(provider, 1, WGPU_OPERATIONS)
}

pub(crate) fn dense_capability_snapshot(
    provider: &(impl AccelProvider + ?Sized),
    revision: u64,
    operations: &[(&'static str, ProviderOperationFamily)],
) -> ProviderCapabilitySnapshot {
    let device = provider.device_info_struct();
    let float_type = ProviderElementType::from(provider.precision());
    ProviderCapabilitySnapshot {
        schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
        revision,
        device: device.clone(),
        operations: operations
            .iter()
            .map(|(identity, family)| ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(*identity),
                family: *family,
            })
            .collect(),
        element_types: vec![
            ProviderElementType::Logical,
            ProviderElementType::I8,
            ProviderElementType::I16,
            ProviderElementType::I32,
            ProviderElementType::I64,
            ProviderElementType::U8,
            ProviderElementType::U16,
            ProviderElementType::U32,
            ProviderElementType::U64,
            float_type,
            match float_type {
                ProviderElementType::F32 => ProviderElementType::ComplexF32,
                _ => ProviderElementType::ComplexF64,
            },
        ],
        max_rank: None,
        max_allocation_bytes: device.memory_bytes,
        concurrency: ProviderConcurrencyCapabilities {
            spawn_handles: provider.spawn_handle_concurrency(),
            concurrent_dispatch: true,
            cancellation: false,
            transactional_results: true,
        },
    }
}

pub(crate) fn dense_feasibility(
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> ProviderFeasibility {
    if !snapshot.supports_operation(&query.operation, query.family) {
        return ProviderFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "provider.operation.unsupported",
        );
    }

    let mut total_bytes = 0_u64;
    for representation in query.inputs.iter().chain(&query.outputs) {
        if !snapshot
            .element_types
            .contains(&representation.element_type)
        {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedElementType,
                "provider.element_type.unsupported",
            );
        }
        if matches!(representation.storage, ProviderStorage::Sparse) {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedStorage,
                "provider.storage.sparse_unsupported",
            );
        }
        if representation.layout != ProviderLayout::ColumnMajorContiguous {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedLayout,
                "provider.layout.unsupported",
            );
        }
        if snapshot
            .max_rank
            .is_some_and(|max_rank| representation.shape.len() > max_rank as usize)
        {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedRank,
                "provider.rank.unsupported",
            );
        }
        let Some(bytes) = representation.checked_byte_len() else {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::InvalidShape,
                "provider.shape.overflow",
            );
        };
        let Some(next_total) = total_bytes.checked_add(bytes) else {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::ResourceLimit,
                "provider.bytes.overflow",
            );
        };
        total_bytes = next_total;
    }
    if snapshot
        .max_allocation_bytes
        .is_some_and(|maximum| total_bytes > maximum)
    {
        return ProviderFeasibility::rejected(
            ProviderRejectionCode::ResourceLimit,
            "provider.memory.limit",
        );
    }
    let output_bytes = query.outputs.iter().try_fold(0_u64, |total, output| {
        total.checked_add(output.checked_byte_len()?)
    });
    ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: Some(total_bytes),
        output_bytes,
        dispatches: Some(
            if matches!(
                query.family,
                ProviderOperationFamily::Upload | ProviderOperationFamily::Download
            ) {
                0
            } else {
                1
            },
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderStorage,
        ProviderWorkload,
    };

    fn representation(element_type: ProviderElementType) -> ProviderRepresentation {
        ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![4, 4],
            residency: ProviderResidency::Host,
        }
    }

    #[test]
    fn unsupported_operation_is_rejected_before_resources_are_estimated() {
        let snapshot = ProviderCapabilitySnapshot {
            schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
            revision: 1,
            device: runmat_accelerate_api::ApiDeviceInfo {
                device_id: 1,
                name: "test".to_string(),
                vendor: "test".to_string(),
                memory_bytes: Some(1024),
                backend: None,
            },
            operations: Vec::new(),
            element_types: vec![ProviderElementType::F64],
            max_rank: None,
            max_allocation_bytes: Some(1024),
            concurrency: ProviderConcurrencyCapabilities {
                spawn_handles: runmat_accelerate_api::SpawnHandleConcurrency::Reject,
                concurrent_dispatch: false,
                cancellation: false,
                transactional_results: false,
            },
        };
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("legacy.elementwise"),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![representation(ProviderElementType::F64)],
            outputs: vec![representation(ProviderElementType::F64)],
            workload: ProviderWorkload::default(),
        };

        assert!(matches!(
            dense_feasibility(&snapshot, &query),
            ProviderFeasibility::Rejected { .. }
        ));
    }
}
