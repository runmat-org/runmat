use super::*;
use runmat_accelerate_api::{
    ProviderCapabilityOperation, ProviderConcurrencyCapabilities, ProviderOperationIdentity,
    ProviderPrecision, ProviderRepresentation, ProviderTransferOutputBound, ProviderWorkload,
    PROVIDER_CAPABILITY_SCHEMA_VERSION,
};

#[test]
fn empty_wgpu_upload_admission_counts_physical_storage_without_copy_volume() {
    let snapshot = |element_type, limit| ProviderCapabilitySnapshot {
        schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
        revision: 1,
        device: runmat_accelerate_api::ApiDeviceInfo {
            device_id: 1,
            name: "test".into(),
            vendor: "test".into(),
            memory_bytes: None,
            backend: None,
        },
        operations: vec![ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new("transfer.upload"),
            family: ProviderOperationFamily::Upload,
        }],
        element_types: vec![element_type],
        max_rank: None,
        max_allocation_bytes: Some(limit),
        concurrency: ProviderConcurrencyCapabilities {
            spawn_handles: runmat_accelerate_api::SpawnHandleConcurrency::Reject,
            concurrent_dispatch: false,
            cancellation: false,
            transactional_results: false,
        },
    };
    for (precision, class, width) in [
        (ProviderPrecision::F32, ProviderElementType::F32, 4),
        (ProviderPrecision::F64, ProviderElementType::F64, 8),
    ] {
        let output = ProviderRepresentation {
            element_type: class,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 0],
            residency: ProviderResidency::Device,
        };
        let exact = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("transfer.upload"),
            family: ProviderOperationFamily::Upload,
            inputs: vec![ProviderRepresentation {
                residency: ProviderResidency::Host,
                ..output.clone()
            }],
            outputs: vec![output.clone(), output.clone()],
            workload: ProviderWorkload::default(),
        };
        let ProviderFeasibility::Supported { estimate } =
            wgpu_dense_feasibility(&snapshot(class, width), &exact, precision)
        else {
            panic!("two empty buffers fit the per-buffer limit")
        };
        assert_eq!(output.checked_byte_len(), Some(0));
        assert_eq!(estimate.output_bytes, Some(2 * width));
        assert_eq!(estimate.transient_bytes, Some(3 * width));
        assert!(matches!(
            wgpu_dense_feasibility(&snapshot(class, width - 1), &exact, precision),
            ProviderFeasibility::Rejected { .. }
        ));

        let bound = ProviderTransferOutputBound {
            element_type: class,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            residency: ProviderResidency::Device,
            max_elements: 0,
        };
        let bounded = ProviderFeasibilityQuery {
            inputs: vec![],
            outputs: vec![],
            workload: ProviderWorkload {
                requested_outputs: Some(2),
                transfer_output_bounds: Some(vec![bound.clone(), bound]),
                ..ProviderWorkload::default()
            },
            ..exact.clone()
        };
        let ProviderFeasibility::Supported { estimate } =
            wgpu_dense_feasibility(&snapshot(class, width), &bounded, precision)
        else {
            panic!("bounded empty buffers fit the same physical floor")
        };
        assert_eq!(estimate.output_bytes, Some(2 * width));
        assert!(matches!(
            wgpu_dense_feasibility(&snapshot(class, width - 1), &bounded, precision),
            ProviderFeasibility::Rejected { .. }
        ));
    }
}
