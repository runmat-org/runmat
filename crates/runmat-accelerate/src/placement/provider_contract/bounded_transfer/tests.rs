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

#[test]
fn exact_wgpu_upload_uses_physical_width_for_every_numeric_class() {
    let cases = [
        (ProviderElementType::Logical, 8),
        (ProviderElementType::I8, 4),
        (ProviderElementType::I16, 4),
        (ProviderElementType::I32, 4),
        (ProviderElementType::I64, 8),
        (ProviderElementType::U8, 4),
        (ProviderElementType::U16, 4),
        (ProviderElementType::U32, 4),
        (ProviderElementType::U64, 8),
        (ProviderElementType::F32, 4),
        (ProviderElementType::F64, 8),
        (ProviderElementType::ComplexF32, 8),
        (ProviderElementType::ComplexF64, 16),
    ];
    for (class, physical_width) in cases {
        for (elements, bytes_per_output) in [(0, 8), (2, 2 * physical_width)] {
            let storage = match class {
                ProviderElementType::ComplexF32 | ProviderElementType::ComplexF64 => {
                    ProviderStorage::DenseComplexInterleaved
                }
                _ => ProviderStorage::DenseReal,
            };
            let output = ProviderRepresentation {
                element_type: class,
                storage,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: vec![1, elements],
                residency: ProviderResidency::Device,
            };
            let query = ProviderFeasibilityQuery {
                operation: ProviderOperationIdentity::new("transfer.upload"),
                family: ProviderOperationFamily::Upload,
                inputs: vec![ProviderRepresentation {
                    residency: ProviderResidency::Host,
                    ..output.clone()
                }],
                outputs: vec![output.clone(), output],
                workload: ProviderWorkload::default(),
            };
            let snapshot = |limit| ProviderCapabilitySnapshot {
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
                element_types: vec![class],
                max_rank: None,
                max_allocation_bytes: Some(limit),
                concurrency: ProviderConcurrencyCapabilities {
                    spawn_handles: runmat_accelerate_api::SpawnHandleConcurrency::Reject,
                    concurrent_dispatch: false,
                    cancellation: false,
                    transactional_results: false,
                },
            };
            let ProviderFeasibility::Supported { estimate } =
                wgpu_dense_feasibility(&snapshot(bytes_per_output), &query, ProviderPrecision::F64)
            else {
                panic!("{class:?} with {elements} elements fits each buffer");
            };
            assert_eq!(estimate.output_bytes, Some(2 * bytes_per_output));
            assert_eq!(estimate.transient_bytes, Some(3 * bytes_per_output));
            assert!(matches!(
                wgpu_dense_feasibility(
                    &snapshot(bytes_per_output - 1),
                    &query,
                    ProviderPrecision::F64
                ),
                ProviderFeasibility::Rejected { .. }
            ));
            let mut mismatched = query.clone();
            mismatched.outputs[0].storage = if storage == ProviderStorage::DenseReal {
                ProviderStorage::DenseComplexInterleaved
            } else {
                ProviderStorage::DenseReal
            };
            assert!(matches!(
                wgpu_dense_feasibility(
                    &snapshot(bytes_per_output),
                    &mismatched,
                    ProviderPrecision::F64
                ),
                ProviderFeasibility::Rejected { .. }
            ));
            let mut strided = query.clone();
            strided.outputs[0].layout = ProviderLayout::Strided;
            assert!(matches!(
                wgpu_dense_feasibility(
                    &snapshot(bytes_per_output),
                    &strided,
                    ProviderPrecision::F64
                ),
                ProviderFeasibility::Rejected { .. }
            ));
        }
    }
}
