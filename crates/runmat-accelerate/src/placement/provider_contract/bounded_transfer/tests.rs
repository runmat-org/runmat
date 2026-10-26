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
        sparse: None,
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
        (ProviderPrecision::F64, ProviderElementType::F32, 8),
        (ProviderPrecision::F32, ProviderElementType::F64, 4),
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
            inputs: vec![
                ProviderRepresentation {
                    residency: ProviderResidency::Host,
                    ..output.clone()
                };
                2
            ],
            outputs: vec![output.clone(), output.clone()],
            workload: ProviderWorkload {
                requested_outputs: Some(2),
                ..ProviderWorkload::default()
            },
        };
        let ProviderFeasibility::Supported { estimate } =
            wgpu_dense_feasibility(&snapshot(class, width), &exact, precision)
        else {
            panic!("two empty buffers fit the per-buffer limit")
        };
        assert_eq!(output.checked_byte_len(), Some(0));
        assert_eq!(estimate.output_bytes, Some(2 * width));
        assert_eq!(estimate.transient_bytes, Some(4 * width));
        assert!(matches!(
            wgpu_dense_feasibility(&snapshot(class, width - 1), &exact, precision),
            ProviderFeasibility::Rejected { .. }
        ));

        for max_elements in [0, 1] {
            let bound = ProviderTransferOutputBound {
                element_type: class,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                residency: ProviderResidency::Device,
                max_elements,
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
            let maximum = (max_elements * class.byte_width()).max(width);
            let ProviderFeasibility::Supported { estimate } =
                wgpu_dense_feasibility(&snapshot(class, maximum), &bounded, precision)
            else {
                panic!("bounded empty buffers fit the same physical floor")
            };
            assert_eq!(estimate.output_bytes, Some(2 * maximum));
            assert!(matches!(
                wgpu_dense_feasibility(&snapshot(class, maximum - 1), &bounded, precision),
                ProviderFeasibility::Rejected { .. }
            ));
            let mut overflow = bounded.clone();
            overflow.workload.transfer_output_bounds.as_mut().unwrap()[0].max_elements = u64::MAX;
            assert!(
                !wgpu_dense_feasibility(&snapshot(class, u64::MAX), &overflow, precision)
                    .is_supported()
            );
            let mut malformed = bounded.clone();
            malformed.workload.transfer_output_bounds.as_mut().unwrap()[0].storage =
                ProviderStorage::DenseComplexInterleaved;
            assert!(
                !wgpu_dense_feasibility(&snapshot(class, u64::MAX), &malformed, precision)
                    .is_supported()
            );
        }
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
                inputs: vec![
                    ProviderRepresentation {
                        residency: ProviderResidency::Host,
                        ..output.clone()
                    };
                    2
                ],
                outputs: vec![output.clone(), output],
                workload: ProviderWorkload {
                    requested_outputs: Some(2),
                    ..ProviderWorkload::default()
                },
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
                sparse: None,
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
            assert_eq!(estimate.transient_bytes, Some(4 * bytes_per_output));
            if matches!(
                class,
                ProviderElementType::F32
                    | ProviderElementType::F64
                    | ProviderElementType::ComplexF32
                    | ProviderElementType::ComplexF64
            ) {
                let mut bound = ProviderTransferOutputBound {
                    element_type: class,
                    storage,
                    layout: ProviderLayout::ColumnMajorContiguous,
                    residency: ProviderResidency::Device,
                    max_elements: elements,
                };
                assert_eq!(
                    crate::backend::wgpu_storage::checked_upload_bound_allocation_bytes(
                        &bound,
                        ProviderPrecision::F64
                    ),
                    Some(bytes_per_output)
                );
                bound.max_elements = u64::MAX;
                assert_eq!(
                    crate::backend::wgpu_storage::checked_upload_bound_allocation_bytes(
                        &bound,
                        ProviderPrecision::F64
                    ),
                    None
                );
                if storage == ProviderStorage::DenseComplexInterleaved {
                    // Exact native complex uploads are supported, but the
                    // existing public late-shaped bound contract stays real.
                    bound.max_elements = elements;
                    let bounded = ProviderFeasibilityQuery {
                        inputs: vec![],
                        outputs: vec![],
                        workload: ProviderWorkload {
                            requested_outputs: Some(1),
                            transfer_output_bounds: Some(vec![bound]),
                            ..ProviderWorkload::default()
                        },
                        ..query.clone()
                    };
                    assert!(!wgpu_dense_feasibility(
                        &snapshot(u64::MAX),
                        &bounded,
                        ProviderPrecision::F64
                    )
                    .is_supported());
                }
            }
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
            let mut wrong_shape = query.clone();
            wrong_shape.outputs[0].shape = vec![elements, 1];
            assert!(!wgpu_dense_feasibility(
                &snapshot(bytes_per_output),
                &wrong_shape,
                ProviderPrecision::F64
            )
            .is_supported());
            let mut missing_input = query.clone();
            missing_input.inputs.pop();
            assert!(!wgpu_dense_feasibility(
                &snapshot(bytes_per_output),
                &missing_input,
                ProviderPrecision::F64
            )
            .is_supported());
            let mut wrong_residency = query.clone();
            wrong_residency.inputs[0].residency = ProviderResidency::Device;
            assert!(!wgpu_dense_feasibility(
                &snapshot(bytes_per_output),
                &wrong_residency,
                ProviderPrecision::F64
            )
            .is_supported());
            let mut wrong_output_count = query.clone();
            wrong_output_count.workload.requested_outputs = Some(1);
            assert!(!wgpu_dense_feasibility(
                &snapshot(bytes_per_output),
                &wrong_output_count,
                ProviderPrecision::F64
            )
            .is_supported());
        }
    }
}
