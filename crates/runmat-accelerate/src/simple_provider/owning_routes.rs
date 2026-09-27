use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::HostLogicalTensorView;

#[test]
fn structure_hooks_preserve_native_storage_or_request_typed_fallback() {
    let provider = InProcessProvider::new();
    for (data, storage) in [
        (
            HostNumericDataView::F32(&[1.0, 0.0, 0.0, 1.0]),
            GpuTensorStorage::Real,
        ),
        (
            HostNumericDataView::F64(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            GpuTensorStorage::ComplexInterleaved,
        ),
    ] {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data,
                shape: &[2, 2],
                storage,
            })
            .unwrap();
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &provider
                .issymmetric(&input, ProviderSymmetryKind::Symmetric, 0.0)
                .unwrap_err()
        ));
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &block_on(provider.ishermitian(&input, ProviderHermitianKind::Hermitian, 0.0))
                .unwrap_err()
        ));
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &provider.bandwidth(&input).unwrap_err()
        ));
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &block_on(provider.sym_rcm(&input)).unwrap_err()
        ));
        assert!(input.allocation_lease().validate_live().is_ok());
    }
}

#[test]
fn linear_indexing_distinguishes_unsupported_storage_from_invalid_owner() {
    let provider = InProcessProvider::new();
    let foreign = InProcessProvider::new();
    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.gather_linear(&single, &[0], &[1, 1]).unwrap_err()
    ));
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider
            .scatter_linear(&single, &[0, 1], &single)
            .unwrap_err()
    ));
    let other = foreign
        .upload(&HostTensorView {
            data: &[3.0],
            shape: &[1, 1],
        })
        .unwrap();
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.scatter_linear(&single, &[0], &other).unwrap_err()
    ));
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &foreign.gather_linear(&single, &[0], &[1, 1]).unwrap_err()
    ));
    assert_eq!(
        block_on(provider.download_numeric(&single)).unwrap().data,
        HostNumericDataOwned::F32(vec![1.0, 2.0])
    );
}

#[test]
fn fill_like_and_flip_keep_class_or_explicitly_request_typed_fallback() {
    let provider = InProcessProvider::new();
    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.fill_like(&single, 2.0).unwrap_err()
    ));
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.flip(&single, &[1]).unwrap_err()
    ));
    let real = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let filled = provider.fill_like(&real, 3.0).unwrap();
    let ones = provider.ones_like(&real).unwrap();
    assert_eq!(
        block_on(provider.download(&ones)).unwrap().data,
        vec![1.0; 2]
    );
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.ones_like(&single).unwrap_err()
    ));
    assert_eq!(
        block_on(provider.download(&filled)).unwrap().data,
        vec![3.0; 2]
    );
    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1],
            shape: &[1, 2],
        })
        .unwrap();
    let flipped = provider.flip(&logical, &[1]).unwrap();
    assert!(flipped.view().is_logical());
    assert_eq!(
        block_on(provider.download(&flipped)).unwrap().data,
        vec![1.0, 0.0]
    );
}

#[test]
fn ndgrid_unsupported_representation_does_not_hide_invalid_owner_or_request() {
    let provider = InProcessProvider::new();
    let foreign = InProcessProvider::new();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::U64(&[1, u64::MAX]),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let axes = [runmat_accelerate_api::ProviderNdgridAxis { handle: &input }];
    let request = ProviderNdgridRequest {
        axes: &axes,
        output_shape: &[2],
        output_count: 1,
    };
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.ndgrid(&request).unwrap_err()
    ));
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &foreign.ndgrid(&request).unwrap_err()
    ));
    let malformed = ProviderNdgridRequest {
        axes: &axes,
        output_shape: &[],
        output_count: 1,
    };
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.ndgrid(&malformed).unwrap_err()
    ));
}

#[test]
fn typed_zero_constructors_preserve_class_storage_shape_and_owner() {
    let provider = InProcessProvider::new();
    let foreign = InProcessProvider::new();
    let fixtures = [
        (
            HostNumericDataOwned::F32(vec![7.0, 8.0]),
            GpuTensorStorage::Real,
        ),
        (
            HostNumericDataOwned::F64(vec![7.0, 8.0]),
            GpuTensorStorage::Real,
        ),
        (
            HostNumericDataOwned::F32(vec![7.0, 8.0, 9.0, 10.0]),
            GpuTensorStorage::ComplexInterleaved,
        ),
        (HostNumericDataOwned::U8(vec![7, 8]), GpuTensorStorage::Real),
        (
            HostNumericDataOwned::U64(vec![7, u64::MAX]),
            GpuTensorStorage::Real,
        ),
    ];
    for (data, storage) in fixtures {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: data.as_view(),
                shape: &[1, 2],
                storage,
            })
            .unwrap();
        assert!(foreign.zeros_like(&input, &[2, 2]).is_err());
        for shape in [vec![2, 1, 3], vec![usize::MAX, usize::MAX, 0]] {
            let output = provider.zeros_like(&input, &shape).unwrap();
            assert_eq!(output.shape(), shape);
            assert_eq!(
                output.physical().element_type(),
                input.physical().element_type()
            );
            assert_eq!(output.physical().storage(), storage);
            assert!(!output.shares_allocation_with(&input));
            let actual = block_on(provider.download_numeric(&output)).unwrap();
            assert_eq!(actual.data.element_type(), data.element_type());
            match &actual.data {
                HostNumericDataOwned::F32(values) => {
                    assert!(values.iter().all(|value| value.to_bits() == 0))
                }
                HostNumericDataOwned::F64(values) => {
                    assert!(values.iter().all(|value| value.to_bits() == 0))
                }
                HostNumericDataOwned::U8(values) => assert!(values.iter().all(|value| *value == 0)),
                HostNumericDataOwned::U64(values) => {
                    assert!(values.iter().all(|value| *value == 0))
                }
                _ => panic!("unexpected fixture output type"),
            }
            assert_eq!(
                actual.data.len(),
                if shape.contains(&0) {
                    0
                } else if storage == GpuTensorStorage::Real {
                    6
                } else {
                    12
                }
            );
        }
    }
    for value in [false, true] {
        let logical = provider.logical_fill(&[1, 3], value).unwrap();
        assert!(logical.view().is_logical());
        assert_eq!(
            block_on(provider.download(&logical)).unwrap().data,
            vec![if value { 1.0 } else { 0.0 }; 3]
        );
        let zeros = provider.zeros_like(&logical, &[2, 2]).unwrap();
        assert!(zeros.view().is_logical());
        assert_eq!(
            block_on(provider.download(&zeros)).unwrap().data,
            vec![0.0; 4]
        );
        assert!(provider
            .logical_fill(&[usize::MAX, usize::MAX, 0], value)
            .unwrap()
            .view()
            .is_logical());
    }
}

#[test]
fn same_type_cast_shares_allocation_but_publishes_a_distinct_owned_view() {
    let provider = InProcessProvider::new();
    for shape in [vec![2, 1, 2], vec![usize::MAX, usize::MAX, 0]] {
        let data = if shape.contains(&0) {
            vec![]
        } else {
            vec![1.0_f32, -0.0, 3.0, 4.0]
        };
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&data),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let output = block_on(provider.unary_single(&input)).unwrap();
        assert_eq!(input.allocation_identity(), output.allocation_identity());
        assert_ne!(input.view_identity(), output.view_identity());
        assert_eq!(output.shape(), shape);
        let wire = output.wire_descriptor();
        drop(input);
        assert_eq!(
            block_on(provider.download_numeric(&output)).unwrap().data,
            HostNumericDataOwned::F32(data)
        );
        drop(output);
        assert!(provider.acquire_live(&wire).is_err());
    }
}

#[test]
fn logical_backing_rejects_numeric_mutation_and_numeric_cast_has_independent_storage() {
    let provider = InProcessProvider::new();
    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1],
            shape: &[1, 2],
        })
        .unwrap();
    let numeric_view = logical
        .with_view(logical.view().clone().with_logical(false))
        .unwrap();
    let values = provider
        .upload(&HostTensorView {
            data: &[7.0],
            shape: &[1, 1],
        })
        .unwrap();
    assert!(provider
        .scatter_linear(&numeric_view, &[0], &values)
        .is_err());
    let cast = block_on(provider.unary_double(&numeric_view)).unwrap();
    assert_ne!(cast.allocation_identity(), logical.allocation_identity());
    provider.scatter_linear(&cast, &[0], &values).unwrap();
    assert_eq!(
        block_on(provider.download(&cast)).unwrap().data,
        vec![7.0, 1.0]
    );
    assert_eq!(
        block_on(provider.download(&logical)).unwrap().data,
        vec![0.0, 1.0]
    );
}

#[test]
fn reshape_preserves_backing_and_prefix_or_lazy_wire_views_are_rejected() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[2, 2],
        })
        .unwrap();
    let reshaped = provider.reshape(&input, &[1, 4]).unwrap();
    assert_eq!(input.allocation_identity(), reshaped.allocation_identity());
    assert_eq!(
        block_on(provider.download(&reshaped)).unwrap().shape,
        vec![1, 4]
    );
    let mut wire = input.wire_descriptor();
    wire.view = wire.view.with_shape(vec![1, 2]);
    assert!(provider.acquire_live(&wire).is_err());
    let mut wire = input.wire_descriptor();
    wire.view = wire.view.with_layout(
        runmat_accelerate_api::allocation::GpuTensorLayout::Transpose(
            runmat_accelerate_api::TransposeInfo {
                base_rows: 2,
                base_cols: 2,
            },
        ),
    );
    assert!(provider.acquire_live(&wire).is_err());
    assert_eq!(
        block_on(provider.download(&input)).unwrap().data,
        vec![1.0, 2.0, 3.0, 4.0]
    );
}
