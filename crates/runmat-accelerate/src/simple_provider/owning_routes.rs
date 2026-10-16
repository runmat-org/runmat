use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::HostLogicalTensorView;

#[test]
fn real_floating_linalg_hooks_preserve_class_owner_and_reject_unimplemented_views() {
    use runmat_accelerate_api::{GpuTensorLayout, GpuTensorView, TransposeInfo};

    let provider = InProcessProvider::new();
    for data in [
        HostNumericDataOwned::F32(vec![4.0, 0.0, 0.0, 2.0]),
        HostNumericDataOwned::F64(vec![4.0, 0.0, 0.0, 2.0]),
    ] {
        let input_class = data.element_type();
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: data.as_view(),
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let cases = [
            (
                "inv",
                block_on(provider.inv(&input, ProviderInvOptions::default())).unwrap(),
                vec![0.25, 0.0, 0.0, 0.5],
                vec![2, 2],
                input_class,
            ),
            (
                "pinv",
                block_on(provider.pinv(&input, ProviderPinvOptions::default())).unwrap(),
                vec![0.25, 0.0, 0.0, 0.5],
                vec![2, 2],
                input_class,
            ),
            (
                "cond",
                block_on(provider.cond(&input, ProviderCondNorm::Two)).unwrap(),
                vec![2.0],
                vec![1, 1],
                input_class,
            ),
            (
                "norm",
                block_on(provider.norm(&input, ProviderNormOrder::Two)).unwrap(),
                vec![4.0],
                vec![1, 1],
                input_class,
            ),
            (
                "rank",
                block_on(provider.rank(&input, None)).unwrap(),
                vec![2.0],
                vec![1, 1],
                NumericElementType::F64,
            ),
            (
                "rcond",
                block_on(provider.rcond(&input)).unwrap(),
                vec![0.5],
                vec![1, 1],
                input_class,
            ),
        ];
        for (name, output, expected, shape, class) in cases {
            assert_eq!(output.shape(), shape, "{name}");
            assert_eq!(output.physical().element_type(), class, "{name}");
            assert!(!output.shares_allocation_with(&input), "{name}");
            assert_eq!(output.device_id(), input.device_id(), "{name}");
            let actual = block_on(provider.download_numeric(&output)).unwrap();
            let actual = match actual.data {
                HostNumericDataOwned::F32(values) => {
                    values.into_iter().map(f64::from).collect::<Vec<_>>()
                }
                HostNumericDataOwned::F64(values) => values,
                other => panic!("{name}: unexpected output storage {other:?}"),
            };
            assert_eq!(actual.len(), expected.len(), "{name}");
            let tolerance = if data.element_type() == NumericElementType::F32 {
                1e-5
            } else {
                1e-12
            };
            for (observed, target) in actual.iter().zip(&expected) {
                assert!(
                    (*observed - *target).abs() <= tolerance,
                    "{name}: {observed} vs {target}"
                );
            }
            assert!(output.allocation_lease().validate_live().is_ok(), "{name}");
            assert!(input.allocation_lease().validate_live().is_ok(), "{name}");
        }
        // The owner rejects lazy layouts for every current in-process kernel.
        // A readback implementation must not widen that allocation authority.
        let proposed =
            input
                .view()
                .clone()
                .with_layout(GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 2,
                }));
        assert!(matches!(
            input.with_view(proposed),
            Err(runmat_accelerate_api::allocation::AllocationAcquisitionError::Unsupported)
        ));
    }

    let integer = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::U64(&[1, 3, 2, 4]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let proposed = GpuTensorView::contiguous(vec![2, 2]).with_layout(GpuTensorLayout::Transpose(
        TransposeInfo {
            base_rows: 2,
            base_cols: 2,
        },
    ));
    assert!(matches!(
        integer.with_view(proposed),
        Err(runmat_accelerate_api::allocation::AllocationAcquisitionError::Unsupported)
    ));
    assert_eq!(
        block_on(provider.download_numeric(&integer)).unwrap().data,
        HostNumericDataOwned::U64(vec![1, 3, 2, 4])
    );
}

#[test]
fn polynomial_eligibility_keeps_complex_integration_and_authenticates_both_inputs() {
    let provider = InProcessProvider::new();
    let complex = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[2., 4., 6., 8.]),
            shape: &[1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let integrated = provider.polyint(&complex, 3.).unwrap();
    assert_eq!(integrated.shape(), &[1, 3]);
    assert_eq!(
        integrated.physical().storage(),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(
        block_on(provider.download_numeric(&integrated))
            .unwrap()
            .data,
        HostNumericDataOwned::F64(vec![1., 2., 6., 8., 3., 0.])
    );
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.polyder_single(&complex)).unwrap_err()
    ));
    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1., 2.]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.polyint(&single, 0.).unwrap_err()
    ));
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.polyder_product(&single, &complex)).unwrap_err()
    ));
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.polyder_quotient(&single, &complex)).unwrap_err()
    ));
    let foreign = InProcessProvider::new()
        .upload(&HostTensorView {
            data: &[1., 2.],
            shape: &[1, 2],
        })
        .unwrap();
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.polyder_product(&single, &foreign)).unwrap_err()
    ));
    assert!(complex.allocation_lease().validate_live().is_ok());
}

#[test]
fn structure_hooks_preserve_native_storage_or_request_typed_fallback() {
    let provider = InProcessProvider::new();
    let shape = [u32::MAX as usize, u32::MAX as usize, 2, 0];
    let empty = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[]),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let permuted = provider.permute(&empty, &[3, 2, 1, 0]).unwrap().unwrap();
    assert_eq!(
        permuted.shape(),
        [0, 2, u32::MAX as usize, u32::MAX as usize]
    );
    let tiled = provider.repmat(&empty, &[1, 1, 1, 1]).unwrap().unwrap();
    assert_eq!(tiled.shape(), shape);
    assert_eq!(
        block_on(provider.download_numeric(&tiled)).unwrap().data,
        HostNumericDataOwned::F32(vec![])
    );
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
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &block_on(provider.lu(&input)).unwrap_err()
        ));
        assert!(input.allocation_lease().validate_live().is_ok());
    }
}

#[test]
fn linear_indexing_preserves_native_single_gather_and_rejects_invalid_owner() {
    let provider = InProcessProvider::new();
    let foreign = InProcessProvider::new();
    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let gathered = provider
        .gather_linear(&single, &[1, 0, 1], &[3, 1])
        .expect("native single gather");
    assert_eq!(gathered.shape(), &[3, 1]);
    assert_eq!(gathered.physical().element_type(), NumericElementType::F32);
    assert_eq!(gathered.physical().storage(), GpuTensorStorage::Real);
    assert_ne!(gathered.buffer_id(), single.buffer_id());
    assert_eq!(
        block_on(provider.download_numeric(&gathered)).unwrap().data,
        HostNumericDataOwned::F32(vec![2.0, 1.0, 2.0])
    );
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
    let scatter_error = provider.scatter_linear(&single, &[0], &other).unwrap_err();
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &scatter_error
    ));
    assert!(scatter_error.to_string().contains("unknown buffer"));
    let gather_error = foreign.gather_linear(&single, &[0], &[1, 1]).unwrap_err();
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &gather_error
    ));
    assert!(gather_error.to_string().contains("unknown buffer"));
    let download =
        runmat_accelerate_api::ProviderFeasibilityQuery::exact_numeric_download(&single).unwrap();
    let runmat_accelerate_api::ProviderFeasibility::Supported { estimate } =
        provider.query_numeric_download_feasibility(&single, &download)
    else {
        panic!("actual native F32 registry readback must be admitted on the F64 owner")
    };
    assert_eq!(estimate.output_bytes, Some(8));
    assert_eq!(estimate.transient_bytes, Some(0));
    assert_eq!(estimate.dispatches, Some(0));
    assert!(!foreign
        .query_numeric_download_feasibility(&single, &download)
        .is_supported());
    assert!(
        !provider.query_feasibility(&download).is_supported(),
        "exact native readback does not widen the general preferred-class snapshot"
    );
    assert_eq!(
        block_on(provider.download_numeric(&single)).unwrap().data,
        HostNumericDataOwned::F32(vec![1.0, 2.0])
    );
    assert!(single.allocation_lease().validate_live().is_ok());
    assert!(gathered.allocation_lease().validate_live().is_ok());
}

#[test]
fn fill_like_preserves_single_and_flip_requests_typed_fallback() {
    let provider = InProcessProvider::new();
    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let single_filled = provider
        .fill_like(&single, 2.0)
        .unwrap()
        .expect("admitted optional constructor");
    assert_eq!(
        block_on(provider.download_numeric(&single_filled))
            .unwrap()
            .data,
        HostNumericDataOwned::F32(vec![2.0, 2.0])
    );
    assert!(provider.flip(&single, &[1]).unwrap().is_none());
    let real = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let filled = provider
        .fill_like(&real, 3.0)
        .unwrap()
        .expect("admitted optional constructor");
    let ones = provider
        .ones_like(&real)
        .unwrap()
        .expect("admitted optional constructor");
    assert_eq!(
        block_on(provider.download(&ones)).unwrap().data,
        vec![1.0; 2]
    );
    let single_ones = provider
        .ones_like(&single)
        .unwrap()
        .expect("admitted optional constructor");
    assert_eq!(
        block_on(provider.download_numeric(&single_ones))
            .unwrap()
            .data,
        HostNumericDataOwned::F32(vec![1.0, 1.0])
    );
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
    let flipped = provider
        .flip(&logical, &[1])
        .unwrap()
        .expect("admitted logical flip");
    assert!(flipped.view().is_logical());
    assert_eq!(
        block_on(provider.download(&flipped)).unwrap().data,
        vec![1.0, 0.0]
    );
}

#[test]
fn floating_cast_preserves_interleaved_components_and_negative_zero_fill() {
    let provider = InProcessProvider::new();
    let source = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[f64::MAX, -0.0, 1.25, -2.5]),
            shape: &[2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let rounded = block_on(provider.unary_single(&source)).unwrap();
    assert_eq!(rounded.shape(), &[2]);
    assert_eq!(
        rounded.physical().storage(),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(rounded.physical().element_type(), NumericElementType::F32);
    assert!(!rounded.shares_allocation_with(&source));
    assert_eq!(
        block_on(provider.download_numeric(&rounded)).unwrap().data,
        HostNumericDataOwned::F32(vec![f32::INFINITY, -0.0, 1.25, -2.5])
    );
    let widened = block_on(provider.unary_double(&rounded)).unwrap();
    assert_eq!(widened.physical().element_type(), NumericElementType::F64);
    assert_eq!(
        widened.physical().storage(),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(widened.shape(), &[2]);
    assert_eq!(
        block_on(provider.download_numeric(&widened)).unwrap().data,
        HostNumericDataOwned::F64(vec![f64::INFINITY, -0.0, 1.25, -2.5])
    );
    let real = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, 2.0]),
            shape: &[2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let filled = provider
        .fill_like(&real, -0.0)
        .unwrap()
        .expect("admitted optional constructor");
    let HostNumericDataOwned::F64(values) =
        block_on(provider.download_numeric(&filled)).unwrap().data
    else {
        panic!("native Double fill")
    };
    assert!(values
        .iter()
        .all(|value| value.to_bits() == (-0.0f64).to_bits()));
    for handle in [source, rounded, widened, real, filled] {
        handle.release().unwrap();
    }
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
            let output = provider
                .zeros_like(&input, &shape)
                .unwrap()
                .expect("admitted optional constructor");
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
        let logical = provider
            .logical_fill(&[1, 3], value)
            .unwrap()
            .expect("admitted optional constructor");
        assert!(logical.view().is_logical());
        assert_eq!(
            block_on(provider.download(&logical)).unwrap().data,
            vec![if value { 1.0 } else { 0.0 }; 3]
        );
        let zeros = provider
            .zeros_like(&logical, &[2, 2])
            .unwrap()
            .expect("admitted optional constructor");
        assert!(zeros.view().is_logical());
        assert_eq!(
            block_on(provider.download(&zeros)).unwrap().data,
            vec![0.0; 4]
        );
        assert!(provider
            .logical_fill(&[usize::MAX, usize::MAX, 0], value)
            .unwrap()
            .expect("admitted optional constructor")
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
