use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorView, HostLogicalTensorView, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, TransposeInfo,
};

#[test]
fn complex_mean_dim_reports_capability_only_after_owner_and_dimension_admission() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0, 8.0,
                80.0,
            ]),
            shape: &[2, 2, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    for dim in 0..3 {
        assert!(block_on(provider.reduce_mean_dim(&input, dim))
            .unwrap()
            .is_none());
    }
    let error = block_on(provider.reduce_mean_dim(&input, 3)).unwrap_err();
    assert!(!error.is_unsupported());
    input.allocation_lease().validate_live().unwrap();
}

#[test]
fn ordinary_single_global_uses_double_accumulator_before_output_class_selection() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[16_777_216.0, 1.0, -16_777_216.0]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    provider.reset_telemetry();
    let global = block_on(provider.reduce_sum(&input)).unwrap();
    let dimensional = block_on(provider.reduce_sum_dim(&input, 0)).unwrap();
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    for output in [&global, &dimensional] {
        assert_eq!(output.physical().element_type(), NumericElementType::F64);
        assert_eq!(
            block_on(provider.download_numeric(output)).unwrap().data,
            HostNumericDataOwned::F64(vec![1.0])
        );
    }
}

#[test]
fn ordinary_complex_single_and_double_sum_both_components_and_preserve_storage() {
    let provider = InProcessProvider::new();
    for input_data in [
        HostNumericDataView::F32(&[16_777_216.0, 3.0, 1.0, 4.0, -16_777_216.0, 5.0]),
        HostNumericDataView::F64(&[16_777_216.0, 3.0, 1.0, 4.0, -16_777_216.0, 5.0]),
    ] {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: input_data,
                shape: &[3, 1],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .unwrap();
        provider.reset_telemetry();
        let outputs = [
            block_on(provider.reduce_sum(&input)).unwrap(),
            block_on(provider.reduce_sum_dim(&input, 0)).unwrap(),
        ];
        // Observe reduction transfers before the oracle downloads its results.
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        for output in outputs {
            assert_eq!(
                output.physical().storage(),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(output.physical().element_type(), NumericElementType::F64);
            assert_eq!(
                block_on(provider.download_numeric(&output)).unwrap().data,
                HostNumericDataOwned::F64(vec![1.0, 12.0])
            );
        }
    }
    let matrix = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1., 2., 3., 4., 5., 6., 7., 8.]),
            shape: &[2, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let rows = block_on(provider.reduce_sum_dim(&matrix, 1)).unwrap();
    assert_eq!(rows.shape(), [2, 1]);
    assert_eq!(
        block_on(provider.download_numeric(&rows)).unwrap().data,
        HostNumericDataOwned::F64(vec![6., 8., 10., 12.])
    );
}

#[test]
#[ignore = "large-count arithmetic oracle; explicitly run in numerical qualification"]
fn logical_single_payload_count_above_f32_integer_precision_is_exact_double() {
    let count = 16_777_217;
    let values = vec![1.0_f32; count];
    let mut output = [0.0_f64];
    accumulate_sum_as_double(
        &values,
        &mut output,
        &[count, 1],
        None,
        GpuTensorLayout::Contiguous,
        1,
    );
    assert_eq!(output, [count as f64]);
}

#[test]
fn ordinary_single_reductions_accumulate_in_double_and_preserve_dimensions() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1., 2., 3., 4., 5., 6.]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    provider.reset_telemetry();
    for (dim, shape, values) in [
        (0, &[1, 3][..], &[3., 7., 11.][..]),
        (1, &[2, 1][..], &[9., 12.][..]),
    ] {
        let before = provider.telemetry_snapshot().download_bytes;
        let result = block_on(provider.reduce_sum_dim(&input, dim)).unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, before);
        assert_eq!(result.shape(), shape);
        assert_eq!(result.physical().element_type(), NumericElementType::F64);
        assert_eq!(
            block_on(provider.download_numeric(&result)).unwrap().data,
            HostNumericDataOwned::F64(values.to_vec())
        );
    }
    assert!(input
        .with_view(
            GpuTensorView::contiguous(vec![3, 2]).with_layout(GpuTensorLayout::Transpose(
                TransposeInfo {
                    base_rows: 2,
                    base_cols: 3,
                }
            ),)
        )
        .is_err());
    // This provider owns a physical transpose, not an admitted lazy view.
    let transposed = provider.transpose(&input).unwrap();
    let before = provider.telemetry_snapshot().download_bytes;
    let result = block_on(provider.reduce_sum_dim(&transposed, 0)).unwrap();
    assert_eq!(provider.telemetry_snapshot().download_bytes, before);
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(
        block_on(provider.download_numeric(&result)).unwrap().data,
        HostNumericDataOwned::F64(vec![9., 12.])
    );
    assert!(input.allocation_lease().validate_live().is_ok());
}

#[test]
fn double_logical_and_empty_reductions_keep_existing_contract() {
    let provider = InProcessProvider::new();
    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0, 1, 1, 0, 1],
            shape: &[2, 3],
        })
        .unwrap();
    let result = block_on(provider.reduce_sum_dim(&logical, 0)).unwrap();
    assert_eq!(result.shape(), &[1, 3]);
    assert!(!runmat_accelerate_api::handle_is_logical(&result));
    assert_eq!(
        block_on(provider.download_numeric(&result)).unwrap().data,
        HostNumericDataOwned::F64(vec![1., 2., 1.])
    );
    let empty = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[]),
            shape: &[0, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let result = block_on(provider.reduce_sum_dim(&empty, 0)).unwrap();
    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(
        block_on(provider.download_numeric(&result)).unwrap().data,
        HostNumericDataOwned::F64(vec![0., 0., 0.])
    );
    let foreign = InProcessProvider::new()
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(block_on(provider.reduce_sum_dim(&foreign, 0)).is_err());
}

#[test]
fn native_double_both_dimensions_and_empty_second_axis_have_exact_geometry() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1., 2., 3., 4., 5., 6.]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    provider.reset_telemetry();
    for (dim, shape, expected) in [
        (0, &[1, 3][..], &[3., 7., 11.][..]),
        (1, &[2, 1][..], &[9., 12.][..]),
    ] {
        let output = block_on(provider.reduce_sum_dim(&input, dim)).unwrap();
        assert_eq!(output.shape(), shape);
        assert_eq!(output.physical().element_type(), NumericElementType::F64);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(
            block_on(provider.download_numeric(&output)).unwrap().data,
            HostNumericDataOwned::F64(expected.to_vec())
        );
        provider.reset_telemetry();
    }

    let empty = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[]),
            shape: &[2, 0],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    for (dim, shape, expected) in [(0, &[1, 0][..], &[][..]), (1, &[2, 1][..], &[0., 0.][..])] {
        let output = block_on(provider.reduce_sum_dim(&empty, dim)).unwrap();
        assert_eq!(output.shape(), shape);
        assert_eq!(output.physical().element_type(), NumericElementType::F64);
        assert_eq!(
            block_on(provider.download_numeric(&output)).unwrap().data,
            HostNumericDataOwned::F64(expected.to_vec())
        );
    }
}

#[test]
fn native_means_preserve_single_and_double_without_a_host_roundtrip() {
    let provider = InProcessProvider::new();
    for data in [
        HostNumericDataView::F32(&[1.0, 3.0, 5.0, 7.0]),
        HostNumericDataView::F64(&[1.0, 3.0, 5.0, 7.0]),
    ] {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data,
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        provider.reset_telemetry();
        let first = block_on(provider.reduce_mean_dim(&input, 0))
            .unwrap()
            .expect("admitted mean");
        let second = block_on(provider.reduce_mean_dim(&input, 1))
            .unwrap()
            .expect("admitted mean");
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(first.shape(), &[1, 2]);
        assert_eq!(second.shape(), &[2, 1]);
        assert_eq!(
            first.physical().element_type(),
            input.physical().element_type()
        );
        assert_eq!(
            second.physical().element_type(),
            input.physical().element_type()
        );
        match block_on(provider.download_numeric(&first)).unwrap().data {
            HostNumericDataOwned::F32(values) => assert_eq!(values, [2.0, 6.0]),
            HostNumericDataOwned::F64(values) => assert_eq!(values, [2.0, 6.0]),
            other => panic!("mean changed physical class: {other:?}"),
        }
        match block_on(provider.download_numeric(&second)).unwrap().data {
            HostNumericDataOwned::F32(values) => assert_eq!(values, [3.0, 5.0]),
            HostNumericDataOwned::F64(values) => assert_eq!(values, [3.0, 5.0]),
            other => panic!("mean changed physical class: {other:?}"),
        }
    }
}
