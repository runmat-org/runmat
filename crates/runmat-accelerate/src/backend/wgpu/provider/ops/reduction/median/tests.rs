use super::*;

#[test]
fn empty_output_does_not_allocate_dimension_sized_scratch() {
    for (rows, cols, axis) in [(usize::MAX, 0, 0), (0, usize::MAX, 1)] {
        let output = medians(
            &[],
            rows,
            cols,
            axis,
            f64::NAN,
            f64::is_nan,
            compute_median_inplace,
        );
        assert!(output.is_empty());
    }
}

#[test]
fn native_slices_obey_zero_based_axes_and_single_arithmetic() {
    let values = [1.0f32, 100.0, 9.0, 8.0, 3.0, -20.0];
    assert_eq!(
        medians(&values, 2, 3, 0, f32::NAN, f32::is_nan, median_f32),
        [50.5, 8.5, -8.5]
    );
    assert_eq!(
        medians(&values, 2, 3, 1, f32::NAN, f32::is_nan, median_f32),
        [3.0, 8.0]
    );
    assert_eq!(
        medians(&values, 6, 1, 0, f32::NAN, f32::is_nan, median_f32),
        [5.5]
    );
    // The owning host reducer computes the even pair in native precision.
    assert!(median_f32(&mut [f32::MAX, f32::MAX]).is_infinite());
    assert!(medians(&[1.0, f32::NAN], 2, 1, 0, f32::NAN, f32::is_nan, median_f32)[0].is_nan());
    assert_eq!(
        medians(&[-0.0f32], 1, 1, 0, f32::NAN, f32::is_nan, median_f32)[0].to_bits(),
        (-0.0f32).to_bits()
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires an actual WGPU adapter; missing hardware is a failure"]
async fn native_median_axes_global_and_input_ownership() {
    use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView};
    let provider = WgpuProvider::new(Default::default()).expect("required WGPU adapter");
    // Both storage classes use real transfer allocations, regardless of the
    // device's preferred shader precision. Median remains host-mediated.
    for data in [
        HostNumericDataOwned::F32(vec![1.0, 100.0, 9.0, 8.0, 3.0, -20.0]),
        HostNumericDataOwned::F64(vec![1.0, 100.0, 9.0, 8.0, 3.0, -20.0]),
    ] {
        let host = HostNumericTensorOwned {
            data,
            shape: vec![2, 3],
            storage: GpuTensorStorage::Real,
        };
        let input = provider
            .upload_numeric_exec(&host.as_view())
            .expect("native upload");
        let caller = input.clone();
        for (axis, shape, expected) in [
            (Some(0), vec![1, 3], vec![50.5, 8.5, -8.5]),
            (Some(1), vec![2, 1], vec![3.0, 8.0]),
            (None, vec![1, 1], vec![5.5]),
        ] {
            let output = match axis {
                Some(axis) => provider.reduce_median_dim_exec(&input, axis).await,
                None => provider.reduce_median_exec(&input).await,
            }
            .expect("supported host-mediated median");
            assert_eq!(output.shape(), shape);
            assert_eq!(
                output.physical().element_type(),
                input.physical().element_type()
            );
            assert!(!output.shares_allocation_with(&input));
            let actual = provider
                .download_numeric_exec(&output)
                .await
                .expect("typed output download");
            assert_eq!(actual.shape, shape);
            match (actual.data, &host.data) {
                (HostNumericDataOwned::F32(actual), HostNumericDataOwned::F32(_)) => {
                    assert_eq!(
                        actual,
                        expected.iter().map(|&v| v as f32).collect::<Vec<_>>()
                    );
                }
                (HostNumericDataOwned::F64(actual), HostNumericDataOwned::F64(_)) => {
                    assert_eq!(actual, expected)
                }
                other => panic!("median changed native storage: {other:?}"),
            }
            output.release().expect("release output allocation");
        }
        let invalid_axis = provider
            .reduce_median_dim_exec(&input, 2)
            .await
            .unwrap_err();
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &invalid_axis
        ));
        drop(input);
        let surviving = provider
            .download_numeric_exec(&caller)
            .await
            .expect("caller retains original allocation");
        assert_eq!(surviving.shape, host.shape);
        match (surviving.data, host.data) {
            (HostNumericDataOwned::F32(actual), HostNumericDataOwned::F32(expected)) => {
                assert_eq!(actual, expected)
            }
            (HostNumericDataOwned::F64(actual), HostNumericDataOwned::F64(expected)) => {
                assert_eq!(actual, expected)
            }
            other => panic!("input native payload changed: {other:?}"),
        }
        caller.release().expect("release last input owner");
    }
    let nd = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0]),
            shape: &[2, 1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let error = provider.reduce_median_dim_exec(&nd, 0).await.unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
    nd.release().unwrap();
    let empty = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[]),
            shape: &[0, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    for axis in [None, Some(0), Some(1)] {
        let error = provider.median_native(&empty, axis).await.unwrap_err();
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &error
        ));
    }
    empty.release().unwrap();
}
