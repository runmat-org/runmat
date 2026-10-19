#![cfg(feature = "wgpu")]

use runmat_accelerate_api::{
    AccelProvider, DeepLearningPool1DCapability, DeepLearningPool1DRequest,
    DeepLearningPool1DResultAxis, DeepLearningPool1DRoute, GpuTensorStorage, HostTensorView,
    ProviderPrecision,
};

#[tokio::test]
#[ignore = "requires an actual compatible WGPU adapter"]
async fn stable_resident_global_pool_retains_precision_shape_owner_and_special_values() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("WGPU pooling qualification requires a compatible adapter");
    let single = provider.precision() == ProviderPrecision::F32;
    let tolerance = if single { 2e-5 } else { 2e-12 };
    for (shape, axis, values, expected) in [
        (
            vec![2, 3, 4],
            2,
            (1..=24).map(f64::from).collect::<Vec<_>>(),
            vec![10., 11., 12., 13., 14., 15.],
        ),
        (
            vec![3, 2, 2],
            0,
            (1..=12).map(f64::from).collect::<Vec<_>>(),
            vec![2., 5., 8., 11.],
        ),
    ] {
        let input = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .unwrap();
        let before = input.wire_descriptor();
        provider.reset_telemetry();
        let mut source_shape = shape.clone();
        source_shape[axis] = 1;
        let route = DeepLearningPool1DRoute {
            reduction: DeepLearningPool1DRequest {
                axis_zero_based: axis,
            },
            result_axis: DeepLearningPool1DResultAxis::KeepSingleton,
            target_shape: source_shape.clone(),
            target_to_source: (0..source_shape.len()).collect(),
            source_shape,
        };
        assert_eq!(
            provider
                .deep_learning_global_average_pool_1d_capability(&input, &route)
                .unwrap(),
            DeepLearningPool1DCapability::Ready
        );
        let mut malformed = route.clone();
        malformed.target_to_source.fill(0);
        assert!(provider
            .deep_learning_global_average_pool_1d_capability(&input, &malformed)
            .is_err());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let output = provider
            .deep_learning_global_average_pool_1d(
                &input,
                DeepLearningPool1DRequest {
                    axis_zero_based: axis,
                },
            )
            .await
            .unwrap();
        let mut output_shape = shape.clone();
        output_shape[axis] = 1;
        assert_eq!(output.shape(), output_shape);
        assert_eq!(output.device_id(), input.device_id());
        assert_eq!(
            output.descriptor().element_type,
            input.descriptor().element_type
        );
        assert_eq!(output.descriptor().storage, Some(GpuTensorStorage::Real));
        assert!(!output.shares_allocation_with(&input));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(input.wire_descriptor(), before);
        let actual = provider.download(&output).await.unwrap();
        for (actual, expected) in actual.data.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} != {expected}"
            );
        }
        output.release().unwrap();
        input.release().unwrap();
    }

    let large = if single { 2.0e38 } else { 1.0e308 };
    let maximum = if single {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    for (values, finite_expected) in [
        (vec![large, large], Some(large)),
        (vec![maximum, maximum], Some(maximum)),
        (vec![maximum / 2.0, maximum / 2.0], Some(maximum / 2.0)),
        (vec![maximum, maximum, -maximum], Some(maximum / 3.0)),
        (vec![large, -large], Some(0.0)),
        (vec![f64::INFINITY, 1.0], None),
        (vec![f64::INFINITY, f64::NEG_INFINITY], None),
        (vec![f64::NAN, 1.0], None),
    ] {
        let input = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &[values.len(), 1],
            })
            .unwrap();
        let output = provider
            .deep_learning_global_average_pool_1d(
                &input,
                DeepLearningPool1DRequest { axis_zero_based: 0 },
            )
            .await
            .unwrap();
        let actual = provider.download(&output).await.unwrap().data[0];
        if let Some(expected) = finite_expected {
            assert!(actual.is_finite(), "finite mean overflowed: {actual}");
            if expected == 0.0 {
                assert_eq!(actual, 0.0);
            } else {
                assert!((actual / expected - 1.0).abs() < tolerance);
            }
        } else if values[0].is_infinite() && values[1].is_finite() {
            assert_eq!(actual, f64::INFINITY);
        } else {
            assert!(actual.is_nan(), "expected NaN, got {actual}");
        }
        output.release().unwrap();
        input.release().unwrap();
    }

    for tiny in if single {
        [1e-10, 1e-38]
    } else {
        [1e-20, 1e-320]
    } {
        let values = [large, -large, tiny];
        let input = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &[3, 1],
            })
            .unwrap();
        let output = provider
            .deep_learning_global_average_pool_1d(
                &input,
                DeepLearningPool1DRequest { axis_zero_based: 0 },
            )
            .await
            .unwrap();
        let actual = provider.download(&output).await.unwrap().data[0];
        let expected = if single {
            f64::from((tiny as f32) / 3.0_f32)
        } else {
            tiny / 3.0
        };
        let ulp = if single {
            f64::from(f32::from_bits(1))
        } else {
            f64::from_bits(1)
        };
        assert!(
            actual > 0.0 && (actual - expected).abs() <= expected.abs() * tolerance + 4.0 * ulp,
            "cancellation residual lost: {actual} != {expected} (input {tiny})"
        );
        output.release().unwrap();
        input.release().unwrap();
    }
}

#[tokio::test]
#[ignore = "requires an actual compatible WGPU adapter"]
async fn stable_resident_global_pool_materializes_lazy_transpose_without_host_download() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("WGPU pooling qualification requires a compatible adapter");
    let original = provider
        .upload(&HostTensorView {
            data: &[1., 2., 3., 4., 5., 6.],
            shape: &[2, 3],
        })
        .unwrap();
    let transposed = provider.transpose(&original).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    provider.reset_telemetry();
    let output = provider
        .deep_learning_global_average_pool_1d(
            &transposed,
            DeepLearningPool1DRequest { axis_zero_based: 0 },
        )
        .await
        .unwrap();
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(!output.shares_allocation_with(&original));
    assert!(!output.shares_allocation_with(&transposed));
    assert_eq!(provider.download(&output).await.unwrap().data, vec![3., 4.]);
    assert_eq!(
        provider.download(&original).await.unwrap().data,
        vec![1., 2., 3., 4., 5., 6.]
    );
    output.release().unwrap();
    transposed.release().unwrap();
    original.release().unwrap();
}

#[tokio::test]
#[ignore = "requires an actual compatible WGPU adapter"]
async fn stable_resident_global_pool_spans_multiple_workgroups_without_column_overlap() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("WGPU pooling qualification requires a compatible adapter");
    let output_count = 1025usize;
    let values = (0..output_count)
        .flat_map(|column| [column as f64, column as f64 + 10.0])
        .collect::<Vec<_>>();
    let input = provider
        .upload(&HostTensorView {
            data: &values,
            shape: &[2, 5, 205],
        })
        .unwrap();
    provider.reset_telemetry();
    let output = provider
        .deep_learning_global_average_pool_1d(
            &input,
            DeepLearningPool1DRequest { axis_zero_based: 0 },
        )
        .await
        .unwrap();
    assert_eq!(output.shape(), &[1, 5, 205]);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    let actual = provider.download(&output).await.unwrap().data;
    assert_eq!(actual.len(), output_count);
    for (column, &value) in actual.iter().enumerate() {
        assert_eq!(value, column as f64 + 5.0, "column {column}");
    }
    output.release().unwrap();
    input.release().unwrap();
}
