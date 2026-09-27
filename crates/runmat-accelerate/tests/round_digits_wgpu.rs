#![cfg(feature = "wgpu")]
use runmat_accelerate_api::{
    allocation::AllocationReleaseOutcome, AccelProvider, GpuTensorHandle, HostTensorView,
    ProviderPrecision,
};

fn release_final(handle: GpuTensorHandle) {
    let weak = handle.allocation_lease().downgrade();
    assert_eq!(handle.release().unwrap(), AllocationReleaseOutcome::Retired);
    assert!(weak.acquire().is_none());
}

#[tokio::test]
#[ignore = "requires an actual compatible WGPU adapter"]
async fn round_digits_hardware_preserves_integral_extreme_and_tiny_values() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("digit-rounding qualification requires a compatible WGPU adapter");
    let single = provider.precision() == ProviderPrecision::F32;
    let represented = |value: f64| {
        if single {
            f64::from(value as f32)
        } else {
            value
        }
    };
    let large = if single {
        8_388_609.0
    } else {
        4_503_599_627_370_497.0
    };
    for (source, digits, significant, expected) in [
        (large, 0, false, large),
        (-large, 0, false, -large),
        (represented(1e30), 10, false, represented(1e30)),
        (23.75, 20, false, 23.75),
        (23.75, 22, true, 23.75),
        (23.75, i32::MAX, true, 23.75),
        (-23.75, i32::MIN, false, -0.0),
        (1253.0, 2, true, 1300.0),
        (
            represented(750_000_000_000.0),
            -11,
            false,
            represented(800_000_000_000.0),
        ),
        (150_000.0, -5, false, 200_000.0),
        (2.05, 1, false, represented(2.1)),
        (-0.0, 0, false, -0.0),
        (f64::INFINITY, 1, false, f64::INFINITY),
        (f64::NEG_INFINITY, 1, true, f64::NEG_INFINITY),
        (f64::NAN, 1, true, f64::NAN),
    ] {
        let input = provider
            .upload(&HostTensorView {
                data: &[source],
                shape: &[1, 1, 1],
            })
            .unwrap();
        let output = provider
            .round_digits(&input, digits, significant)
            .await
            .unwrap();
        assert_eq!(output.shape(), input.shape());
        assert_ne!(output.allocation_identity(), input.allocation_identity());
        assert!(!output.shares_allocation_with(&input));
        let actual = provider.download(&output).await.unwrap().data;
        assert_eq!(actual.len(), 1);
        if expected.is_nan() {
            assert!(actual[0].is_nan());
        } else {
            assert_eq!(
                actual[0].to_bits(),
                expected.to_bits(),
                "{source}, N={digits}, significant={significant}"
            );
        }
        release_final(output);
        let original = provider.download(&input).await.unwrap().data;
        assert_eq!(original.len(), 1);
        if source.is_nan() {
            assert!(original[0].is_nan());
        } else {
            assert_eq!(original[0].to_bits(), represented(source).to_bits());
        }
        release_final(input);
    }

    let (tiny, digits, quantum, ulp) = if single {
        (
            f64::from(1.234e-40_f32),
            40,
            f64::from(1e-40_f32),
            f64::from(f32::from_bits(1)),
        )
    } else {
        (1.234e-310, 310, 1e-310, f64::from_bits(1))
    };
    let input = provider
        .upload(&HostTensorView {
            data: &[tiny, -tiny],
            shape: &[1, 1, 2],
        })
        .unwrap();
    let output = provider.round_digits(&input, digits, false).await.unwrap();
    let actual = provider.download(&output).await.unwrap().data;
    assert_eq!(actual.len(), 2);
    assert!((actual[0] - quantum).abs() <= 4.0 * ulp);
    assert!((actual[1] + quantum).abs() <= 4.0 * ulp);
    release_final(output);
    release_final(input);
}

#[tokio::test]
#[ignore = "requires an actual compatible WGPU adapter"]
async fn round_digits_preserves_lazy_transpose_for_real_and_complex_components() {
    use runmat_accelerate_api::{
        upload_numeric_protecting, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
        HostNumericTensorView,
    };
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("digit-rounding qualification requires a compatible WGPU adapter");
    for complex in [false, true] {
        let original = [1.25, 2.75, 3.25, 4.75, 5.25, 6.75];
        let values: Vec<f64> = original
            .iter()
            .flat_map(|&value| {
                if complex {
                    vec![value, -value]
                } else {
                    vec![value]
                }
            })
            .collect();
        let singles: Vec<f32> = values.iter().map(|value| *value as f32).collect();
        let data = if provider.precision() == ProviderPrecision::F32 {
            HostNumericDataView::F32(&singles)
        } else {
            HostNumericDataView::F64(&values)
        };
        let storage = if complex {
            GpuTensorStorage::ComplexInterleaved
        } else {
            GpuTensorStorage::Real
        };
        let input = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data,
                shape: &[2, 3],
                storage,
            },
            &[],
        )
        .unwrap();
        let transposed = provider.transpose(&input).unwrap();
        let output = provider.round_digits(&transposed, 0, false).await.unwrap();
        assert_eq!(
            runmat_accelerate_api::handle_transpose_info(&output),
            runmat_accelerate_api::handle_transpose_info(&transposed)
        );
        let host = provider.download_numeric(&output).await.unwrap();
        assert_eq!(host.shape, vec![3, 2]);
        assert_eq!(host.storage, storage);
        let actual: Vec<f64> = match host.data {
            HostNumericDataOwned::F64(values) => values,
            HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect(),
            _ => panic!("floating output"),
        };
        let expected: Vec<f64> = [1.0, 3.0, 5.0, 3.0, 5.0, 7.0]
            .into_iter()
            .flat_map(|value| {
                if complex {
                    vec![value, -value]
                } else {
                    vec![value]
                }
            })
            .collect();
        assert_eq!(actual, expected);
        assert!(!output.shares_allocation_with(&transposed));
        release_final(output);
        let input_weak = input.allocation_lease().downgrade();
        if complex {
            assert!(!transposed.shares_allocation_with(&input));
            release_final(transposed);
        } else {
            assert!(transposed.shares_allocation_with(&input));
            assert_ne!(transposed.view_identity(), input.view_identity());
            assert_eq!(
                transposed.release().unwrap(),
                AllocationReleaseOutcome::ReferenceReleased
            );
        }
        assert!(input_weak.acquire().is_some());
        let original = provider.download_numeric(&input).await.unwrap();
        assert_eq!(original.shape, [2, 3]);
        assert_eq!(original.storage, storage);
        match original.data {
            HostNumericDataOwned::F32(actual) => assert_eq!(actual, singles),
            HostNumericDataOwned::F64(actual) => assert_eq!(actual, values),
            _ => panic!("original floating storage must survive output retirement"),
        }
        release_final(input);
        assert!(input_weak.acquire().is_none());
    }
}
