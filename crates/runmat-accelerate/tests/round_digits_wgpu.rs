#![cfg(feature = "wgpu")]
use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

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
        assert_eq!(output.shape, input.shape);
        assert_ne!(output.buffer_id, input.buffer_id);
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
        provider.download(&input).await.unwrap();
        provider.free(&output).unwrap();
        provider.free(&input).unwrap();
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
    provider.free(&output).unwrap();
    provider.free(&input).unwrap();
}
