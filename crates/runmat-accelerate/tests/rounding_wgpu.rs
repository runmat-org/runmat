#![cfg(feature = "wgpu")]
use runmat_accelerate_api::{HostTensorView, ProviderPrecision};

#[tokio::test]
#[ignore = "requires an actual compatible WGPU adapter"]
async fn round_hardware_preserves_integral_values_and_rounds_ties_away() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("rounding qualification requires a compatible WGPU adapter");
    let large = if provider.precision() == ProviderPrecision::F32 {
        8_388_609.0
    } else {
        4_503_599_627_370_497.0
    };
    let source = [
        0.5,
        -0.5,
        2.5,
        -2.5,
        large,
        -large,
        0.0,
        -0.0,
        -0.25,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let expected = [
        1.0,
        -1.0,
        3.0,
        -3.0,
        large,
        -large,
        0.0,
        -0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let input = provider
        .upload(&HostTensorView {
            data: &source,
            shape: &[2, 2, 3],
        })
        .unwrap();
    let output = provider.unary_round(&input).await.unwrap();
    assert_eq!(output.shape, input.shape);
    assert_ne!(output.buffer_id, input.buffer_id);
    for (actual, expected) in provider
        .download(&output)
        .await
        .unwrap()
        .data
        .into_iter()
        .zip(expected)
    {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    provider.download(&input).await.unwrap();
    provider.free(&output).unwrap();
    provider.free(&input).unwrap();
}
