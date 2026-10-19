#![cfg(feature = "wgpu")]

#[path = "support/retirement.rs"]
mod retirement;
use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

#[test]
fn actual_wgpu_classifies_nonfinite_payloads_and_maximum_finite_values() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required for floating classification qualification");
    let maximum = match provider.precision() {
        ProviderPrecision::F32 => f64::from(f32::MAX),
        ProviderPrecision::F64 => f64::MAX,
    };
    let values = [
        0.0,
        -0.0,
        maximum,
        -maximum,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        1.0,
        -1.0,
    ];
    let input = provider
        .upload(&HostTensorView {
            data: &values,
            shape: &[2, 1, 5],
        })
        .unwrap();
    for (output, expected) in [
        (
            provider.logical_isnan(&input).unwrap(),
            values.map(f64::is_nan),
        ),
        (
            provider.logical_isinf(&input).unwrap(),
            values.map(f64::is_infinite),
        ),
        (
            provider.logical_isfinite(&input).unwrap(),
            values.map(f64::is_finite),
        ),
    ] {
        assert_eq!(output.shape(), vec![2, 1, 5]);
        assert_ne!(output.buffer_id(), input.buffer_id());
        let downloaded = block_on(provider.download(&output)).unwrap();
        assert_eq!(
            downloaded.data,
            expected.map(|value| if value { 1.0 } else { 0.0 })
        );
        retirement::release_all([output]);
    }
    let unchanged = block_on(provider.download(&input)).unwrap();
    for (actual, expected) in unchanged.data.iter().zip(values) {
        assert!(actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()));
    }
    retirement::release_all([input]);
}
