#![cfg(feature = "wgpu")]
#[path = "support/retirement.rs"]
mod retirement;
use runmat_accelerate_api::{AccelProvider, GpuTensorStorage, HostTensorView, ProviderPrecision};

#[tokio::test]
#[ignore = "requires an actual compatible WGPU adapter"]
async fn actual_logarithm_hooks_match_represented_inputs_and_own_distinct_outputs() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("logarithm qualification requires a compatible WGPU adapter");
    let single = provider.precision() == ProviderPrecision::F32;
    let tolerance = if single { 2e-5 } else { 1e-11 };
    for operation in 0..4 {
        let source = [0.0, 1e-10, -1.0, -2.0, 2.0, 16.0, f64::INFINITY, f64::NAN];
        let input = provider
            .upload(&HostTensorView {
                data: &source,
                shape: &[2, 2, 2],
            })
            .unwrap();
        let before = input.wire_descriptor();
        let output = match operation {
            0 => provider.unary_log2(&input).await,
            1 => provider.unary_log10(&input).await,
            2 => provider.unary_log1p(&input).await,
            _ => provider.unary_log(&input).await,
        }
        .unwrap();
        assert_ne!(input.buffer_id(), output.buffer_id());
        assert_eq!(input.wire_descriptor(), before);
        assert_eq!(output.device_id(), input.device_id());
        assert_eq!(output.shape(), input.shape());
        assert_eq!(
            output.descriptor().element_type,
            input.descriptor().element_type
        );
        assert_eq!(output.descriptor().storage, Some(GpuTensorStorage::Real));
        for (actual, original) in provider
            .download(&output)
            .await
            .unwrap()
            .data
            .iter()
            .zip(source)
        {
            let represented = if single {
                f64::from(original as f32)
            } else {
                original
            };
            let expected = match operation {
                0 => represented.log2(),
                1 => represented.log10(),
                2 => represented.ln_1p(),
                _ => represented.ln(),
            };
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else if expected.is_infinite() || expected == 0.0 {
                assert_eq!(*actual, expected);
            } else {
                assert!(
                    (actual / expected - 1.0).abs() < tolerance,
                    "operation {operation}: {actual} != {expected}"
                );
            }
        }
        provider.download(&input).await.unwrap();
        retirement::release_all([output]);
        retirement::release_all([input]);
    }
}
