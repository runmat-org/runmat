use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate_api::{HostTensorView, WgpuScopeErrors};

async fn provider() -> WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("FFT qualification requires an actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("FFT qualification requires an actual WebGPU adapter");
    if let Ok(requested) = std::env::var("RUNMAT_WGPU_FORCE_PRECISION") {
        let actual = match provider.precision {
            NumericPrecision::F32 => "f32",
            NumericPrecision::F64 => "f64",
        };
        assert_eq!(requested.to_ascii_lowercase(), actual);
    }
    provider
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter; run explicitly with --ignored"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn fft_debug_scope_retains_validation_source_and_balances_scopes() {
    let provider = provider().await;
    let (layout, pipeline) = provider
        .with_emission(|emission, _| {
            let layout =
                emission
                    .device()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("fft-scope-test-layout"),
                        entries: &[],
                    });
            let pipeline_layout =
                crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
                    emission,
                    "fft-scope-test-pipeline-layout",
                    &layout,
                );
            let module = crate::backend::wgpu::cache::factory::create_shader_module(
                emission,
                "fft-scope-test-shader",
                "@compute @workgroup_size(1) fn main() {}",
            );
            let pipeline = crate::backend::wgpu::cache::factory::create_compute_pipeline(
                emission,
                "fft-scope-test-pipeline",
                &pipeline_layout,
                &module,
            );
            (layout, pipeline)
        })
        .unwrap();
    let prepare = |workgroups| {
        let layout = &layout;
        let pipeline = &pipeline;
        move |emission: &WgpuEmission<'_>| {
            let bind = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fft-scope-test-bind"),
                    layout,
                    entries: &[],
                });
            Ok((pipeline, bind, workgroups))
        }
    };
    let error = provider
        .fft_dispatch_with_debug(true, "invalid-workgroup-count", prepare(u32::MAX))
        .await
        .expect_err("invalid dispatch must not publish success");
    assert!(!error.is_unsupported());
    assert!(error.to_string().contains("invalid-workgroup-count"));
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    let scope = loop {
        if let Some(scope) = cause.downcast_ref::<WgpuScopeErrors>() {
            break scope;
        }
        cause = cause
            .source()
            .expect("original scope errors remain reachable");
    };
    scope.with_errors(|errors| {
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|error| matches!(error, wgpu::Error::Validation { .. })));
    });
    // A later valid emission must not inherit the previous diagnostic scope.
    provider
        .fft_dispatch_with_debug(true, "valid-after-error", prepare(1))
        .await
        .unwrap();
    provider
        .fft_dispatch_with_debug(false, "normal-after-error", prepare(1))
        .await
        .unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter; run explicitly with --ignored"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn fft_stage_families_match_dft_and_preserve_source_ownership() {
    let provider = provider().await;
    for len in [4usize, 9, 25, 6, 7] {
        let values = (0..len)
            .map(|index| (index % 5) as f64 - 1.5)
            .collect::<Vec<_>>();
        let input = provider
            .upload_exec(&HostTensorView {
                data: &values,
                shape: &[len, 1],
            })
            .unwrap();
        let source_weak = input.allocation_lease().downgrade();
        // An unpolled operation owns no hidden submission or published output.
        drop(provider.fft_dim_exec(&input, None, 0));
        assert!(source_weak.acquire().is_some());
        let output = provider.fft_dim_exec(&input, None, 0).await.unwrap();
        assert!(!output.shares_allocation_with(&input));
        let actual = provider.download_exec(&output).await.unwrap().data;
        assert_eq!(actual.len(), 2 * len);
        for frequency in 0..len {
            let mut expected = Complex::new(0.0, 0.0);
            for (index, value) in values.iter().enumerate() {
                let angle = -std::f64::consts::TAU * (frequency * index) as f64 / len as f64;
                expected += Complex::new(angle.cos(), angle.sin()) * value;
            }
            assert!(
                (actual[2 * frequency] - expected.re).abs() < 2.0e-4,
                "len={len} bin={frequency} real"
            );
            assert!(
                (actual[2 * frequency + 1] - expected.im).abs() < 2.0e-4,
                "len={len} bin={frequency} imaginary"
            );
        }
        assert_eq!(provider.download_exec(&input).await.unwrap().data, values);
        let output_weak = output.allocation_lease().downgrade();
        drop(output);
        assert!(output_weak.acquire().is_none());
        drop(input);
        assert!(source_weak.acquire().is_none());
    }
}
