use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate_api::{HostTensorView, WgpuScopeErrors};

#[path = "emission_pending_tests.rs"]
mod pending_tests;

#[cfg(not(target_arch = "wasm32"))]
#[path = "emission_retention_tests.rs"]
mod retention_tests;

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

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires an actual WGPU adapter and timestamp-capability qualification"]
async fn submitted_compute_pass_reports_two_physical_dispatches_and_optional_device_time() {
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationStatus, ProviderOperationReceipt,
    };

    let provider = provider().await;
    let (layout, pipeline) = scope_test_pipeline(&provider);
    let progress = provider
        .observation_progress
        .clone()
        .expect("native progress owner required");
    let mut receipt =
        ProviderOperationReceipt::with_wgpu_progress(provider.observation_budget.clone(), progress);
    with_provider_observation(Some(receipt.recorder()), || {
        provider.with_emission(|emission, pending| -> anyhow::Result<()> {
            let bind = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("two-physical-dispatches-bind"),
                    layout: &layout,
                    entries: &[],
                });
            let mut encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("two-physical-dispatches"),
                },
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("two-physical-dispatches-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                pass.dispatch_workgroups(1, 1, 1);
            }
            provider.submit(emission, pending, encoder)?;
            Ok(())
        })
    })
    .unwrap()
    .unwrap();
    receipt.publish().unwrap();
    let observed = futures::future::poll_fn(|cx| receipt.poll_completed(cx))
        .await
        .expect("one receipt");
    assert!(matches!(
        observed.status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
    assert_eq!(observed.verified_wgpu_dispatches, Some(2));
    let query_allocations = observed
        .wgpu_query_allocations
        .expect("physical query disposition is published by the submission owner");
    if provider.timestamp_capability.is_some() {
        assert_eq!(query_allocations.query_sets, 1);
        assert_eq!(query_allocations.timestamp_slots, 16);
        assert_eq!(query_allocations.resolve_requested_bytes, 16);
        assert_eq!(query_allocations.staging_requested_bytes, 16);
        assert_eq!(query_allocations.map_callbacks, 1);
    } else {
        assert_eq!(query_allocations.unsupported_submissions, 1);
    }
    assert_eq!(
        observed.verified_wgpu_compute_elapsed.is_some(),
        provider.timestamp_capability.is_some(),
    );
}

fn scope_test_pipeline(provider: &WgpuProvider) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    provider
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
            )
            .expect("prepare FFT scope test shader");
            let pipeline = crate::backend::wgpu::cache::factory::create_compute_pipeline(
                emission,
                "fft-scope-test-pipeline",
                &pipeline_layout,
                &module,
            );
            (layout, pipeline)
        })
        .unwrap()
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter; run explicitly with --ignored"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn fft_debug_scope_retains_validation_source_and_balances_scopes() {
    let provider = provider().await;
    let (layout, pipeline) = scope_test_pipeline(&provider);
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
            Ok(FftPreparedDispatch::new(
                pipeline,
                bind,
                workgroups,
                FftDispatchResources::new(&[], &[])?,
            ))
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn suspended_browser_fft_scope_can_be_cancelled_before_promise_delivery() {
    use std::{
        cell::Cell,
        future::Future,
        task::{Context, Poll},
    };

    let provider = provider().await;
    let (layout, pipeline) = scope_test_pipeline(&provider);
    let preparations = Cell::new(0usize);
    let prepare = |emission: &WgpuEmission<'_>| {
        preparations.set(preparations.get() + 1);
        let bind = emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft-cancelled-scope-bind"),
                layout: &layout,
                entries: &[],
            });
        Ok(FftPreparedDispatch::new(
            &pipeline,
            bind,
            1,
            FftDispatchResources::new(&[], &[])?,
        ))
    };
    let mut pending = Box::pin(provider.fft_dispatch_with_debug(
        true,
        "cancelled-before-scope-delivery",
        prepare,
    ));
    let waker = futures::task::noop_waker();
    // No JS event-loop yield occurs between production submission, polling the
    // detached popErrorScope promises, and cancellation. Native may resolve
    // immediately; this is deliberately a browser-only suspension witness.
    assert!(matches!(
        pending.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(
        preparations.get(),
        1,
        "production dispatch was actually entered"
    );
    provider
        .access
        .with_shared(|_| ())
        .expect("still-pending receipt must not retain access");
    drop(pending);
    // Cancellation cannot retain the exclusive emission permit. The dispatch
    // helper publishes no tensor handles; its temporary bind group and popped
    // receipt are dropped here, while WGPU owns any submitted command resources.
    provider
        .access
        .with_shared(|_| ())
        .expect("cancelled receipt released access");
    provider
        .fft_dispatch_with_debug(true, "valid-after-pending-cancellation", prepare)
        .await
        .expect("later debug scope must resolve without stale scope errors");
    provider
        .fft_dispatch_with_debug(false, "shared-after-pending-cancellation", prepare)
        .await
        .expect("normal emission remains usable after scope cancellation");
    assert_eq!(preparations.get(), 3);
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

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter; run explicitly with --ignored"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn fft_forward_and_inverse_scalar_shape_matches_canonical_host_contract() {
    let provider = provider().await;
    for shape in [vec![], vec![1], vec![1, 1]] {
        let input = provider
            .upload_exec(&HostTensorView {
                data: &[3.25],
                shape: &shape,
            })
            .unwrap();
        for inverse in [false, true] {
            let output = if inverse {
                provider.ifft_dim_exec(&input, None, 0).await
            } else {
                provider.fft_dim_exec(&input, None, 0).await
            }
            .unwrap();
            assert_eq!(
                output.shape(),
                &[1, 1],
                "inverse={inverse}, input={shape:?}"
            );
            let data = provider.download_exec(&output).await.unwrap();
            assert_eq!(data.data, vec![3.25, 0.0]);
            assert!(!output.shares_allocation_with(&input));
        }
        assert_eq!(
            provider.download_exec(&input).await.unwrap().data,
            vec![3.25]
        );
    }
}
