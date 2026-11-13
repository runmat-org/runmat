//! Actual-device ordering witness, not a throughput measurement. Native scope
//! receipts can be immediately ready; browser promises are explicitly pending.
use super::*;
#[cfg(not(target_arch = "wasm32"))]
use crate::backend::wgpu::dispatch::observation::PendingSubmissions;
use runmat_accelerate_api::{HostNumericDataOwned, HostNumericDataView, HostNumericTensorView};

async fn submit_while_receipt_is_outstanding() {
    let provider = provider().await;
    let value = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[3.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let entry = provider.get_entry_raw(&value).unwrap();
    let (layout, pipeline) = provider
        .with_emission(|emission, _| {
            let layout =
                emission
                    .device()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("outstanding-receipt-layout"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
            let pipeline_layout =
                crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
                    emission,
                    "outstanding-receipt-pipeline-layout",
                    &layout,
                );
            let module = crate::backend::wgpu::cache::factory::create_shader_module(
                emission,
                "outstanding-receipt-shader",
                "@group(0) @binding(0) var<storage, read_write> data: array<f32>;
                 @compute @workgroup_size(1) fn main() { data[0] = data[0] * 2.0 + 1.0; }",
            )
            .expect("prepare outstanding receipt test shader");
            let pipeline = crate::backend::wgpu::cache::factory::create_compute_pipeline(
                emission,
                "outstanding-receipt-pipeline",
                &pipeline_layout,
                &module,
            );
            (layout, pipeline)
        })
        .unwrap();
    let prepare = |emission: &WgpuEmission<'_>| {
        emission.ensure_device(&provider.access).unwrap();
        let bind = emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("outstanding-receipt-bind"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry.buffer.as_entire_binding(),
                }],
            });
        Ok(FftPreparedDispatch::new(
            &pipeline,
            bind,
            1,
            FftDispatchResources::new(&[&value], &[&entry.buffer])?,
        ))
    };

    #[cfg(not(target_arch = "wasm32"))]
    let outstanding = {
        // Retain the actual detached receipt unconsumed. Pinned native WGPU
        // resolves its scope futures immediately: do not manufacture suspension.
        let prepared = provider
            .with_emission(|emission, _| prepare(emission))
            .unwrap()
            .unwrap();
        let mut submissions = PendingSubmissions::new(provider.access.clone());
        let receipt = provider
            .access
            .with_scoped(|emission| {
                crate::backend::wgpu::dispatch::fft::run(
                    emission,
                    &mut submissions,
                    &pipeline,
                    &prepared.bind_group,
                    1,
                    "first-scoped-update",
                )
            })
            .unwrap();
        submissions.commit();
        receipt
    };
    #[cfg(target_arch = "wasm32")]
    let outstanding = {
        use std::{future::Future, task::Context};
        let mut future =
            Box::pin(provider.fft_dispatch_with_debug(true, "first-scoped-update", prepare));
        let waker = futures::task::noop_waker();
        assert!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending(),
            "actual browser scope promise must suspend before event-loop delivery"
        );
        future
    };

    // The first receipt/future is deliberately still alive. Both synchronous
    // access and a distinct production submission must succeed on the SAME access.
    provider
        .access
        .with_shared(|emission| {
            emission.ensure_device(&provider.access).unwrap();
        })
        .unwrap();
    provider
        .fft_dispatch_with_debug(false, "second-shared-update", prepare)
        .await
        .unwrap();
    let actual = provider.download_numeric_exec(&value).await.unwrap();
    assert_eq!(
        actual.data,
        HostNumericDataOwned::F32(vec![15.0]),
        "ordered submissions must execute 3 -> 7 -> 15 before the first receipt is consumed"
    );
    value.allocation_lease().validate_live().unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    outstanding.resolve().await.unwrap().unwrap();
    #[cfg(target_arch = "wasm32")]
    drop(outstanding); // cancellation after independent work also leaves no scope

    provider
        .fft_dispatch_with_debug(true, "fresh-scope-after-outstanding", prepare)
        .await
        .unwrap();
    assert_eq!(
        provider.download_numeric_exec(&value).await.unwrap().data,
        HostNumericDataOwned::F32(vec![31.0])
    );

    #[cfg(target_arch = "wasm32")]
    browser_callback_retains_final_tensor(&provider, &layout, &pipeline).await;
}

#[cfg(target_arch = "wasm32")]
async fn browser_callback_retains_final_tensor(
    provider: &WgpuProvider,
    layout: &wgpu::BindGroupLayout,
    pipeline: &wgpu::ComputePipeline,
) {
    use crate::backend::wgpu::dispatch::observation::PendingSubmissions;
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_name = setTimeout)]
        fn yield_callback(callback: &js_sys::Function, milliseconds: i32) -> i32;
    }
    let baseline = provider.allocations.retained_bytes();
    let value = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[3.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let entry = provider.get_entry_raw(&value).unwrap();
    let weak = value.allocation_lease().downgrade();
    let charged = provider.allocations.retained_bytes();
    let resources = FftDispatchResources::new(&[&value], &[&entry.buffer]).unwrap();
    // No recorder must not mean unfunded Browser completion. Refusal occurs
    // before encoding/submission; the original carrier remains usable.
    let mut unfunded = PendingSubmissions::new(provider.access.clone());
    assert!(unfunded
        .retain_submission_resources(resources.into_retention())
        .is_err());
    value.allocation_lease().validate_live().unwrap();
    drop(unfunded);

    let mut pending = PendingSubmissions::new(provider.access.clone())
        .with_profiling(None, provider.observation_budget.clone());
    pending
        .retain_submission_resources(
            FftDispatchResources::new(&[&value], &[&entry.buffer])
                .unwrap()
                .into_retention(),
        )
        .unwrap();
    provider
        .access
        .with_shared(|emission| {
            let bind = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("browser-fft-retained-final-handle"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_entire_binding(),
                    }],
                });
            crate::backend::wgpu::dispatch::fft::run(
                emission,
                &mut pending,
                pipeline,
                &bind,
                1,
                "browser-fft-retained-final-handle",
            )
        })
        .unwrap()
        .unwrap();
    pending.commit();
    drop(entry);
    drop(value);
    // No event-loop yield has delivered the real private map callback yet.
    assert!(weak.acquire().is_some());
    assert_eq!(provider.allocations.retained_bytes(), charged);
    let deadline = js_sys::Date::now() + 10_000.0;
    while weak.acquire().is_some() {
        assert!(
            js_sys::Date::now() < deadline,
            "real private fence must retire the lease"
        );
        provider.access.poll(wgpu::Maintain::Poll);
        let turn = js_sys::Promise::new(&mut |resolve, _| {
            yield_callback(&resolve, 0);
        });
        wasm_bindgen_futures::JsFuture::from(turn).await.unwrap();
    }
    assert_eq!(provider.allocations.retained_bytes(), baseline);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires actual WGPU adapter; run this filter alone in a dedicated test process with --ignored"]
fn native_submission_progresses_with_detached_receipt_outstanding() {
    let (finished, completion) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        futures::executor::block_on(submit_while_receipt_is_outstanding());
        finished.send(()).unwrap();
    });
    // This watchdog detects failure; it cannot cancel a driver call or reclaim
    // a blocked worker. Run only this filtered test in its dedicated process,
    // so process termination supplies containment after a watchdog failure.
    completion
        .recv_timeout(std::time::Duration::from_secs(45))
        .expect("real-device witness must complete; a gate deadlock is a failure");
    worker.join().unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_submission_progresses_with_scope_promise_pending() {
    use futures::future::{select, Either};
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_name = setTimeout)]
        fn deadline(callback: &js_sys::Function, milliseconds: i32) -> i32;
        #[wasm_bindgen(js_name = clearTimeout)]
        fn clear_deadline(id: i32);
    }
    let mut timer = 0;
    let timeout = js_sys::Promise::new(&mut |resolve, _| {
        timer = deadline(&resolve, 45_000);
    });
    let result = select(
        Box::pin(submit_while_receipt_is_outstanding()),
        Box::pin(wasm_bindgen_futures::JsFuture::from(timeout)),
    )
    .await;
    clear_deadline(timer);
    match result {
        Either::Left(((), remaining_timeout)) => drop(remaining_timeout),
        Either::Right((_, unfinished_witness)) => {
            // A WASM panic may abort rather than unwind. Explicitly cancel the
            // witness first so its pending receipt/readback owners are dropped.
            drop(unfinished_witness);
            panic!("real browser submissions/readback must finish before deadline");
        }
    }
}
