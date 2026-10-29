//! Actual-device ordering witness, not a throughput measurement. Native scope
//! receipts can be immediately ready; browser promises are explicitly pending.
use super::*;
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
        Ok((&pipeline, bind, 1))
    };

    #[cfg(not(target_arch = "wasm32"))]
    let outstanding = {
        // Retain the actual detached receipt unconsumed. Pinned native WGPU
        // resolves its scope futures immediately: do not manufacture suspension.
        let (_, bind, _) = provider
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
                    &bind,
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
