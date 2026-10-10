use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType, WgpuAccessError,
    WgpuBufferOwner, WgpuBufferRef,
};
use std::future::Future;
use std::task::{Context, Poll};
use wasm_bindgen_test::wasm_bindgen_test;
use wgpu::util::DeviceExt;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn exported_buffer_keeps_same_browser_backing_and_exact_live_device_authority() {
    async fn access() -> WgpuDeviceAccess {
        let instance = crate::wgpu_compat::instance_new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("browser WebGPU adapter required");
        let (device, queue) = adapter
            .request_device(&crate::wgpu_compat::default_device_descriptor(), None)
            .await
            .expect("browser WebGPU device required");
        WgpuDeviceAccess::new(device, queue)
    }

    let selected = access().await;
    let other = access().await;
    let allocation = TestAllocationOwner::default()
        .allocate_zeroed(vec![0, 0], NumericElementType::F32, GpuTensorStorage::Real)
        .expect("typed zero-length owned allocation");
    let backing = selected
        .with_shared(|permit| {
            WgpuBufferOwner::new(permit.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("plot-browser-export-owner"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        })
        .expect("same-device buffer creation");
    let export = WgpuBufferRef::from_owned_buffer(
        backing.clone(),
        allocation.handle.clone(),
        selected.clone(),
    )
    .expect("authenticated export");
    let input = crate::gpu::GpuInputBuffer::Exported(export);
    let renderer_input: crate::gpu::GpuInputBuffer = backing.clone().into();
    selected
        .with_shared(|permit| {
            assert!(std::ptr::eq(
                input.buffer(permit).unwrap(),
                backing.as_ref()
            ));
            assert!(std::ptr::eq(
                renderer_input.buffer(permit).unwrap(),
                backing.as_ref()
            ));
        })
        .unwrap();
    other
        .with_shared(|permit| {
            assert!(matches!(
                input.buffer(permit),
                Err(WgpuAccessError::WrongDevice)
            ));
        })
        .unwrap();
    allocation.probe.mark_unavailable();
    selected
        .with_shared(|permit| {
            assert!(matches!(
                input.buffer(permit),
                Err(WgpuAccessError::Allocation(_))
            ));
        })
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[wasm_bindgen_test(async)]
async fn pending_production_readback_cancellation_unmaps_for_reuse() {
    let instance = crate::wgpu_compat::instance_new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("this browser hardware gate requires an actual WebGPU adapter");
    let (device, queue) = adapter
        .request_device(&crate::wgpu_compat::default_device_descriptor(), None)
        .await
        .expect("this browser hardware gate requires an actual WebGPU device");
    let access = WgpuDeviceAccess::new(device, queue);
    let expected = [1.25f32, -3.5, 0.0, 9.0];
    let buffer = access
        .with_shared(|emission| {
            let buffer = emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("plot-browser-cancellation-readback"),
                    contents: bytemuck::cast_slice(&expected),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                });
            emission.queue().submit(std::iter::empty());
            buffer
        })
        .expect("create same-device staging buffer");

    drop(readback_f32_buffer(&access, &buffer, expected.len()));
    let mut readback = Box::pin(readback_f32_buffer(&access, &buffer, expected.len()));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    // No JS event-loop yield occurs between mapAsync registration and this
    // poll returning. Its promise cannot deliver the map callback in this turn.
    match readback.as_mut().poll(&mut context) {
        Poll::Pending => {}
        Poll::Ready(result) => panic!("expected registered pending WebGPU mapping, got {result:?}"),
    }
    // Poll and cancellation are both outside any emission permit.
    drop(readback);
    let actual = readback_f32_buffer(&access, &buffer, expected.len())
        .await
        .expect("cancelled production mapping must be reusable");
    assert_eq!(actual, expected);
}
