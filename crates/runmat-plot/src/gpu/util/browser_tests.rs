use super::*;
use std::future::Future;
use std::task::{Context, Poll};
use wasm_bindgen_test::wasm_bindgen_test;
use wgpu::util::DeviceExt;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

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
