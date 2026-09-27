use super::*;
use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
use std::{
    future::Future,
    task::{Context, Poll},
};
use wasm_bindgen_test::wasm_bindgen_test;
use wgpu::util::DeviceExt;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn pending_owned_readback_cancellation_unmaps_same_device_buffer() {
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("this browser gate requires an actual WebGPU adapter/device");
    let access = provider.access.clone();
    let expected = [1.25f32, -3.5, 0.0, 9.0];
    let buffer = access
        .with_shared(|emission| {
            emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("provider-browser-cancel-readback"),
                    contents: bytemuck::cast_slice(&expected),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                })
        })
        .expect("same-device staging creation");
    // WGPU 24's buffer clone retains the same native backing; keep one test
    // witness so successful remapping can prove that cancellation unmapped it.
    drop(read_with(
        access.clone(),
        buffer.clone(),
        "unpolled",
        |_| -> Result<()> { panic!("an unpolled decoder must not execute") },
    ));
    let mut pending = Box::pin(read_with(
        access.clone(),
        buffer.clone(),
        "pending",
        |_| -> Result<()> { panic!("cancelled decoder must not execute") },
    ));
    let waker = futures::task::noop_waker();
    // The JS promise cannot deliver its callback before this event-loop turn
    // yields. Poll and cancellation both occur outside any emission permit.
    assert!(matches!(
        pending.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    drop(pending);
    let actual = read_with(access, buffer, "remap after cancellation", |bytes| {
        Ok(bytemuck::cast_slice::<u8, f32>(bytes).to_vec())
    })
    .await
    .expect("cancelled production mapping must permit remapping");
    assert_eq!(actual, expected);
}
