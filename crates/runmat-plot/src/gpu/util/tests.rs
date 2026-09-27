use super::*;
use pollster::FutureExt;
use wgpu::util::DeviceExt;

fn device_access() -> WgpuDeviceAccess {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .block_on()
        .expect("this hardware gate requires an actual WGPU adapter");
    let (device, queue) = adapter
        .request_device(&crate::wgpu_compat::default_device_descriptor(), None)
        .block_on()
        .expect("this hardware gate requires an actual WGPU device");
    WgpuDeviceAccess::new(device, queue)
}

#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn readback_releases_access_and_mapping_for_reuse() {
    let access = device_access();
    let expected = [1.25f32, -3.5, 0.0, 9.0];
    let buffer = access
        .with_shared(|emission| {
            emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("plot-readback-owner-test"),
                    contents: bytemuck::cast_slice(&expected),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                })
        })
        .expect("create test buffer");

    // An unpolled future does not register a mapping or hold a device permit.
    drop(readback_f32_buffer(&access, &buffer, expected.len()));
    for _ in 0..2 {
        let actual = readback_f32_buffer(&access, &buffer, expected.len())
            .block_on()
            .expect("map, progress outside the permit, and unmap");
        assert_eq!(actual, expected);
    }
    assert!(readback_f32_buffer(&access, &buffer, expected.len() + 1)
        .block_on()
        .is_err());
    assert_eq!(
        readback_f32_buffer(&access, &buffer, expected.len())
            .block_on()
            .expect("extent rejection leaves the mapping reusable"),
        expected
    );
}

#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn pending_mapping_owner_cancellation_unmaps_outside_emission() {
    let access = device_access();
    let buffer = access
        .with_shared(|emission| {
            emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("plot-cancelled-mapping-owner-test"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .expect("create test buffer");
    let (tx, rx) = oneshot::channel();
    let mut mapping = Mapping {
        access: &access,
        buffer: &buffer,
        registered: false,
    };
    access
        .with_shared(|_| {
            mapping.registered = true;
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
        })
        .expect("register pending mapping");
    // This is the same owner used by readback_bytes on cancellation. No
    // emission is active when unmap may synchronously invoke the callback.
    drop(mapping);
    access.poll(wgpu::Maintain::Wait);
    let _completed_mapping_result = rx.block_on().expect("map callback delivered once");
    assert_eq!(
        readback_f32_buffer(&access, &buffer, 4)
            .block_on()
            .expect("cancelled mapping can be remapped"),
        [0.0; 4]
    );
}
