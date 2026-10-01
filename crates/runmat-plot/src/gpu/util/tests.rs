use super::*;
use pollster::FutureExt;
use std::error::Error;
use wgpu::util::DeviceExt;

#[test]
fn cleanup_failure_cannot_report_success_or_erase_the_original_error() {
    let cleanup = PlotReadbackError::UnmapPanicked;
    let primary = PlotReadbackError::Decode("decode");
    assert_eq!(
        combine_cleanup(Ok(42), Err(cleanup.clone())),
        Err(cleanup.clone())
    );
    assert_eq!(
        combine_cleanup::<()>(Err(primary.clone()), Err(cleanup.clone())),
        Err(PlotReadbackError::WithCleanup {
            primary: Box::new(primary),
            cleanup: Box::new(cleanup)
        })
    );
    assert_eq!(combine_cleanup(Ok(42), Ok(())), Ok(42));

    let error = combine_cleanup::<()>(
        Err(PlotReadbackError::Access(
            runmat_accelerate_api::WgpuAccessError::WrongDevice,
        )),
        Err(PlotReadbackError::CleanupAccess(
            runmat_accelerate_api::WgpuAccessError::Poisoned,
        )),
    )
    .unwrap_err();
    assert_eq!(error.to_string(),
        "WGPU permit belongs to another device authority; additionally, GPU readback cleanup could not acquire its device: WGPU emission gate is poisoned");
    assert_eq!(
        error.cleanup_cause(),
        Some(&PlotReadbackError::CleanupAccess(
            runmat_accelerate_api::WgpuAccessError::Poisoned,
        ))
    );
    let primary = error.source().expect("primary source remains inspectable");
    assert!(primary.downcast_ref::<PlotReadbackError>().is_some());
    assert!(primary
        .source()
        .and_then(|source| source.downcast_ref::<runmat_accelerate_api::WgpuAccessError>())
        .is_some());
}

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
            let buffer = emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("plot-readback-owner-test"),
                    contents: bytemuck::cast_slice(&expected),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                });
            // Initial mapped writes may be staged until the next submission.
            // Readback consumes completed producer work; polling alone does
            // not submit the producer's pending upload.
            emission.queue().submit(std::iter::empty());
            buffer
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
    assert_eq!(
        readback_f32(&access, &buffer).block_on().unwrap(),
        expected[0]
    );
    let decode_error = readback_mapped::<()>(&access, &buffer, 16, |_| {
        Err(PlotReadbackError::Decode("decoder rejected payload"))
    })
    .block_on()
    .unwrap_err();
    assert_eq!(
        decode_error,
        PlotReadbackError::Decode("decoder rejected payload")
    );
    #[cfg(panic = "unwind")]
    {
        // Exercise the actual async readback owner, not a standalone Mapping.
        // The BufferView and read permit must unwind before its cleanup runs.
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            readback_mapped::<()>(&access, &buffer, 16, |_| panic!("decoder panic witness"))
                .block_on()
        }));
        assert!(panic.is_err(), "decoder panic must remain observable");
    }
    // Decoder error/panic must end the mapped view and unmap before remapping.
    assert_eq!(
        readback_f32_buffer(&access, &buffer, expected.len())
            .block_on()
            .unwrap(),
        expected
    );
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
