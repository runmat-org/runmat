use super::*;
#[cfg(not(target_arch = "wasm32"))]
use futures::channel::oneshot;

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn production_decoder_failure_and_unwind_unmap_same_staging_buffer() {
    use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
    use wgpu::util::DeviceExt;
    let provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
    let access = provider.access.clone();
    let expected = [1.25f32, -3.5, 9.0, 0.125];
    let buffer = std::sync::Arc::new(
        access
            .with_shared(|emission| {
                let buffer =
                    emission
                        .device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("same-staging-readback-lifecycle"),
                            contents: bytemuck::cast_slice(&expected),
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        });
                // Submit the initialization upload before testing mapping cleanup.
                emission.queue().submit(std::iter::empty());
                buffer
            })
            .expect("same-device staging"),
    );
    let remap = || {
        let actual = futures::executor::block_on(read_with(
            access.clone(),
            buffer.clone(),
            "same staging remap",
            |bytes| Ok(bytemuck::cast_slice::<u8, f32>(bytes).to_vec()),
        ))
        .expect("production cleanup must permit remapping the same staging buffer");
        assert_eq!(actual, expected);
    };
    drop(read_with(
        access.clone(),
        buffer.clone(),
        "unpolled",
        |_| -> Result<()> { panic!("unpolled decoder must not execute") },
    ));
    remap();
    let error = futures::executor::block_on(read_with(
        access.clone(),
        buffer.clone(),
        "decode error",
        |bytes| -> Result<()> {
            assert_eq!(bytemuck::cast_slice::<u8, f32>(bytes), expected);
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "decode sentinel").into())
        },
    ))
    .expect_err("decoder error must survive cleanup");
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::InvalidData
    );
    remap();
    #[cfg(panic = "unwind")]
    {
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            futures::executor::block_on(read_with(
                access.clone(),
                buffer.clone(),
                "decode unwind",
                |bytes| -> Result<()> {
                    assert_eq!(bytemuck::cast_slice::<u8, f32>(bytes), expected);
                    panic!("decoder unwind sentinel")
                },
            ))
        }));
        assert!(unwind.is_err());
        remap();
    }
}

#[test]
fn completion_retains_primary_and_typed_cleanup_failures() {
    let error = complete::<()>(
        Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "decode sentinel").into()),
        Err(CleanupError::Access(WgpuAccessError::Poisoned)),
    )
    .unwrap_err();
    let combined = error.downcast_ref::<CompletionError>().unwrap();
    assert_eq!(
        combined
            .primary
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::BrokenPipe
    );
    assert!(matches!(
        combined.cleanup,
        CleanupError::Access(WgpuAccessError::Poisoned)
    ));
    assert!(combined.source().unwrap().is::<std::io::Error>());
    assert!(error.to_string().contains("decode sentinel"));
    assert!(error.to_string().contains("poisoned"));

    let primary = complete::<()>(
        Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "original").into()),
        Ok(()),
    )
    .unwrap_err();
    assert!(primary.is::<std::io::Error>());
}

#[test]
fn failed_cleanup_drops_unpublished_result_and_stays_typed() {
    let output = std::sync::Arc::new(());
    let weak = std::sync::Arc::downgrade(&output);
    let error = complete(Ok(output), Err(CleanupError::UnmapPanicked)).unwrap_err();
    assert!(weak.upgrade().is_none());
    assert!(matches!(
        error.downcast_ref::<CleanupError>(),
        Some(CleanupError::UnmapPanicked)
    ));
    assert_eq!(complete(Ok(17), Ok(())).unwrap(), 17);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn registered_mapping_cleanup_disarms_before_drop_and_is_idempotent() {
    use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
    let provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
    let access = provider.access.clone();
    let buffer = access
        .with_shared(|emission| {
            emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("registered-readback-cleanup"),
                size: 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .unwrap();
    let mut staging = Readback {
        access: access.clone(),
        buffer: Some(buffer),
        map_registered: false,
    };
    let (send, receive) = oneshot::channel();
    access
        .with_shared(|_| {
            staging.map_registered = true;
            staging.buffer.as_ref().unwrap().slice(..).map_async(
                wgpu::MapMode::Read,
                move |result| {
                    let _ = send.send(result);
                },
            );
        })
        .unwrap();
    // Explicitly abandon registration before progress. This exercises the same
    // cleanup state as Drop; native production readback itself waits in poll.
    staging.cleanup().unwrap();
    assert!(!staging.map_registered);
    assert!(staging.buffer.is_none());
    staging.cleanup().unwrap();
    drop(staging);
    access.poll(wgpu::Maintain::Wait);
    // Delivery may have completed before cancellation on some native drivers.
    // Either mapping outcome is valid; its callback must not own the cleanup.
    let _mapping_result = futures::executor::block_on(receive).expect("map callback delivered");
}
