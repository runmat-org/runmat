use super::*;

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
