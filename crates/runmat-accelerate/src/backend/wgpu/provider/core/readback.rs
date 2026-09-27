//! Staging ownership across asynchronous mapping. No emission permit survives
//! a poll/await; cancellation performs only a fresh restricted cleanup phase.
use anyhow::{anyhow, Result};
use futures::channel::oneshot;
use runmat_accelerate_api::WgpuDeviceAccess;

struct Readback {
    access: WgpuDeviceAccess,
    buffer: Option<wgpu::Buffer>,
    map_registered: bool,
}

impl Readback {
    fn cleanup(&mut self) -> Result<()> {
        if !self.map_registered {
            // Unpolled futures own an ordinary buffer, not an active mapping.
            // Raw Buffer Drop cannot feed the pinned backends' error scopes.
            drop(self.buffer.take());
            return Ok(());
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.access.with_cleanup(|cleanup| {
                if let Some(buffer) = self.buffer.take() {
                    if self.map_registered {
                        cleanup.unmap_buffer(&buffer);
                    }
                    drop(buffer);
                }
            })
        }));
        if !matches!(result, Ok(Ok(()))) {
            if let Some(recorder) =
                runmat_accelerate_api::observation::current_provider_observation()
            {
                recorder.fail(
                    runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                );
            }
            return Err(anyhow!("WGPU staging teardown failed; no retry attempted"));
        }
        Ok(())
    }
}

impl Drop for Readback {
    fn drop(&mut self) {
        if self.buffer.is_some() && self.cleanup().is_err() {
            log::error!("WGPU staging teardown failed after mapping; no retry attempted");
        }
    }
}

pub(super) fn read(
    access: WgpuDeviceAccess,
    buffer: wgpu::Buffer,
    size_bytes: u64,
    context: &str,
) -> impl std::future::Future<Output = Result<Vec<u8>>> + '_ {
    read_with(access, buffer, context, move |data| {
        let size = usize::try_from(size_bytes)
            .map_err(|_| anyhow!("{context}: readback size overflow"))?;
        if data.len() != size {
            return Err(anyhow!("{context}: mapped readback size mismatch"));
        }
        Ok(data.to_vec())
    })
}

pub(super) fn read_with<'a, T: 'a>(
    access: WgpuDeviceAccess,
    buffer: wgpu::Buffer,
    context: &'a str,
    decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
) -> impl std::future::Future<Output = Result<T>> + 'a {
    // Establish ownership before constructing the future. Unpolled teardown
    // needs only ordinary Buffer Drop; active mapping needs explicit unmap.
    let mut staging = Readback {
        access,
        buffer: Some(buffer),
        map_registered: false,
    };
    async move {
        let (send, receive) = oneshot::channel();
        staging.access.with_shared(|_| {
            staging.map_registered = true;
            staging
                .buffer
                .as_ref()
                .expect("readback owns staging")
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = send.send(result);
                });
        })?;
        #[cfg(not(target_arch = "wasm32"))]
        staging.access.poll(wgpu::Maintain::Wait);
        receive
            .await
            .map_err(|_| anyhow!("{context}: map_async callback dropped"))??;
        let output = staging.access.with_shared(|_| {
            let data = staging
                .buffer
                .as_ref()
                .expect("readback owns staging")
                .slice(..)
                .get_mapped_range();
            let output = decode(&data);
            drop(data);
            output
        })??;
        staging.cleanup()?;
        Ok(output)
    }
}
