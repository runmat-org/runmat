use anyhow::{anyhow, Result};
use bytemuck::{bytes_of, Pod};
#[cfg(not(target_arch = "wasm32"))]
use futures::channel::oneshot;
#[cfg(not(target_arch = "wasm32"))]
use pollster::block_on;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
#[cfg(target_arch = "wasm32")]
use runmat_time::Instant;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use std::time::Duration;
use tracing::info_span;
use wgpu::util::DeviceExt;

use super::backend_shared::NEXT_SUBMISSION_ID;
use super::backend_types::{BufferEntry, WgpuProvider};
use crate::backend::wgpu::residency::BufferUsageClass;
use crate::fusion::active_fusion;

impl WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) async fn map_readback_bytes(
        &self,
        staging: wgpu::Buffer,
        size_bytes: u64,
        context: &str,
    ) -> Result<Vec<u8>> {
        let size_usize = usize::try_from(size_bytes)
            .map_err(|_| anyhow!("{context}: readback size overflow"))?;
        let slice = staging.slice(..);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device.poll(wgpu::Maintain::Wait);
        }
        let map_result = rx
            .await
            .map_err(|_| anyhow!("{context}: map_async callback dropped"))?;
        map_result.map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
        let data = slice.get_mapped_range();
        let mut out = vec![0u8; size_usize];
        out.copy_from_slice(&data);
        drop(data);
        staging.unmap();
        Ok(out)
    }

    pub(super) fn map_readback_bytes_sync(
        &self,
        staging: wgpu::Buffer,
        size_bytes: u64,
        context: &str,
    ) -> Result<Vec<u8>> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (staging, size_bytes);
            Err(anyhow!("{context}: readback requires async path on wasm"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            block_on(self.map_readback_bytes(staging, size_bytes, context))
        }
    }
    pub(super) const BUFFER_RESIDENCY_MAX_PER_KEY: usize = 8;
    pub(super) const IMAGE_NORMALIZE_AUTOTUNE_VERSION: u8 = 1;
    pub(super) const IMAGE_NORMALIZE_STREAM_COLD_CAP: u32 = 8;
    pub(super) const IMAGE_NORMALIZE_TARGET_SAMPLES_PER_LANE: f64 = 256.0;
    pub(super) const IMAGE_NORMALIZE_TARGET_LOOP_ITERS_PER_LANE: f64 = 16.0;

    pub(crate) fn device_ref(&self) -> &wgpu::Device {
        self.device.as_ref()
    }
    pub(crate) fn queue_ref(&self) -> &wgpu::Queue {
        self.queue.as_ref()
    }
    pub(super) fn register_existing_buffer(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
    ) -> GpuTensorHandle {
        self.register_existing_buffer_with_usage(buffer, shape, len, BufferUsageClass::Generic)
    }

    pub(super) fn register_existing_buffer_with_storage(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
        storage: GpuTensorStorage,
    ) -> GpuTensorHandle {
        self.register_existing_buffer_with_usage_and_storage(
            buffer,
            shape,
            len,
            storage,
            BufferUsageClass::Generic,
        )
    }

    pub(super) fn register_existing_buffer_with_usage(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
        usage: BufferUsageClass,
    ) -> GpuTensorHandle {
        self.register_existing_buffer_with_usage_and_storage(
            buffer,
            shape,
            len,
            GpuTensorStorage::Real,
            usage,
        )
    }

    pub(super) fn register_existing_buffer_with_usage_and_storage(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
        storage: GpuTensorStorage,
        usage: BufferUsageClass,
    ) -> GpuTensorHandle {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let entry = BufferEntry {
            buffer,
            len,
            shape: shape.clone(),
            storage: storage.clone(),
            precision: self.precision,
            usage,
            last_submission_id: None,
        };
        self.buffers
            .lock()
            .expect("buffer mutex poisoned")
            .insert(id, entry);
        log::trace!("wgpu register id={} len={} shape={:?}", id, len, &shape);
        let handle = GpuTensorHandle {
            shape,
            device_id: self.runtime_device_id,
            buffer_id: id,
        };
        runmat_accelerate_api::set_handle_logical(&handle, false);
        runmat_accelerate_api::set_handle_storage(&handle, storage);
        runmat_accelerate_api::clear_handle_transpose(&handle);
        handle
    }

    pub(super) fn remember_matmul_sources(
        &self,
        product: &GpuTensorHandle,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) {
        if lhs.device_id != self.runtime_device_id || rhs.device_id != self.runtime_device_id {
            return;
        }
        log::debug!(
            "remember_matmul_sources: product={} lhs={} rhs={} active_fusion={:?}",
            product.buffer_id,
            lhs.buffer_id,
            rhs.buffer_id,
            active_fusion()
        );
        self.kernel_resources
            .remember_matmul_sources(product, lhs, rhs);
    }

    pub(super) fn mark_buffer_usage(&self, handle: &GpuTensorHandle, usage: BufferUsageClass) {
        if let Ok(mut guard) = self.buffers.lock() {
            if let Some(entry) = guard.get_mut(&handle.buffer_id) {
                entry.usage = usage;
            }
        }
    }

    pub(super) fn record_buffer_submission(&self, buffer_id: u64, submission_id: u32) {
        if let Ok(mut guard) = self.buffers.lock() {
            if let Some(entry) = guard.get_mut(&buffer_id) {
                entry.last_submission_id = Some(submission_id);
            }
        }
    }

    pub(super) fn create_storage_buffer_for_usage(
        &self,
        usage: BufferUsageClass,
        len: usize,
        label: &str,
    ) -> (Arc<wgpu::Buffer>, bool) {
        self.buffer_residency
            .acquire(self.device_ref(), usage, len, self.element_size, label)
    }

    pub(super) fn create_storage_buffer(&self, len: usize, label: &str) -> Arc<wgpu::Buffer> {
        self.create_storage_buffer_for_usage(BufferUsageClass::Generic, len, label)
            .0
    }

    pub(super) fn uniform_buffer<T: Pod>(&self, data: &T, label: &str) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub(super) fn prepare_matmul_pipeline(&self) {
        let mut enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-warmup"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-matmul-warmup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul.pipeline);
        }
        self.submit(enc);

        let enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-flush-gap"),
            });
        self.submit(enc);
    }

    pub(super) fn prepare_matmul_vec4_pipeline(&self) {
        let mut enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-vec4-warmup"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-matmul-vec4-warmup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul_vec4.pipeline);
        }
        self.submit(enc);

        let enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-vec4-flush-gap"),
            });
        self.submit(enc);
    }

    pub(super) fn submit(&self, encoder: wgpu::CommandEncoder) -> u32 {
        let submission_id = NEXT_SUBMISSION_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let _span = info_span!(
            "gpu.dispatch",
            label = "runmat-wgpu-submit",
            submission_id = submission_id
        )
        .entered();
        log::trace!("wgpu submit {}: begin", submission_id);
        let work = encoder.finish();
        let submission_index = self.queue.submit(Some(work));
        log::trace!(
            "wgpu submit {}: submitted, polling (index={:?})",
            submission_id,
            submission_index
        );
        // On Web/WASM, blocking `Maintain::Wait` can starve the worker event loop, preventing
        // plot presentation and other cooperative tasks from running. Prefer non-blocking polls.
        #[cfg(target_arch = "wasm32")]
        let poll_start = Instant::now();
        #[cfg(target_arch = "wasm32")]
        self.device.poll(wgpu::Maintain::Poll);
        #[cfg(target_arch = "wasm32")]
        log::trace!("wgpu submit {}: poll complete", submission_id);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.queue.on_submitted_work_done(move || {
                log::trace!(
                    "wgpu submit {}: on_submitted_work_done callback fired",
                    submission_id
                );
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            log::trace!(
                "wgpu submit {}: on_submitted_work_done unavailable on wasm",
                submission_id
            );
        }
        #[cfg(target_arch = "wasm32")]
        {
            let elapsed = poll_start.elapsed();
            if elapsed > Duration::from_millis(250) {
                log::warn!(
                    "wgpu submit {}: device.poll took {}ms on wasm target",
                    submission_id,
                    elapsed.as_millis()
                );
            }
        }
        submission_id
    }
    pub(super) fn get_entry(&self, handle: &GpuTensorHandle) -> Result<BufferEntry> {
        if handle.device_id != self.runtime_device_id {
            return Err(anyhow!(
                "handle device mismatch: expected {}, got {}",
                self.runtime_device_id,
                handle.device_id
            ));
        }
        let guard = self.buffers.lock().expect("buffer mutex poisoned");
        guard
            .get(&handle.buffer_id)
            .map(|entry| BufferEntry {
                buffer: entry.buffer.clone(),
                len: entry.len,
                shape: entry.shape.clone(),
                storage: entry.storage.clone(),
                precision: entry.precision,
                usage: entry.usage,
                last_submission_id: entry.last_submission_id,
            })
            .ok_or_else(|| anyhow!("buffer not found: {}", handle.buffer_id))
    }
}
