//! Existing WGPU allocation table and physical retirement dependencies.
//! This preparation slice deliberately preserves explicit-free semantics.
use super::backend_types::BufferEntry;
use crate::backend::wgpu::cache::bind_group::BindGroupCache;
use crate::backend::wgpu::residency::{BufferResidency, BufferUsageClass};
use crate::backend::wgpu::resources::KernelResourceRegistry;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::GpuTensorHandle;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LockResult, Mutex, MutexGuard, Weak};

pub(super) struct AllocationState {
    device_id: u32,
    buffers: Mutex<HashMap<u64, BufferEntry>>,
    next_id: AtomicU64,
    pool: BufferResidency,
    max_poolable_bytes: u64,
    bind_groups: Arc<BindGroupCache>,
    // Provenance contains tensor handles. It must not form a strong path from
    // allocation state back through future handle leases to this same state.
    kernel_resources: Weak<KernelResourceRegistry>,
}

impl AllocationState {
    pub(super) fn new(
        device_id: u32,
        pool_limit: usize,
        max_poolable_bytes: u64,
        bind_groups: Arc<BindGroupCache>,
        kernel_resources: Weak<KernelResourceRegistry>,
    ) -> Self {
        Self {
            device_id,
            buffers: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            pool: BufferResidency::new(pool_limit),
            max_poolable_bytes,
            bind_groups,
            kernel_resources,
        }
    }

    /// Internal semantic operations retain their existing scoped map access.
    /// No external resource teardown may execute while this guard is held.
    pub(super) fn lock_entries(&self) -> LockResult<MutexGuard<'_, HashMap<u64, BufferEntry>>> {
        self.buffers.lock()
    }

    pub(super) fn register(&self, entry: BufferEntry) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let replaced = self
            .buffers
            .lock()
            .expect("buffer mutex poisoned")
            .insert(id, entry);
        drop(replaced);
        id
    }

    pub(super) fn acquire(
        &self,
        device: &wgpu::Device,
        usage: BufferUsageClass,
        len: usize,
        element_size: usize,
        label: &str,
    ) -> (Arc<wgpu::Buffer>, bool) {
        self.pool.acquire(device, usage, len, element_size, label)
    }

    pub(super) fn pooled_bytes(&self, element_size: usize) -> u64 {
        self.pool.pooled_bytes(element_size)
    }

    pub(super) fn free(&self, handle: &GpuTensorHandle) -> Result<()> {
        if handle.device_id != self.device_id {
            return Err(anyhow!(
                "free: handle belongs to device {}, but this provider owns device {}",
                handle.device_id,
                self.device_id
            ));
        }
        log::trace!("wgpu free id={}", handle.buffer_id);
        // Checked cache failure must leave the existing allocation and its
        // annotations reachable. This is still the old explicit-free surface:
        // a later owning-lease slice must mark Retiring before invalidation to
        // prevent concurrent new cache admission for this allocation.
        let buffer = self
            .buffers
            .lock()
            .expect("buffer mutex poisoned")
            .get(&handle.buffer_id)
            .map(|entry| entry.buffer.clone());
        if let Some(buffer) = buffer {
            let buffer_ptr = buffer.as_ref() as *const wgpu::Buffer as usize;
            self.bind_groups.invalidate_buffer(buffer_ptr)?;
        }
        let entry = self
            .buffers
            .lock()
            .expect("buffer mutex poisoned")
            .remove(&handle.buffer_id);
        // The allocation-map guard is gone before cache/pool/metadata work.
        // Sibling view IDs and exported buffers retain their existing physical
        // Arc references; neither is an independent physical allocation claim.
        if let Some(entry) = entry {
            if entry.len > 0 {
                let poolable_by_size =
                    self.max_poolable_bytes > 0 && entry.allocated_bytes <= self.max_poolable_bytes;
                let strong_count = Arc::strong_count(&entry.buffer);
                if poolable_by_size && strong_count == 1 {
                    if let Err(rejected) =
                        self.pool
                            .release(entry.usage, entry.allocated_bytes, entry.buffer.clone())
                    {
                        // The pool did not consume this Arc. Releasing it here
                        // preserves ordinary non-pooled cleanup, outside locks.
                        log::trace!("buffer_residency: reuse declined: {:?}", rejected.reason);
                        drop(rejected.buffer);
                    }
                } else {
                    log::trace!(
                        "buffer_residency: not pooling buffer id={} len={} bytes={} strong_count={} poolable_by_size={}",
                        handle.buffer_id, entry.len, entry.allocated_bytes, strong_count, poolable_by_size
                    );
                }
            }
        }
        if let Some(resources) = self.kernel_resources.upgrade() {
            resources.clear_matmul_source(handle.buffer_id);
        }
        runmat_accelerate_api::clear_handle_metadata(handle);
        Ok(())
    }
}

#[cfg(test)]
#[path = "allocation_state/tests.rs"]
mod tests;
