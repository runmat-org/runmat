//! Existing WGPU allocation table and physical retirement dependencies.
//! This preparation slice deliberately preserves explicit-free semantics.
use super::backend_types::BufferEntry;
use crate::backend::wgpu::cache::bind_group::BindGroupCache;
use crate::backend::wgpu::residency::{BufferResidency, BufferUsageClass};
use crate::backend::wgpu::resources::KernelResourceRegistry;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind;
use runmat_accelerate_api::{GpuTensorDescriptor, GpuTensorHandle};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

#[path = "allocation_state/lifecycle.rs"]
mod lifecycle;
#[path = "allocation_state/publication.rs"]
mod publication;
#[path = "allocation_state/retirement.rs"]
mod retirement;
use lifecycle::{AllocationStateError, Disposition, Failure, Phase, Record, Stage};

pub(super) struct AllocationState {
    device_id: u32,
    buffers: Mutex<HashMap<u64, Record>>,
    next_id: AtomicU64,
    lost: AtomicBool,
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
            lost: AtomicBool::new(false),
            pool: BufferResidency::new(pool_limit),
            max_poolable_bytes,
            bind_groups,
            kernel_resources,
        }
    }

    fn lock_records(&self) -> Result<MutexGuard<'_, HashMap<u64, Record>>> {
        self.buffers
            .lock()
            .map_err(|_| AllocationStateError::Poisoned.into())
    }

    pub(super) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn mark_device_lost(&self) {
        self.lost.store(true, Ordering::Release);
    }

    pub(super) fn live_entry(&self, id: u64) -> Result<BufferEntry> {
        if self.is_lost() {
            return Err(AllocationStateError::DeviceLost.into());
        }
        let records = self.lock_records()?;
        let record = records.get(&id).ok_or(AllocationStateError::Missing)?;
        if record.phase != Phase::Live {
            return Err(AllocationStateError::NotLive.into());
        }
        record
            .entry
            .clone()
            .ok_or_else(|| AllocationStateError::NotLive.into())
    }

    pub(super) fn retained_bytes(&self) -> u64 {
        self.lock_records()
            .map(|records| {
                records.values().fold(0_u64, |sum, record| {
                    sum.saturating_add(record.retained_bytes)
                })
            })
            .unwrap_or(u64::MAX)
    }

    pub(super) fn mark_usage(&self, id: u64, usage: BufferUsageClass) -> Result<()> {
        let mut records = self.lock_records()?;
        let record = records.get_mut(&id).ok_or(AllocationStateError::Missing)?;
        if record.phase != Phase::Live {
            return Err(AllocationStateError::NotLive.into());
        }
        record
            .entry
            .as_mut()
            .ok_or(AllocationStateError::NotLive)?
            .usage = usage;
        Ok(())
    }

    pub(super) fn record_submission(&self, id: u64, submission: u32) -> Result<()> {
        let mut records = self.lock_records()?;
        let record = records.get_mut(&id).ok_or(AllocationStateError::Missing)?;
        if record.phase != Phase::Live {
            return Err(AllocationStateError::NotLive.into());
        }
        record
            .entry
            .as_mut()
            .ok_or(AllocationStateError::NotLive)?
            .last_submission_id = Some(submission);
        Ok(())
    }

    pub(super) fn reshape(
        &self,
        id: u64,
        shape: Vec<usize>,
        expected_len: usize,
        storage: runmat_accelerate_api::GpuTensorStorage,
    ) -> Result<()> {
        let old_shape = {
            let mut records = self.lock_records()?;
            let record = records.get_mut(&id).ok_or(AllocationStateError::Missing)?;
            if record.phase != Phase::Live {
                return Err(AllocationStateError::NotLive.into());
            }
            let entry = record.entry.as_mut().ok_or(AllocationStateError::NotLive)?;
            anyhow::ensure!(
                entry.len == expected_len,
                "reshape: product of dimensions must equal original tensor length"
            );
            entry.storage = storage;
            std::mem::replace(&mut entry.shape, shape)
        };
        drop(old_shape);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn record_count(&self) -> Result<usize> {
        Ok(self.lock_records()?.len())
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
}

#[cfg(test)]
#[path = "allocation_state/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "allocation_state/transition_tests.rs"]
mod transition_tests;
