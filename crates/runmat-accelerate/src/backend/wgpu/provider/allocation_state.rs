//! Existing WGPU allocation table and physical retirement dependencies.
//! Physical records hold weak leases; tensor views retain the captured owner.
use super::backend_types::BufferEntry;
use super::owner_shared::OwnerWeak;
use crate::backend::wgpu::cache::bind_group::BindGroupCache;
use crate::backend::wgpu::residency::{BufferResidency, BufferUsageClass};
use crate::backend::wgpu::resources::KernelResourceRegistry;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind;
use runmat_accelerate_api::GpuTensorHandle;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex, MutexGuard};

#[path = "allocation_state/lifecycle.rs"]
mod lifecycle;
#[path = "allocation_state/publication.rs"]
mod publication;
#[path = "allocation_state/retirement.rs"]
mod retirement;
#[path = "allocation_state/record_table.rs"]
mod record_table;
#[path = "allocation_state/managed_admission.rs"]
mod managed_admission;
#[path = "allocation_state/numeric_download_stage.rs"]
mod numeric_download_stage;
pub(super) use numeric_download_stage::{
    mapped_stage_host_bytes, numeric_download_stage_host_bytes, NumericDownloadStageHandle,
    NumericStageBuffer,
};
#[path = "allocation_state/owner.rs"]
mod owner;
use lifecycle::{AllocationStateError, Disposition, Failure, Phase, Record, Stage};
use record_table::RecordTable;

pub(super) struct AllocationState {
    device_id: u32,
    buffers: Mutex<RecordTable>,
    numeric_download_staging: Mutex<numeric_download_stage::LiveNumericDownloadStaging>,
    next_id: AtomicU64,
    lost: AtomicBool,
    pool: BufferResidency,
    max_poolable_bytes: u64,
    bind_groups: Arc<BindGroupCache>,
    // Provenance contains tensor handles. It must not form a strong path from
    // allocation state back through future handle leases to this same state.
    kernel_resources: OwnerWeak<KernelResourceRegistry>,
    reduction_reuse: OnceLock<OwnerWeak<Mutex<super::moments_cache::ReductionReuseCache>>>,
}

impl AllocationState {
    pub(super) fn new(
        device_id: u32,
        pool_limit: usize,
        max_poolable_bytes: u64,
        bind_groups: Arc<BindGroupCache>,
        kernel_resources: OwnerWeak<KernelResourceRegistry>,
    ) -> Self {
        Self {
            device_id,
            buffers: Mutex::new(RecordTable::default()),
            numeric_download_staging: Mutex::new(
                numeric_download_stage::LiveNumericDownloadStaging::default(),
            ),
            next_id: AtomicU64::new(1),
            lost: AtomicBool::new(false),
            pool: BufferResidency::new(pool_limit),
            max_poolable_bytes,
            bind_groups,
            kernel_resources,
            reduction_reuse: OnceLock::new(),
        }
    }

    pub(super) fn set_reduction_reuse(
        &self,
        cache: OwnerWeak<Mutex<super::moments_cache::ReductionReuseCache>>,
    ) -> Result<()> {
        self.reduction_reuse
            .set(cache)
            .map_err(|_| anyhow!("WGPU reduction cache owner already installed"))
    }

    fn lock_records(&self) -> Result<MutexGuard<'_, RecordTable>> {
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

    /// Associate an issued receipt's byte ticket with this exact live tensor
    /// allocation. The existing lease owner, not an ambient callback context,
    /// determines when those requested bytes leave the operation ledger.
    pub(super) fn attach_probe_resource_ticket(
        &self,
        handle: &GpuTensorHandle,
        ticket: runmat_accelerate_api::observation::ProviderResourceTicket,
    ) -> Result<()> {
        let entry = self.entry_for_handle(handle)?;
        anyhow::ensure!(
            ticket.requested_bytes() == entry.buffer.size(),
            "probe resource ticket extent differs from physical allocation"
        );
        let mut records = self.lock_records()?;
        let record = records
            .get_mut(&handle.buffer_id())
            .ok_or(AllocationStateError::Missing)?;
        anyhow::ensure!(
            record.phase == Phase::Live
                && record.resource_ticket.is_none()
                && record.physical == *handle.physical(),
            "probe resource ticket has no unique live allocation"
        );
        record.resource_ticket = Some(ticket);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn record_count(&self) -> Result<usize> {
        Ok(self.lock_records()?.len())
    }

    pub(super) fn acquire(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        usage: BufferUsageClass,
        len: usize,
        element_size: usize,
        label: &str,
    ) -> Result<(runmat_accelerate_api::WgpuBufferOwner, bool)> {
        self.pool.acquire(emission, usage, len, element_size, label)
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

#[cfg(test)]
#[path = "allocation_state/reusable_registration_tests.rs"]
mod reusable_registration_tests;
