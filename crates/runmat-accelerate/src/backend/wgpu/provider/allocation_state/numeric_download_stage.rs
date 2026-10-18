//! Provider-owned staging admission for exact selected numeric downloads.
//! WGPU exposes no whole-device scratch/queue quota; this owner checks the
//! actual per-buffer limit and tracks request-local in-flight physical use.
use super::super::owner_shared::OwnerShared;
use super::AllocationState;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::{
    GpuTensorHandle, NumericDownloadStageLease, PreparedNumericDownloadStage,
};
use runmat_execution::resource::{ExactShared, RetainedHostAllocationClaim};
use std::{any::Any, sync::Arc};

#[derive(Default)]
pub(super) struct LiveNumericDownloadStaging {
    bytes: u64,
    queue_slots: u64,
}

struct NumericDownloadStageOwner {
    allocation: OwnerShared<AllocationState>,
    staging: Option<wgpu::Buffer>,
    bytes: u64,
    queue_slots: u32,
}

impl Drop for NumericDownloadStageOwner {
    fn drop(&mut self) {
        drop(self.staging.take());
        let mut live = self
            .allocation
            .numeric_download_staging
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        live.bytes = live
            .bytes
            .checked_sub(self.bytes)
            .expect("numeric staging bytes");
        live.queue_slots = live
            .queue_slots
            .checked_sub(u64::from(self.queue_slots))
            .expect("numeric staging queue slots");
    }
}

pub(in super::super) struct NumericDownloadStageHandle {
    owner: ExactShared<NumericDownloadStageOwner>,
}

impl NumericDownloadStageHandle {
    pub(in super::super) fn matches_allocation_owner(
        &self,
        other: &OwnerShared<AllocationState>,
    ) -> bool {
        OwnerShared::ptr_eq(&self.owner.get().allocation, other)
    }

    pub(in super::super) fn staging_buffer(&self) -> Option<&wgpu::Buffer> {
        self.owner.get().staging.as_ref()
    }
}

impl NumericDownloadStageLease for NumericDownloadStageHandle {
    fn clone_for_callback(&self) -> Box<dyn NumericDownloadStageLease> {
        Box::new(Self {
            owner: self.owner.clone(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One exact shared owner plus two boxed trait handles: the awaiting future
/// and the SDK map callback. The Arc claim itself is admitted by the caller.
pub(in super::super) fn numeric_download_stage_host_bytes() -> Option<u64> {
    ExactShared::<NumericDownloadStageOwner>::requested_bytes()
        .checked_add(2 * std::mem::size_of::<NumericDownloadStageHandle>())?
        .try_into()
        .ok()
}

impl AllocationState {
    pub(in super::super) fn prepare_numeric_download_stage(
        owner: &OwnerShared<Self>,
        source: &GpuTensorHandle,
        scratch_bytes: u64,
        queue_slots: u32,
        staging: Option<wgpu::Buffer>,
        claim: Arc<dyn RetainedHostAllocationClaim>,
    ) -> Result<PreparedNumericDownloadStage> {
        if owner.is_lost() || source.allocation_lease().validate_live().is_err() {
            return Err(anyhow!("numeric download source is no longer live"));
        }
        if staging.as_ref().map(wgpu::Buffer::size).unwrap_or(0) > scratch_bytes
            || (staging.is_some() != (queue_slots != 0))
        {
            return Err(anyhow!("numeric download physical staging plan changed"));
        }
        {
            let mut live = owner
                .numeric_download_staging
                .lock()
                .map_err(|_| anyhow!("numeric staging owner poisoned"))?;
            let bytes = live
                .bytes
                .checked_add(scratch_bytes)
                .ok_or_else(|| anyhow!("numeric staging byte overflow"))?;
            let slots = live
                .queue_slots
                .checked_add(u64::from(queue_slots))
                .ok_or_else(|| anyhow!("numeric staging queue overflow"))?;
            live.bytes = bytes;
            live.queue_slots = slots;
        }
        let owner = NumericDownloadStageOwner {
            allocation: owner.clone(),
            staging,
            bytes: scratch_bytes,
            queue_slots,
        };
        let shared = ExactShared::try_new(owner)
            .map_err(|_| anyhow!("numeric staging control allocation failed"))?;
        Ok(PreparedNumericDownloadStage::issued(
            Box::new(NumericDownloadStageHandle { owner: shared }),
            claim,
            source,
            scratch_bytes,
            queue_slots,
        ))
    }
}
