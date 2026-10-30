//! Provider-owned staging admission for exact selected numeric downloads.
//! WGPU exposes no whole-device scratch/queue quota; this owner checks the
//! actual per-buffer limit and tracks request-local in-flight physical use.
use super::super::owner_shared::OwnerShared;
use super::AllocationState;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::{DeviceAllocationScope, WgpuBufferOwner};
use runmat_accelerate_api::{
    GpuTensorHandle, NumericDownloadStageLease, PreparedNumericDownloadStage,
};
use runmat_execution::resource::{ExactShared, RetainedHostAllocationClaim};
use std::{
    any::Any,
    sync::{
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

pub(in super::super) enum NumericStageBuffer {
    Ordinary(WgpuBufferOwner),
    Readback(runmat_accelerate_api::WgpuNumericReadbackBuffer),
}
impl NumericStageBuffer {
    fn creation_scope(&self) -> Option<DeviceAllocationScope> {
        match self {
            Self::Ordinary(buffer) => buffer.creation_scope(),
            Self::Readback(buffer) => buffer.creation_scope(),
        }
    }
}
impl std::ops::Deref for NumericStageBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ordinary(buffer) => buffer,
            Self::Readback(buffer) => std::borrow::Borrow::borrow(buffer),
        }
    }
}
impl From<WgpuBufferOwner> for NumericStageBuffer {
    fn from(buffer: WgpuBufferOwner) -> Self {
        Self::Ordinary(buffer)
    }
}

#[derive(Default)]
pub(super) struct LiveNumericDownloadStaging {
    bytes: u64,
    queue_slots: u64,
    /// Facts only: this list cannot keep a physical staging owner alive. Each
    /// node is exactly funded by that stage's existing completion claim.
    head: Option<ExactShared<StageFact>>,
}

struct StageFact {
    scope: Option<DeviceAllocationScope>,
    bytes: u64,
    slots: u32,
    handles: AtomicUsize,
    map: AtomicU8, // 0 prepared, 1 pending, 2 callback terminal
    next: Mutex<Option<ExactShared<StageFact>>>,
    previous: Mutex<Option<ExactShared<StageFact>>>,
    _claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

struct NumericDownloadStageOwner {
    allocation: OwnerShared<AllocationState>,
    staging: Option<NumericStageBuffer>,
    bytes: u64,
    queue_slots: u32,
    fact: ExactShared<StageFact>,
    registered: AtomicBool,
    _resource_ticket: Option<runmat_accelerate_api::observation::ProviderResourceTicket>,
    tail_permit: Option<runmat_accelerate_api::observation::WgpuTailPermit>,
}

impl Drop for NumericDownloadStageOwner {
    fn drop(&mut self) {
        drop(self.staging.take());
        if !self.registered.load(Ordering::Acquire) {
            return;
        }
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
        // Facts point only to facts, never back to a physical StageOwner.
        // Last physical owner drop always unlinks both links in O(1), breaking
        // the temporary fact-only bidirectional chain before refunding claims.
        let previous = self
            .fact
            .get()
            .previous
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        let next = self
            .fact
            .get()
            .next
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        if let Some(previous) = &previous {
            *previous
                .get()
                .next
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = next.clone();
        } else {
            live.head = next.clone();
        }
        if let Some(next) = &next {
            *next
                .get()
                .previous
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = previous;
        }
    }
}

pub(crate) struct NumericDownloadStageHandle {
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
        self.owner.get().staging.as_deref()
    }

    pub(in super::super) fn readback_buffer(
        &self,
    ) -> Option<&runmat_accelerate_api::WgpuNumericReadbackBuffer> {
        match self.owner.get().staging.as_ref()? {
            NumericStageBuffer::Readback(buffer) => Some(buffer),
            NumericStageBuffer::Ordinary(_) => None,
        }
    }
    pub(in super::super) fn validate_scope(&self) -> Result<()> {
        runmat_accelerate_api::validate_device_allocation_scope(self.owner.get().fact.get().scope)?;
        Ok(())
    }
    pub(in super::super) fn callback_owner(&self) -> Self {
        self.owner
            .get()
            .fact
            .get()
            .handles
            .fetch_add(1, Ordering::Relaxed);
        Self {
            owner: self.owner.clone(),
        }
    }
    pub(in super::super) fn map_pending(&self) {
        self.owner.get().fact.get().map.store(1, Ordering::Release);
    }
    pub(in super::super) fn map_complete(&self) {
        self.owner.get().fact.get().map.store(2, Ordering::Release);
    }
    pub(in super::super) fn map_failed(&self) {
        self.map_complete();
        if let Some(permit) = &self.owner.get().tail_permit {
            permit.poison_budget();
        }
        if let Some(ticket) = &self.owner.get()._resource_ticket {
            ticket.fail_owner();
        }
    }
}
impl Drop for NumericDownloadStageHandle {
    fn drop(&mut self) {
        self.owner
            .get()
            .fact
            .get()
            .handles
            .fetch_sub(1, Ordering::AcqRel);
    }
}

impl NumericDownloadStageLease for NumericDownloadStageHandle {
    fn clone_for_callback(&self) -> Box<dyn NumericDownloadStageLease> {
        Box::new(self.callback_owner())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One exact shared owner plus two boxed trait handles: the awaiting future
/// and the SDK map callback. The Arc claim itself is admitted by the caller.
pub(in super::super) fn numeric_download_stage_host_bytes() -> Option<u64> {
    mapped_stage_host_bytes()?
        .checked_add(u64::try_from(2 * std::mem::size_of::<NumericDownloadStageHandle>()).ok()?)
}
pub(in super::super) fn mapped_stage_host_bytes() -> Option<u64> {
    ExactShared::<NumericDownloadStageOwner>::requested_bytes()
        .checked_add(ExactShared::<StageFact>::requested_bytes())?
        .try_into()
        .ok()
}

impl AllocationState {
    pub(super) fn has_numeric_download_scope(&self, scope: DeviceAllocationScope) -> Result<bool> {
        let live = self
            .numeric_download_staging
            .lock()
            .map_err(|_| anyhow!("numeric staging owner poisoned"))?;
        let mut cursor = live.head.clone();
        while let Some(node) = cursor {
            let fact = node.get();
            if (fact.bytes != 0 || fact.slots != 0)
                && fact.scope.is_some_and(|created| scope.covers(created))
            {
                return Ok(true);
            }
            cursor = fact
                .next
                .lock()
                .map_err(|_| anyhow!("numeric staging facts poisoned"))?
                .clone();
        }
        Ok(false)
    }
    pub(super) fn ensure_scope_staging_retired(&self, scope: DeviceAllocationScope) -> Result<()> {
        let live = self
            .numeric_download_staging
            .lock()
            .map_err(|_| anyhow!("numeric staging owner poisoned"))?;
        let mut cursor = live.head.clone();
        let (mut stages, mut bytes, mut slots, mut handles, mut callbacks) =
            (0u64, 0u64, 0u64, 0usize, 0u64);
        while let Some(node) = cursor {
            let fact = node.get();
            if (fact.bytes != 0 || fact.slots != 0)
                && fact.scope.is_some_and(|created| scope.covers(created))
            {
                stages = stages
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("numeric staging count overflow"))?;
                bytes = bytes
                    .checked_add(fact.bytes)
                    .ok_or_else(|| anyhow!("numeric staging byte count overflow"))?;
                slots = slots
                    .checked_add(u64::from(fact.slots))
                    .ok_or_else(|| anyhow!("numeric staging slot count overflow"))?;
                handles = handles
                    .checked_add(fact.handles.load(Ordering::Acquire))
                    .ok_or_else(|| anyhow!("numeric staging handle count overflow"))?;
                callbacks = callbacks
                    .checked_add(u64::from(fact.map.load(Ordering::Acquire) == 1))
                    .ok_or_else(|| anyhow!("numeric staging callback count overflow"))?;
            }
            cursor = fact
                .next
                .lock()
                .map_err(|_| anyhow!("numeric stage facts poisoned"))?
                .clone();
        }
        anyhow::ensure!(stages == 0,
            "WGPU reset retains numeric staging: session={} through_epoch={} stages={stages} bytes={bytes} queue_slots={slots} owner_handles={handles} pending_map_callbacks={callbacks}; release/complete these original owners before retrying reset",
            scope.session(), scope.epoch());
        Ok(())
    }

    pub(in super::super) fn own_numeric_download_stage(
        owner: &OwnerShared<Self>,
        staging: Option<NumericStageBuffer>,
        scratch_bytes: u64,
        queue_slots: u32,
        claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
        resource_ticket: Option<runmat_accelerate_api::observation::ProviderResourceTicket>,
        tail_permit: Option<runmat_accelerate_api::observation::WgpuTailPermit>,
    ) -> Result<NumericDownloadStageHandle> {
        let scope = staging.as_ref().map_or_else(
            runmat_accelerate_api::current_device_allocation_scope,
            NumericStageBuffer::creation_scope,
        );
        runmat_accelerate_api::validate_device_allocation_scope(scope)?;
        if let Some(staging) = &staging {
            runmat_accelerate_api::validate_device_allocation_scope(staging.creation_scope())?;
        }
        let fact = ExactShared::try_new(StageFact {
            scope,
            bytes: scratch_bytes,
            slots: queue_slots,
            handles: AtomicUsize::new(1),
            map: AtomicU8::new(0),
            next: Mutex::new(None),
            previous: Mutex::new(None),
            _claim: claim,
        })
        .map_err(|_| anyhow!("numeric staging fact allocation failed"))?;
        let stage = ExactShared::try_new(NumericDownloadStageOwner {
            allocation: owner.clone(),
            staging,
            bytes: scratch_bytes,
            queue_slots,
            fact: fact.clone(),
            registered: AtomicBool::new(false),
            _resource_ticket: resource_ticket,
            tail_permit,
        })
        .map_err(|_| anyhow!("numeric staging control allocation failed"))?;
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
            let mut next = fact
                .get()
                .next
                .lock()
                .map_err(|_| anyhow!("numeric staging facts poisoned"))?;
            let head = live.head.clone();
            let mut previous = head
                .as_ref()
                .map(|head| {
                    head.get()
                        .previous
                        .lock()
                        .map_err(|_| anyhow!("numeric staging facts poisoned"))
                })
                .transpose()?;
            // Every fallible lock has succeeded before publishing any links.
            if let Some(previous) = &mut previous {
                **previous = Some(fact.clone());
            }
            *next = live.head.take();
            live.head = Some(fact.clone());
            live.bytes = bytes;
            live.queue_slots = slots;
            stage.get().registered.store(true, Ordering::Release);
        }
        Ok(NumericDownloadStageHandle { owner: stage })
    }
    pub(in super::super) fn prepare_numeric_download_stage(
        owner: &OwnerShared<Self>,
        source: &GpuTensorHandle,
        scratch_bytes: u64,
        queue_slots: u32,
        staging: Option<NumericStageBuffer>,
        claim: Arc<dyn RetainedHostAllocationClaim>,
    ) -> Result<PreparedNumericDownloadStage> {
        if owner.is_lost() || source.allocation_lease().validate_live().is_err() {
            return Err(anyhow!("numeric download source is no longer live"));
        }
        if staging.as_ref().map(|buffer| buffer.size()).unwrap_or(0) > scratch_bytes
            || (staging.is_some() != (queue_slots != 0))
        {
            return Err(anyhow!("numeric download physical staging plan changed"));
        }
        let stage = Self::own_numeric_download_stage(
            owner,
            staging,
            scratch_bytes,
            queue_slots,
            Some(claim.clone()),
            None,
            None,
        )?;
        Ok(PreparedNumericDownloadStage::issued(
            Box::new(stage),
            claim,
            source,
            scratch_bytes,
            queue_slots,
        ))
    }
}
