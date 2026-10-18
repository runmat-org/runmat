//! Exact selected numeric-readback staging authority. Diagnostic handle IDs
//! alone never mint this lease; only a provider can issue its physical owner.
use crate::{handle_identity, AcceleratorThreadBound, GpuHandleIdentity, GpuTensorHandle};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::{any::Any, sync::Arc};

/// A provider-owned, shareable physical staging lease. Clone creates a second
/// small control handle for the callback; both retain the same physical owner.
pub trait NumericDownloadStageLease: AcceleratorThreadBound {
    fn clone_for_callback(&self) -> Box<dyn NumericDownloadStageLease>;
    fn as_any(&self) -> &dyn Any;
}

/// The lease field precedes its funding claim so the final control handle is
/// deallocated before the claim can refund. The provider callback retains a
/// second copy after the awaiting future is cancelled.
pub struct PreparedNumericDownloadStage {
    lease: Box<dyn NumericDownloadStageLease>,
    claim: Arc<dyn RetainedHostAllocationClaim>,
    device_id: u32,
    source: GpuHandleIdentity,
    source_requested_bytes: u64,
    provider_scratch_bytes: u64,
    queue_slots: u32,
    callback_issued: bool,
}

impl PreparedNumericDownloadStage {
    pub fn issued(
        lease: Box<dyn NumericDownloadStageLease>,
        claim: Arc<dyn RetainedHostAllocationClaim>,
        source: &GpuTensorHandle,
        provider_scratch_bytes: u64,
        queue_slots: u32,
    ) -> Self {
        Self {
            lease,
            claim,
            device_id: source.device_id(),
            source: handle_identity(source),
            source_requested_bytes: source.physical().requested_bytes(),
            provider_scratch_bytes,
            queue_slots,
            callback_issued: false,
        }
    }

    pub fn matches_source(&self, source: &GpuTensorHandle) -> bool {
        !self.callback_issued
            && self.device_id == source.device_id()
            && self.source == handle_identity(source)
            && self.source_requested_bytes == source.physical().requested_bytes()
            && source.allocation_lease().validate_live().is_ok()
    }

    pub fn lease_as_any(&self) -> &dyn Any {
        self.lease.as_any()
    }

    pub fn provider_scratch_bytes(&self) -> u64 {
        self.provider_scratch_bytes
    }

    pub fn queue_slots(&self) -> u32 {
        self.queue_slots
    }

    pub fn completion_claim(&self) -> Arc<dyn RetainedHostAllocationClaim> {
        self.claim.clone()
    }

    /// At most one callback control may be created from one quoted stage.
    pub fn take_callback(&mut self) -> Option<Self> {
        if self.callback_issued {
            return None;
        }
        self.callback_issued = true;
        Some(Self {
            lease: self.lease.clone_for_callback(),
            claim: self.claim.clone(),
            device_id: self.device_id,
            source: self.source,
            source_requested_bytes: self.source_requested_bytes,
            provider_scratch_bytes: self.provider_scratch_bytes,
            queue_slots: self.queue_slots,
            callback_issued: true,
        })
    }
}
