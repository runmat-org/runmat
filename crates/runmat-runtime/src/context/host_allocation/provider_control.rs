//! Explicit original-Runtime authority for provider-owned first-party controls.
//! This stack-borrowed adapter allocates nothing until a checked domain asks
//! for one exact record. Claims follow the actual call/retirement/cache owner.
use super::{
    checkpoint::HostAllocationCheckpoint, receipt::provider_metadata_error, HostAllocationError,
    HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use runmat_execution::resource::{
    ProviderHostMetadataAdmission, ProviderHostMetadataError, RetainedHostAllocationClaim,
    RetainedHostViewGrant,
};
use std::sync::Arc;

pub struct ProviderHostControlAdmission<'runtime> {
    runtime: &'runtime RuntimeContext,
    checkpoint: HostAllocationCheckpoint,
}

impl std::fmt::Debug for ProviderHostControlAdmission<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProviderHostControlAdmission(..)")
    }
}

impl<'runtime> ProviderHostControlAdmission<'runtime> {
    /// None is the existing unbounded policy. A finite context without its
    /// actual managed owner is an error, never an unbounded substitute.
    pub fn capture(
        runtime: Option<&'runtime RuntimeContext>,
    ) -> Result<Option<Self>, HostAllocationError> {
        let Some(checkpoint) = HostAllocationReceipt::capture_owner(runtime)? else {
            return Ok(None);
        };
        let runtime = runtime.expect("a captured managed owner has its original Runtime");
        checkpoint.check(runtime)?;
        Ok(Some(Self {
            runtime,
            checkpoint,
        }))
    }

    fn claim(
        &self,
        payload_bytes: u64,
    ) -> Result<Arc<dyn RetainedHostAllocationClaim>, HostAllocationError> {
        self.checkpoint.check(self.runtime)?;
        let retained = HostAllocationPlan::retained_metadata_bytes(payload_bytes)?;
        let receipt = HostAllocationReceipt::admit_captured(
            self.checkpoint.clone(),
            HostAllocationPlan::new(0, retained)?,
        )?;
        let claim = receipt.claim_retained_metadata(self.runtime, payload_bytes)?;
        self.checkpoint.check(self.runtime)?;
        Ok(claim)
    }
}

impl ProviderHostMetadataAdmission for ProviderHostControlAdmission<'_> {
    fn claim_record(
        &self,
        payload_bytes: u64,
    ) -> Result<Arc<dyn RetainedHostAllocationClaim>, ProviderHostMetadataError> {
        self.claim(payload_bytes).map_err(provider_metadata_error)
    }

    // Control admission is not a tensor view or provider registration grant.
    fn claim_view(
        &self,
        _: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, ProviderHostMetadataError> {
        Err(ProviderHostMetadataError::OwnerFailure)
    }
}
