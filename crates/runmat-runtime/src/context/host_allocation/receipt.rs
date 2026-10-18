use std::fmt;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

use runmat_execution::cancellation::CancellationToken;
use runmat_execution::resource::{
    ProviderHostMetadataAdmission, ProviderHostMetadataError, RetainedHostAllocationClaim,
    RetainedHostViewError, RetainedHostViewGrant, RetainedHostViewIssuer,
};
use runmat_value::{
    ExactHostAllocationPermit, ExactHostAllocationReservation, ExactHostNumericAllocationError,
    ExactHostTensorAllocation, NumericDType,
};

use super::{HostAllocationPlan, HostComplexWorkspace, HostNumericWorkspace};
use crate::context::host_effect::{
    HostEffectAllocationError, HostEffectOutputHeadroom, ManagedHostEffectLedger,
    ManagedHostEffectStageReservation,
};
use crate::context::RuntimeContext;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum HostAllocationError {
    Overflow,
    MissingManagedOwner,
    StaleAuthority,
    Cancelled,
    Resource(HostEffectAllocationError),
    Numeric(ExactHostNumericAllocationError),
}

impl HostAllocationError {
    /// A speculative route may decline these resource refusals before any
    /// source effect. Cancellation, revoked authority and invalid geometry
    /// remain errors; they must not become permission to try another route.
    pub fn is_capacity_denial(self) -> bool {
        matches!(
            self,
            Self::Resource(
                HostEffectAllocationError::ExceedsCapacity
                    | HostEffectAllocationError::ExceedsStageEnvelope
            )
        )
    }

    /// Preserve cancellation and revoked service authority as typed runtime
    /// causes at every host-allocation admission boundary.
    pub fn runtime_identifier_suffix(self) -> &'static str {
        match self {
            Self::Cancelled | Self::Resource(HostEffectAllocationError::Cancelled) => "Cancelled",
            Self::StaleAuthority
            | Self::MissingManagedOwner
            | Self::Resource(
                HostEffectAllocationError::Revoked | HostEffectAllocationError::InvalidInvocation,
            ) => "parallel:InfrastructureFailure",
            Self::Overflow | Self::Resource(HostEffectAllocationError::Overflow) => {
                "Placement:ResourceOverflow"
            }
            _ => "Placement:ResourceLimit",
        }
    }
}

impl fmt::Display for HostAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HostAllocationError {}

impl From<HostEffectAllocationError> for HostAllocationError {
    fn from(error: HostEffectAllocationError) -> Self {
        Self::Resource(error)
    }
}

/// The current operation's host-allocation policy, without reserving bytes or
/// constructing a receipt. This is a policy fact, not an allocation grant.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum HostAllocationPolicy {
    Unbounded,
    Managed,
}

enum ResolvedHostAllocationPolicy {
    Unbounded,
    Managed {
        ledger: ManagedHostEffectLedger,
        epoch: u64,
    },
}

impl ResolvedHostAllocationPolicy {
    fn for_runtime(runtime: Option<&RuntimeContext>) -> Result<Self, HostAllocationError> {
        let Some(runtime) = runtime else {
            return Ok(Self::Unbounded);
        };
        if runtime.is_cancelled() {
            return Err(HostAllocationError::Cancelled);
        }
        let Some(parallel) = runtime.service_ports().parallel() else {
            return Ok(Self::Unbounded);
        };
        let resources = parallel.placement_resources();
        if resources.memory_available_bytes.is_none() {
            return Ok(Self::Unbounded);
        }
        let ledger = parallel
            .managed_host_effect_ledger()
            .ok_or(HostAllocationError::MissingManagedOwner)?;
        let snapshot = ledger.snapshot();
        if !snapshot.current || snapshot.epoch != resources.epoch {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(Self::Managed {
            ledger,
            epoch: resources.epoch,
        })
    }

    fn policy(&self) -> HostAllocationPolicy {
        match self {
            Self::Unbounded => HostAllocationPolicy::Unbounded,
            Self::Managed { .. } => HostAllocationPolicy::Managed,
        }
    }
}

#[derive(Clone)]
pub(super) struct AllocationFence {
    active: Arc<AtomicBool>,
    ledger: ManagedHostEffectLedger,
    epoch: u64,
    cancellation: CancellationToken,
    generation_owner: Arc<AtomicU64>,
    generation: u64,
}

impl AllocationFence {
    pub(super) fn check(&self) -> Result<(), HostAllocationError> {
        if self.cancellation.is_cancelled() {
            return Err(HostAllocationError::Cancelled);
        }
        let snapshot = self.ledger.snapshot();
        if !self.active.load(Ordering::Acquire)
            || self.generation_owner.load(Ordering::Acquire) != self.generation
            || !snapshot.current
            || snapshot.epoch != self.epoch
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(())
    }
}

/// One ordinary host call's actual managed scratch and output precharges.
/// Exact returned-value claims outlive this receipt until their storage drops.
pub struct HostAllocationReceipt {
    runtime: RuntimeContext,
    ledger: ManagedHostEffectLedger,
    stage: Arc<ManagedHostEffectStageReservation>,
    output: HostOutputPermit,
    fence: AllocationFence,
}

/// A precharged output-headroom capability. Claims survive the invocation
/// only when moved into the published value that owns the metadata.
#[derive(Clone)]
pub struct HostRetainedViewPermit {
    headroom: Arc<dyn HostEffectOutputHeadroom>,
    fence: AllocationFence,
    issuer: Arc<dyn RetainedHostViewIssuer>,
}

impl fmt::Debug for HostRetainedViewPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HostRetainedViewPermit(..)")
    }
}

struct ManagedViewIssuer {
    ledger: ManagedHostEffectLedger,
    epoch: u64,
    _own_metadata_claim: Arc<dyn ExactHostAllocationReservation>,
}

#[derive(Debug)]
struct ManagedViewGrant {
    view_bytes: u64,
    issuer: Arc<dyn RetainedHostViewIssuer>,
    _claim: Arc<dyn RetainedHostAllocationClaim>,
}

impl RetainedHostViewGrant for ManagedViewGrant {
    fn view_bytes(&self) -> u64 {
        self.view_bytes
    }

    fn issuer(&self) -> Arc<dyn RetainedHostViewIssuer> {
        self.issuer.clone()
    }

    fn backing_claim(&self) -> Arc<dyn RetainedHostAllocationClaim> {
        self._claim.clone()
    }
}

impl fmt::Debug for ManagedViewIssuer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedViewIssuer")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

pub(super) fn retained_view_issuer_host_upper_bytes() -> Result<u64, HostAllocationError> {
    let allocation = std::mem::size_of::<ManagedViewIssuer>()
        .checked_add(2 * std::mem::size_of::<usize>())
        .and_then(|bytes| bytes.checked_add(std::mem::align_of::<ManagedViewIssuer>()))
        .ok_or(HostAllocationError::Overflow)?;
    let allocation = u64::try_from(allocation).map_err(|_| HostAllocationError::Overflow)?;
    allocation
        .checked_add(crate::context::host_effect::retained_claim_host_upper_bytes())
        .ok_or(HostAllocationError::Overflow)
}

pub(super) fn retained_view_grant_host_upper_bytes() -> u64 {
    (std::mem::size_of::<ManagedViewGrant>()
        + std::mem::size_of::<usize>() * 2
        + std::mem::align_of::<ManagedViewGrant>()) as u64
}

impl RetainedHostViewIssuer for ManagedViewIssuer {
    fn reserve_view(
        self: Arc<Self>,
        view_bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, RetainedHostViewError> {
        let bytes = HostAllocationPlan::retained_view_bytes(view_bytes)
            .map_err(|_| RetainedHostViewError::Overflow)?;
        let claim = self
            .ledger
            .reserve_retained_allocation(bytes, self.epoch)
            .map_err(|error| match error {
                HostEffectAllocationError::Overflow => RetainedHostViewError::Overflow,
                HostEffectAllocationError::ExceedsCapacity => {
                    RetainedHostViewError::ExceedsCapacity
                }
                _ => RetainedHostViewError::Revoked,
            })?;
        Ok(Box::new(ManagedViewGrant {
            view_bytes,
            issuer: self,
            _claim: claim,
        }))
    }
}

impl HostRetainedViewPermit {
    pub fn issuer(&self) -> Arc<dyn RetainedHostViewIssuer> {
        self.issuer.clone()
    }
    pub fn claim_view(
        &self,
        view_bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, HostAllocationError> {
        self.fence.check()?;
        let bytes = HostAllocationPlan::retained_view_bytes(view_bytes)?;
        let bytes = usize::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?;
        let claim = self.headroom.reserve_exact(bytes)?;
        self.fence.check()?;
        Ok(Box::new(ManagedViewGrant {
            view_bytes,
            issuer: self.issuer.clone(),
            _claim: claim,
        }))
    }

    /// Fund a cache-owned variable metadata payload from the same invocation
    /// headroom. The returned claim follows the cache entry, not this permit.
    pub fn claim_metadata(
        &self,
        payload_bytes: u64,
    ) -> Result<Arc<dyn RetainedHostAllocationClaim>, HostAllocationError> {
        self.fence.check()?;
        let bytes = HostAllocationPlan::retained_metadata_bytes(payload_bytes)?;
        let bytes = usize::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?;
        let claim = self.headroom.reserve_exact(bytes)?;
        self.fence.check()?;
        Ok(claim)
    }
}

impl ProviderHostMetadataAdmission for HostRetainedViewPermit {
    fn claim_view(
        &self,
        view_bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, ProviderHostMetadataError> {
        HostRetainedViewPermit::claim_view(self, view_bytes).map_err(provider_metadata_error)
    }

    fn claim_record(
        &self,
        record_bytes: u64,
    ) -> Result<Arc<dyn RetainedHostAllocationClaim>, ProviderHostMetadataError> {
        self.claim_metadata(record_bytes)
            .map_err(provider_metadata_error)
    }
}

fn provider_metadata_error(error: HostAllocationError) -> ProviderHostMetadataError {
    match error {
        HostAllocationError::Overflow => ProviderHostMetadataError::Overflow,
        HostAllocationError::Cancelled => ProviderHostMetadataError::Cancelled,
        HostAllocationError::StaleAuthority | HostAllocationError::MissingManagedOwner => {
            ProviderHostMetadataError::StaleAuthority
        }
        HostAllocationError::Resource(HostEffectAllocationError::ExceedsCapacity) => {
            ProviderHostMetadataError::ExceedsCapacity
        }
        HostAllocationError::Resource(_) | HostAllocationError::Numeric(_) => {
            ProviderHostMetadataError::OwnerFailure
        }
    }
}

impl HostAllocationReceipt {
    /// Issue exact per-growth trace scratch against the same current managed
    /// ledger as ordinary output admission. An unbounded operation has no
    /// issuer; finite trace growth never falls back to the global allocator.
    pub(crate) fn dynamic_trace_scratch(
        runtime: &RuntimeContext,
    ) -> Result<Option<super::HostTraceScratchPermit>, HostAllocationError> {
        match ResolvedHostAllocationPolicy::for_runtime(Some(runtime))? {
            ResolvedHostAllocationPolicy::Unbounded => Ok(None),
            ResolvedHostAllocationPolicy::Managed { ledger, epoch } => {
                super::HostTraceScratchPermit::new(runtime, ledger, epoch).map(Some)
            }
        }
    }

    /// Inspect the same canonical owner/finite-policy decision used by
    /// `admit`, without reserving a stage or creating output authority.
    pub fn policy(
        runtime: Option<&RuntimeContext>,
    ) -> Result<HostAllocationPolicy, HostAllocationError> {
        Ok(ResolvedHostAllocationPolicy::for_runtime(runtime)?.policy())
    }

    /// A selected provider callback may outlive the invocation frame after
    /// cancellation. Move this exact retained metadata claim into that
    /// callback owner; no view issuer is needed for non-view control state.
    pub fn claim_retained_metadata(
        &self,
        runtime: &RuntimeContext,
        payload_bytes: u64,
    ) -> Result<Arc<dyn RetainedHostAllocationClaim>, HostAllocationError> {
        self.checkpoint(runtime)?;
        let bytes = HostAllocationPlan::retained_metadata_bytes(payload_bytes)?;
        let bytes = usize::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?;
        let claim = self.output.reserve_exact(bytes)?;
        self.checkpoint(runtime)?;
        Ok(claim)
    }

    pub fn retained_view_permit(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<HostRetainedViewPermit, HostAllocationError> {
        self.checkpoint(runtime)?;
        let issuer_bytes = retained_view_issuer_host_upper_bytes()?;
        let issuer_bytes =
            usize::try_from(issuer_bytes).map_err(|_| HostAllocationError::Overflow)?;
        let issuer_claim = self.output.reserve_exact(issuer_bytes)?;
        self.checkpoint(runtime)?;
        let issuer = Arc::new(ManagedViewIssuer {
            ledger: self.ledger.clone(),
            epoch: self.fence.epoch,
            _own_metadata_claim: issuer_claim,
        });
        self.checkpoint(runtime)?;
        Ok(HostRetainedViewPermit {
            headroom: self.output.headroom.clone(),
            fence: self.fence.clone(),
            issuer,
        })
    }
    /// Keep the scratch claim alive until an owned CPU job physically settles.
    pub fn retain_scratch_for_cpu_work(&self) -> Arc<ManagedHostEffectStageReservation> {
        self.stage.clone()
    }

    /// `None` denotes the existing unbounded policy. A finite parallel lease
    /// without the same managed owner fails closed before any allocation.
    pub fn admit(
        runtime: Option<&RuntimeContext>,
        plan: HostAllocationPlan,
    ) -> Result<Option<Self>, HostAllocationError> {
        let (ledger, epoch) = match ResolvedHostAllocationPolicy::for_runtime(runtime)? {
            ResolvedHostAllocationPolicy::Unbounded => return Ok(None),
            ResolvedHostAllocationPolicy::Managed { ledger, epoch } => (ledger, epoch),
        };
        let runtime = runtime.expect("managed allocation policy has a runtime");
        let stage = ledger.reserve_stage(plan.scratch_peak_bytes())?;
        let headroom = stage.reserve_output_headroom(plan.retained_output_bytes())?;
        let (generation_owner, generation) = runtime.host_effect_session_generation();
        let fence = AllocationFence {
            active: Arc::new(AtomicBool::new(true)),
            ledger: ledger.clone(),
            epoch,
            cancellation: runtime.cancellation(),
            generation_owner,
            generation,
        };
        let receipt = Self {
            runtime: runtime.clone(),
            ledger,
            stage,
            output: HostOutputPermit {
                headroom,
                fence: fence.clone(),
            },
            fence,
        };
        receipt.checkpoint(runtime)?;
        Ok(Some(receipt))
    }

    pub fn checkpoint(&self, runtime: &RuntimeContext) -> Result<(), HostAllocationError> {
        self.fence.check()?;
        if self.runtime.local_identity() != runtime.local_identity()
            || !self
                .runtime
                .cancellation()
                .same_generation(&runtime.cancellation())
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        let parallel = runtime
            .service_ports()
            .parallel()
            .ok_or(HostAllocationError::StaleAuthority)?;
        let owner = parallel
            .managed_host_effect_ledger()
            .ok_or(HostAllocationError::StaleAuthority)?;
        if !owner.same_owner(&self.ledger)
            || parallel.placement_resources().epoch != self.fence.epoch
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        self.fence.check()
    }

    /// Revalidate a held continuation against the exact context captured at
    /// admission. No caller-supplied context can substitute a newer session.
    pub fn checkpoint_captured(&self) -> Result<(), HostAllocationError> {
        self.checkpoint(&self.runtime)
    }

    /// Allocates one aligned, zero-initialized workspace inside the precharged
    /// scratch peak. It cannot be published as an output by this API.
    pub fn temporary_numeric(
        &self,
        runtime: &RuntimeContext,
        dtype: NumericDType,
        len: usize,
    ) -> Result<HostNumericWorkspace, HostAllocationError> {
        self.checkpoint(runtime)?;
        let result = HostNumericWorkspace::new(dtype, len, self.stage.clone(), self.fence.clone());
        // The exact allocator has a deliberately narrow physical-error type.
        // A concurrent cancellation or authority rebind takes precedence over
        // its ExceedsPermit result; a successful allocation is dropped if the
        // second checkpoint fails.
        self.checkpoint(runtime)?;
        result
    }

    pub fn temporary_complex(
        &self,
        runtime: &RuntimeContext,
        len: usize,
    ) -> Result<HostComplexWorkspace, HostAllocationError> {
        self.checkpoint(runtime)?;
        let result = HostComplexWorkspace::new(len, self.stage.clone(), self.fence.clone());
        self.checkpoint(runtime)?;
        result
    }

    /// Construct and publish one exact retained output using only this
    /// receipt's precharged headroom. The constructed result is dropped if a
    /// concurrent cancellation or rebind wins the post-allocation checkpoint;
    /// a returned Tensor retains its exact claim after the receipt retires.
    pub fn with_output_allocation<T>(
        &self,
        runtime: &RuntimeContext,
        construct: impl FnOnce(
            &dyn ExactHostAllocationPermit,
        ) -> Result<T, ExactHostNumericAllocationError>,
    ) -> Result<T, HostAllocationError> {
        self.checkpoint(runtime)?;
        let result = construct(&self.output);
        self.checkpoint(runtime)?;
        result.map_err(HostAllocationError::Numeric)
    }

    /// Retain the existing invocation fence and headroom without allocating
    /// another capability owner. Retirement revokes future allocations but
    /// does not revoke the claims already retained by published storage.
    pub(crate) fn output_permit(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<HostOutputPermit, HostAllocationError> {
        self.checkpoint(runtime)?;
        Ok(self.output.clone())
    }

    /// Retain an exact output while its owner performs cooperative host work.
    /// The caller must checkpoint again before publishing after any awaits.
    pub fn exact_output_tensor(
        &self,
        runtime: &RuntimeContext,
        dtype: NumericDType,
        shape: &[usize],
    ) -> Result<ExactHostTensorAllocation, HostAllocationError> {
        self.with_output_allocation(runtime, |permit| {
            ExactHostTensorAllocation::new(dtype, shape, permit)
        })
    }
}

impl Drop for HostAllocationReceipt {
    fn drop(&mut self) {
        self.fence.active.store(false, Ordering::Release);
        self.output.headroom.retire_invocation();
    }
}

#[derive(Clone)]
pub(crate) struct HostOutputPermit {
    headroom: Arc<dyn HostEffectOutputHeadroom>,
    fence: AllocationFence,
}

impl fmt::Debug for HostOutputPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostOutputPermit")
            .finish_non_exhaustive()
    }
}

impl ExactHostAllocationPermit for HostOutputPermit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        let claim = self.headroom.reserve_exact(bytes)?;
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        Ok(claim)
    }

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<runmat_value::ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        let claim = self.headroom.reserve_exact_claim(bytes)?;
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        Ok(claim)
    }
}
