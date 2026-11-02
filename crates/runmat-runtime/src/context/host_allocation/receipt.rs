use std::fmt;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use runmat_execution::resource::{
    ProviderHostMetadataAdmission, ProviderHostMetadataError, RetainedHostAllocationClaim,
    RetainedHostViewError, RetainedHostViewGrant, RetainedHostViewIssuer,
};
use runmat_value::{
    ExactHostAllocationPermit, ExactHostAllocationReservation, ExactHostNumericAllocationError,
    ExactHostTensorAllocation, NumericDType,
};

use super::checkpoint::{AllocationOwnerState, HostAllocationCheckpoint};
use super::{
    AlignedHostWorkspace, HostAllocationPlan, HostComplexWorkspace, HostNumericWorkspace,
    HostTypedElement, HostTypedWorkspace,
};
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
        Self::for_parallel(runtime.service_ports().parallel().map(|port| port.as_ref()))
    }

    fn for_parallel(
        parallel: Option<&dyn crate::context::RuntimeParallelService>,
    ) -> Result<Self, HostAllocationError> {
        let Some(parallel) = parallel else {
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
    owner: AllocationOwnerState,
}

impl AllocationFence {
    pub(super) fn check(&self) -> Result<(), HostAllocationError> {
        self.owner.check()?;
        if !self.active.load(Ordering::Acquire) {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(())
    }
}

/// One ordinary host call's actual managed scratch and output precharges.
/// Exact returned-value claims outlive this receipt until their storage drops.
pub struct HostAllocationReceipt {
    checkpoint: HostAllocationCheckpoint,
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

pub(super) fn provider_metadata_error(error: HostAllocationError) -> ProviderHostMetadataError {
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
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn worker_checkpoint(&self) -> super::checkpoint::HostAllocationWorkerCheckpoint {
        super::checkpoint::HostAllocationWorkerCheckpoint::from_checkpoint(&self.checkpoint)
    }
    /// Issue exact per-growth trace scratch against the same current managed
    /// ledger as ordinary output admission. An unbounded operation has no
    /// issuer; finite trace growth never falls back to the global allocator.
    pub(crate) fn dynamic_trace_scratch(
        runtime: &RuntimeContext,
    ) -> Result<Option<super::HostTraceScratchPermit>, HostAllocationError> {
        super::HostControlAllocationOwner::capture(runtime)
            .map(|owner| owner.map(super::HostTraceScratchPermit::new))
    }

    /// Inspect the same canonical owner/finite-policy decision used by
    /// `admit`, without reserving a stage or creating output authority.
    pub fn policy(
        runtime: Option<&RuntimeContext>,
    ) -> Result<HostAllocationPolicy, HostAllocationError> {
        Ok(ResolvedHostAllocationPolicy::for_runtime(runtime)?.policy())
    }

    /// Read the same policy from a captured original parallel port. This
    /// returns no grant, reservation or runtime; callers check cancellation
    /// independently before and after invoking the external read-only port.
    pub(crate) fn policy_for_parallel(
        parallel: Option<&dyn crate::context::RuntimeParallelService>,
    ) -> Result<HostAllocationPolicy, HostAllocationError> {
        Ok(ResolvedHostAllocationPolicy::for_parallel(parallel)?.policy())
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
            ledger: self.checkpoint.owner.ledger.clone(),
            epoch: self.checkpoint.owner.epoch,
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
        let Some(checkpoint) = Self::capture_owner(runtime)? else {
            return Ok(None);
        };
        let runtime = runtime.expect("managed allocation policy has a runtime");
        let receipt = Self::admit_captured(checkpoint, plan)?;
        receipt.checkpoint(runtime)?;
        Ok(Some(receipt))
    }

    /// Reserve only through the captured original ledger/epoch/token. This
    /// exposes no substitute RuntimeContext or ambient-policy fallback.
    pub(crate) fn admit_captured(
        checkpoint: HostAllocationCheckpoint,
        plan: HostAllocationPlan,
    ) -> Result<Self, HostAllocationError> {
        checkpoint.check_captured()?;
        let stage = checkpoint
            .owner
            .ledger
            .reserve_stage(plan.scratch_peak_bytes())?;
        let headroom = stage.reserve_output_headroom(plan.retained_output_bytes())?;
        let fence = AllocationFence {
            active: Arc::new(AtomicBool::new(true)),
            owner: checkpoint.owner.clone(),
        };
        let receipt = Self {
            checkpoint,
            stage,
            output: HostOutputPermit {
                headroom,
                fence: fence.clone(),
            },
            fence,
        };
        receipt.checkpoint_captured()?;
        Ok(receipt)
    }

    pub fn checkpoint(&self, runtime: &RuntimeContext) -> Result<(), HostAllocationError> {
        self.checkpoint.check(runtime)?;
        self.fence.check()
    }

    /// Capture only the original owner identity. This allocates nothing and
    /// carries neither a live stage nor any allocation capability. It remains
    /// useful after a particular receipt retires, but not after owner rotation.
    pub(crate) fn capture_owner(
        runtime: Option<&RuntimeContext>,
    ) -> Result<Option<HostAllocationCheckpoint>, HostAllocationError> {
        let (ledger, epoch) = match ResolvedHostAllocationPolicy::for_runtime(runtime)? {
            ResolvedHostAllocationPolicy::Unbounded => return Ok(None),
            ResolvedHostAllocationPolicy::Managed { ledger, epoch } => (ledger, epoch),
        };
        let runtime = runtime.expect("managed allocation policy has a runtime");
        let (generation_owner, generation) = runtime.host_effect_session_generation();
        let checkpoint = HostAllocationCheckpoint {
            context: runtime
                .capture_host_allocation_context()
                .ok_or(HostAllocationError::StaleAuthority)?,
            owner: AllocationOwnerState {
                ledger,
                epoch,
                cancellation: runtime.cancellation(),
                generation_owner,
                generation,
            },
        };
        checkpoint.check(runtime)?;
        Ok(Some(checkpoint))
    }

    /// Reassign settled, unclaimed scratch precharge to this receipt's
    /// retained-output headroom without increasing the ledger reservation.
    /// The caller must first prove every provider download/callback touching
    /// these bytes has physically retired and that the actual retained Vec
    /// capacities fit the transferred headroom. This method does not inspect
    /// or authorize provider data. A failed post-transfer checkpoint leaves
    /// coherent shares but forbids publication.
    pub fn transfer_settled_scratch_to_output(
        &self,
        runtime: &RuntimeContext,
        bytes: u64,
    ) -> Result<(), HostAllocationError> {
        self.checkpoint(runtime)?;
        self.output.headroom.transfer_settled_stage_bytes(
            &self.stage,
            bytes,
            self.checkpoint.owner.epoch,
        )?;
        self.checkpoint(runtime)
    }

    /// Revalidate a held continuation against the exact context captured at
    /// admission. No caller-supplied context can substitute a newer session.
    pub fn checkpoint_captured(&self) -> Result<(), HostAllocationError> {
        self.checkpoint.check_captured()?;
        self.fence.check()
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

    /// Admit one exact, non-growing typed numeric scratch backing. The
    /// returned owner keeps the physical allocation and stage charge coupled.
    pub fn temporary_typed<T: HostTypedElement>(
        &self,
        runtime: &RuntimeContext,
        len: usize,
    ) -> Result<HostTypedWorkspace<T>, HostAllocationError> {
        self.checkpoint(runtime)?;
        let result = self.temporary_typed_captured(len);
        self.checkpoint(runtime)?;
        result
    }

    pub(crate) fn temporary_typed_captured<T: HostTypedElement>(
        &self,
        len: usize,
    ) -> Result<HostTypedWorkspace<T>, HostAllocationError> {
        self.checkpoint_captured()?;
        let result = HostTypedWorkspace::new(len, self.stage.clone(), self.fence.clone());
        self.checkpoint_captured()?;
        result
    }

    /// Own one aligned, possibly uninitialized numerical scratch block with
    /// its exact physical claim. Its bytes are never reinterpreted through an
    /// initialized typed Vec, and zero-sized layouts require no allocation.
    pub(crate) fn temporary_aligned(
        &self,
        runtime: &RuntimeContext,
        layout: std::alloc::Layout,
    ) -> Result<AlignedHostWorkspace, HostAllocationError> {
        self.checkpoint(runtime)?;
        let result =
            AlignedHostWorkspace::new_claimed(layout, self.stage.clone(), self.fence.clone());
        self.checkpoint(runtime)?;
        result
    }

    /// Construct and publish one exact retained output using only this
    /// receipt's precharged headroom. The constructed result is dropped if a
    /// concurrent cancellation or rebind wins the post-allocation checkpoint;
    /// a returned Tensor retains its exact claim after the receipt retires.
    pub fn with_output_construction<T, E>(
        &self,
        runtime: &RuntimeContext,
        construct: impl FnOnce(&dyn ExactHostAllocationPermit) -> Result<T, E>,
    ) -> Result<Result<T, E>, HostAllocationError> {
        self.checkpoint(runtime)?;
        let result = construct(&self.output);
        self.checkpoint(runtime)?;
        Ok(result)
    }

    pub fn with_output_allocation<T>(
        &self,
        runtime: &RuntimeContext,
        construct: impl FnOnce(
            &dyn ExactHostAllocationPermit,
        ) -> Result<T, ExactHostNumericAllocationError>,
    ) -> Result<T, HostAllocationError> {
        self.with_output_construction(runtime, construct)?
            .map_err(HostAllocationError::Numeric)
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
