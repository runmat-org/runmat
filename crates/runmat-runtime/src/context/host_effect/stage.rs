//! One pre-scan resource admission for an opened source. This is not a second
//! placement decision: final whole-work selection and its probe occur once,
//! after the retained source has supplied exact result geometry.

use super::*;
use runmat_execution::{PlacementRevisionRef, ResourceByteBound};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Weak;

mod view;
mod identity;
mod authority;
pub(in crate::context) use authority::HostEffectStageAuthority;
use identity::request_identity;
use view::StageReservationView;

/// A closed classification of stage-owned temporary buffers. The kind does
/// not change quota policy; every live claim competes for one scan envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostEffectLiveAllocationKind {
    SourceStaging,
    CodecWorkspace,
    ParsedMetadata,
    TemporaryResult,
    Diagnostic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostEffectAllocationError {
    Unavailable,
    Revoked,
    Cancelled,
    InvalidInvocation,
    Overflow,
    ExceedsStageEnvelope,
    ExceedsCapacity,
    AllocationFailed,
}

impl std::fmt::Display for HostEffectAllocationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HostEffectAllocationError {}

/// The exact claim owner is held through the buffer's actual lifetime.
pub trait HostEffectLiveAllocationReservation:
    std::fmt::Debug + Send + Sync + runmat_async::DiagnosticAllocationLease
{
}

#[derive(Debug)]
pub enum HostEffectLiveAllocationClaim {
    Shared(Arc<dyn HostEffectLiveAllocationReservation>),
    Owned(runmat_async::ManagedDiagnosticAllocation),
}

impl HostEffectLiveAllocationClaim {
    /// Retain this exact physical claim in an error without adding a source
    /// wrapper or allocating a second claim owner.
    pub fn into_diagnostic_retention(self) -> runmat_async::RetainedDiagnosticAllocation {
        match self {
            Self::Shared(claim) => runmat_async::RetainedDiagnosticAllocation::Shared(claim),
            Self::Owned(claim) => runmat_async::RetainedDiagnosticAllocation::ManagedOwned(claim),
        }
    }
}

/// Owns output capacity selected for one whole call. An exact returned-value
/// allocation transfers its bytes out of this precharge into its own lifetime.
pub trait HostEffectOutputHeadroom: std::fmt::Debug + Send + Sync {
    /// Reassign already-precharged, unclaimed stage bytes to this same-ledger
    /// output owner. Only the canonical managed owner implements this; the
    /// caller must first prove that the physical stage work has settled and
    /// that its retained allocation capacities fit the transferred amount.
    fn transfer_settled_stage_bytes(
        &self,
        _stage: &super::ManagedHostEffectStageReservation,
        _bytes: u64,
        _epoch: u64,
    ) -> Result<(), HostEffectAllocationError> {
        Err(HostEffectAllocationError::Unavailable)
    }

    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<
        Arc<dyn runmat_value::ExactHostAllocationReservation>,
        runmat_value::ExactHostNumericAllocationError,
    >;

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<runmat_value::ExactHostAllocationClaim, runmat_value::ExactHostNumericAllocationError>
    {
        self.reserve_exact(bytes)
            .map(runmat_value::ExactHostAllocationClaim::Shared)
    }

    /// End this invocation's ability to allocate output and release any
    /// unclaimed precharge. Claims already transferred to values remain live
    /// until those values drop. External owners can retain their existing
    /// drop-based behavior when they do not support explicit retirement.
    fn retire_invocation(&self) {}
}

/// The selected output allocator and any invocation-local headroom to retire.
pub(super) struct StageOutputAdmission {
    pub permit: Arc<dyn runmat_value::ExactHostAllocationPermit>,
    pub headroom: Option<Arc<dyn HostEffectOutputHeadroom>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostEffectStageEvidence {
    pub authority_revision: Digest,
    /// Complete peak for opened-source/header work before final selection.
    pub scratch_bytes: ResourceByteBound,
}

/// The installed service holds a real scan reservation until the final
/// receipt is dropped and atomically charges exact output allocations to the
/// same managed-byte ledger. A size assertion or separate output budget is
/// not a valid implementation of this trait.
pub trait HostEffectStageReservation: Send + Sync {
    /// Move already-precharged unclaimed stage capacity to this invocation's
    /// setup-owned output permit. External stage owners fail closed. This is
    /// an accounting transfer, not evidence that provider work has settled.
    fn transfer_settled_stage_bytes(
        &self,
        _bytes: u64,
        _epoch: u64,
    ) -> Result<(), HostEffectAllocationError> {
        Err(HostEffectAllocationError::Unavailable)
    }
    /// Exact managed host allocation owner that atomically backs this stage.
    /// A finite stage without the parallel service's same owner is invalid.
    fn managed_ledger(&self) -> Option<super::ManagedHostEffectLedger> {
        None
    }
    fn reserved_bytes(&self) -> u64;
    /// Capacity of the same managed-allocation ledger used for the scan and
    /// every exact output reservation. This is not a process-RSS certificate.
    fn capacity_bytes(&self) -> u64;
    /// Atomically increase the same scan precharge after an opened source has
    /// supplied a retained extent. No new placement selection or authority
    /// may be substituted. Implementers without promotion fail closed.
    fn promote_scan_envelope(
        &self,
        _new_reserved_bytes: u64,
    ) -> Result<(), HostEffectAllocationError> {
        Err(HostEffectAllocationError::Unavailable)
    }
    /// Draw a temporary allocation from the already precharged scan peak.
    /// Implementers must not add these bytes again to global live capacity.
    fn reserve_live_exact(
        &self,
        _kind: HostEffectLiveAllocationKind,
        _bytes: usize,
    ) -> Result<Arc<dyn HostEffectLiveAllocationReservation>, HostEffectAllocationError> {
        Err(HostEffectAllocationError::Unavailable)
    }
    fn reserve_live_exact_claim(
        &self,
        kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        self.reserve_live_exact(kind, bytes)
            .map(HostEffectLiveAllocationClaim::Shared)
    }
    fn live_claim_metadata_bytes(&self) -> usize {
        0
    }
    /// Atomically precharge selected output headroom before the receipt is
    /// returned. Exact allocations consume this headroom without a second
    /// capacity charge; unconsumed bytes are released with its owner.
    fn reserve_output_headroom(
        &self,
        _bytes: u64,
    ) -> Result<Arc<dyn HostEffectOutputHeadroom>, HostEffectAllocationError> {
        Err(HostEffectAllocationError::Unavailable)
    }
    /// Setup-owned exact output permit for a finite staged invocation. The
    /// dynamic headroom route above remains available to external services.
    fn reserve_preallocated_output_permit(
        &self,
        _bytes: u64,
        _cancellation: runmat_execution::cancellation::CancellationToken,
        _session_generation_owner: Arc<AtomicU64>,
        _session_generation: u64,
    ) -> Result<Option<Arc<dyn runmat_value::ExactHostAllocationPermit>>, HostEffectAllocationError>
    {
        Ok(None)
    }
    fn owned_output_claim_metadata_bytes(&self) -> usize {
        0
    }
    fn is_current(&self) -> bool;

    fn slot_generation(&self) -> Option<u64> {
        None
    }

    fn invocation_epoch(&self) -> Option<Arc<AtomicBool>> {
        None
    }

    /// A pooled owner may retire its generation when the unique invocation
    /// permit drops. Dynamic reservations already release through Arc Drop.
    fn retire_invocation(&self) {}
}

struct StagedOutputPermit {
    reservation: Weak<dyn HostEffectStageReservation>,
    headroom: Arc<dyn HostEffectOutputHeadroom>,
    active_epoch: Arc<AtomicBool>,
    cancellation: runmat_execution::cancellation::CancellationToken,
    session_generation_owner: Arc<AtomicU64>,
    session_generation: u64,
    max_output_bytes: u64,
}

impl std::fmt::Debug for StagedOutputPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedOutputPermit")
            .field("max_output_bytes", &self.max_output_bytes)
            .finish_non_exhaustive()
    }
}

impl runmat_value::ExactHostAllocationPermit for StagedOutputPermit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<
        Arc<dyn runmat_value::ExactHostAllocationReservation>,
        runmat_value::ExactHostNumericAllocationError,
    > {
        if !self.active_epoch.load(Ordering::Acquire)
            || self.cancellation.is_cancelled()
            || self.session_generation_owner.load(Ordering::Acquire) != self.session_generation
        {
            return Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit);
        }
        let reservation = self
            .reservation
            .upgrade()
            .ok_or(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)?;
        let requested = u64::try_from(bytes)
            .map_err(|_| runmat_value::ExactHostNumericAllocationError::ExceedsPermit)?;
        if !reservation.is_current() {
            return Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit);
        }
        if requested > self.max_output_bytes {
            return Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit);
        }
        let authority_reservation = self.headroom.reserve_exact(bytes)?;
        if !self.active_epoch.load(Ordering::Acquire)
            || self.cancellation.is_cancelled()
            || !reservation.is_current()
            || self.session_generation_owner.load(Ordering::Acquire) != self.session_generation
        {
            return Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit);
        }
        Ok(authority_reservation)
    }

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<runmat_value::ExactHostAllocationClaim, runmat_value::ExactHostNumericAllocationError>
    {
        if !self.active_epoch.load(Ordering::Acquire)
            || self.cancellation.is_cancelled()
            || self.session_generation_owner.load(Ordering::Acquire) != self.session_generation
        {
            return Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit);
        }
        let reservation = self
            .reservation
            .upgrade()
            .ok_or(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)?;
        let requested = u64::try_from(bytes)
            .map_err(|_| runmat_value::ExactHostNumericAllocationError::ExceedsPermit)?;
        if requested > self.max_output_bytes || !reservation.is_current() {
            return Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit);
        }
        let claim = self.headroom.reserve_exact_claim(bytes)?;
        if !self.active_epoch.load(Ordering::Acquire)
            || self.cancellation.is_cancelled()
            || !reservation.is_current()
            || self.session_generation_owner.load(Ordering::Acquire) != self.session_generation
        {
            return Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit);
        }
        Ok(claim)
    }
}

pub struct HostEffectStageAdmission {
    pub evidence: HostEffectStageEvidence,
    pub reservation: Arc<dyn HostEffectStageReservation>,
}

struct PendingStageRetirement {
    reservation: Arc<dyn HostEffectStageReservation>,
    armed: bool,
}

impl Drop for PendingStageRetirement {
    fn drop(&mut self) {
        if self.armed {
            self.reservation.retire_invocation();
        }
    }
}

/// Retains the original filesystem owner, authority, and pre-scan reservation
/// across suspension. Only the matching complete call can consume it.
pub struct HostEffectStagePermit {
    authority: HostEffectStageAuthority,
    cancellation: runmat_execution::cancellation::CancellationToken,
    identity: Digest,
    filesystem: Arc<dyn runmat_filesystem::FsProvider>,
    authority_revision: Digest,
    revision: Digest,
    reservation: Arc<dyn HostEffectStageReservation>,
    slot_generation: Option<u64>,
    active_epoch: Arc<AtomicBool>,
    session_generation_owner: Arc<AtomicU64>,
    session_generation: u64,
    pub(super) started: Instant,
}

impl HostEffectStagePermit {
    /// Reserve the complete byte payload of one diagnostic under this same
    /// staged invocation. The owning claim includes its own metadata cost.
    pub fn reserve_diagnostic_claim(
        &self,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        self.reserve_live_exact_claim(HostEffectLiveAllocationKind::Diagnostic, bytes)
    }

    pub(super) fn owns_managed_observation(
        &self,
        lease: &crate::context::ManagedObservationLease,
    ) -> bool {
        self.reservation
            .managed_ledger()
            .is_some_and(|ledger| lease.same_ledger(&ledger))
    }

    pub(super) fn active_epoch(&self) -> Arc<AtomicBool> {
        self.active_epoch.clone()
    }

    /// Already-precharged scan envelope, without allocating a reservation view.
    pub fn reserved_bytes(&self) -> u64 {
        self.reservation.reserved_bytes()
    }
    /// Allocate an exact-capacity temporary buffer from this invocation's
    /// precharged scan envelope. The returned owner retains its live claim.
    pub fn allocate_exact_bytes(
        &self,
        kind: HostEffectLiveAllocationKind,
        capacity: usize,
    ) -> Result<super::ExactHostEffectBytes, HostEffectAllocationError> {
        let claim = self.reserve_live_exact_claim(kind, capacity)?;
        super::ExactHostEffectBytes::new(capacity, claim)
    }

    pub fn live_claim_metadata_bytes(&self) -> usize {
        self.reservation.live_claim_metadata_bytes()
    }

    pub fn owned_output_claim_metadata_bytes(&self) -> usize {
        self.reservation.owned_output_claim_metadata_bytes()
    }

    pub fn filesystem(&self) -> &Arc<dyn runmat_filesystem::FsProvider> {
        &self.filesystem
    }

    pub fn reservation(&self) -> Arc<dyn HostEffectStageReservation> {
        Arc::new(StageReservationView::new(
            Arc::downgrade(&self.reservation),
            self.active_epoch.clone(),
            self.cancellation.clone(),
            self.session_generation_owner.clone(),
            self.session_generation,
        ))
    }

    pub(super) fn managed_ledger(&self) -> Option<super::ManagedHostEffectLedger> {
        self.reservation.managed_ledger()
    }

    pub fn reserve_live_exact(
        &self,
        kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<Arc<dyn HostEffectLiveAllocationReservation>, HostEffectAllocationError> {
        if !self.active_epoch.load(Ordering::Acquire) {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        if !self
            .authority
            .original_cancellation_is_live(&self.cancellation)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if self.cancellation.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !self.reservation.is_current() {
            return Err(HostEffectAllocationError::Revoked);
        }
        let claim = self.reservation.reserve_live_exact(kind, bytes)?;
        if self.cancellation.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !self.reservation.is_current() {
            return Err(HostEffectAllocationError::Revoked);
        }
        if !self.active_epoch.load(Ordering::Acquire) {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        Ok(claim)
    }

    fn reserve_live_exact_claim(
        &self,
        kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        if !self.active_epoch.load(Ordering::Acquire) {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        if !self
            .authority
            .original_cancellation_is_live(&self.cancellation)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if self.cancellation.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !self.reservation.is_current() {
            return Err(HostEffectAllocationError::Revoked);
        }
        let claim = self.reservation.reserve_live_exact_claim(kind, bytes)?;
        if self.cancellation.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !self.reservation.is_current() {
            return Err(HostEffectAllocationError::Revoked);
        }
        if !self.active_epoch.load(Ordering::Acquire) {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        Ok(claim)
    }

    pub fn promote_scan_envelope(
        &self,
        new_reserved_bytes: u64,
    ) -> Result<(), HostEffectAllocationError> {
        if !self.active_epoch.load(Ordering::Acquire)
            || !self
                .authority
                .original_cancellation_is_live(&self.cancellation)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        if self.cancellation.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !self.reservation.is_current() {
            return Err(HostEffectAllocationError::Revoked);
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        self.reservation.promote_scan_envelope(new_reserved_bytes)?;
        if self.cancellation.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !self.reservation.is_current() {
            return Err(HostEffectAllocationError::Revoked);
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        Ok(())
    }

    pub(super) fn output_permit(
        &self,
        max_output_bytes: u64,
    ) -> Result<StageOutputAdmission, HostEffectAllocationError> {
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        if let Some(permit) = self.reservation.reserve_preallocated_output_permit(
            max_output_bytes,
            self.cancellation.clone(),
            self.session_generation_owner.clone(),
            self.session_generation,
        )? {
            if self.cancellation.is_cancelled() {
                return Err(HostEffectAllocationError::Cancelled);
            }
            if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation
                || !self.reservation.is_current()
            {
                return Err(HostEffectAllocationError::Revoked);
            }
            return Ok(StageOutputAdmission {
                permit,
                headroom: None,
            });
        }
        let headroom = self.reservation.reserve_output_headroom(max_output_bytes)?;
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        let permit = Arc::new(StagedOutputPermit {
            reservation: Arc::downgrade(&self.reservation),
            headroom: headroom.clone(),
            active_epoch: self.active_epoch.clone(),
            cancellation: self.cancellation.clone(),
            session_generation_owner: self.session_generation_owner.clone(),
            session_generation: self.session_generation,
            max_output_bytes,
        });
        Ok(StageOutputAdmission {
            permit,
            headroom: Some(headroom),
        })
    }

    pub(super) fn terminate_allocations(&self) {
        self.active_epoch.store(false, Ordering::Release);
    }

    pub fn checkpoint(&self) -> BuiltinResult<()> {
        if self.reservation.slot_generation() != self.slot_generation {
            return Err(stale_resource());
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(stale_resource());
        }
        let parallel = self.authority.parallel().ok_or_else(stale_resource)?;
        let parallel_owner = parallel
            .managed_host_effect_ledger()
            .ok_or_else(stale_resource)?;
        let reserved_owner = self
            .reservation
            .managed_ledger()
            .ok_or_else(stale_resource)?;
        let snapshot = parallel_owner.snapshot();
        if !parallel_owner.same_owner(&reserved_owner)
            || !snapshot.current
            || parallel.placement_resources().epoch != snapshot.epoch
        {
            return Err(stale_resource());
        }
        if !self
            .authority
            .original_cancellation_is_live(&self.cancellation)
        {
            return Err(invalid_stage(
                "staged invocation cancellation generation changed",
            ));
        }
        if self.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        if !self.reservation.is_current() {
            return Err(stale_resource());
        }
        Ok(())
    }

    fn validate_runtime(&self, runtime: &RuntimeContext) -> BuiltinResult<()> {
        self.checkpoint()?;
        if !self.authority.matches_runtime(runtime, &self.cancellation) {
            return Err(invalid_stage("final call uses a different runtime session"));
        }
        if runtime.is_cancelled() {
            return Err(cancelled());
        }
        Ok(())
    }

    pub(super) fn validate_request(&self, request: &HostEffectRequest<'_>) -> BuiltinResult<()> {
        self.checkpoint()?;
        if !Arc::ptr_eq(&self.filesystem, request.filesystem)
            || self.identity != request_identity(request)
        {
            return Err(invalid_stage(
                "staged source differs from final complete call",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_final_evidence(
        &self,
        evidence: HostEffectResourceEvidence,
    ) -> BuiltinResult<()> {
        self.checkpoint()?;
        if evidence.authority_revision != self.authority_revision
            || !matches!(evidence.scratch_bytes, ResourceByteBound::Known(bytes) if bytes <= self.reservation.reserved_bytes())
        {
            return Err(invalid_stage(
                "final route does not cover staged resource authority",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_revision(
        &self,
        revision: PlacementRevisionRef<'_>,
    ) -> BuiltinResult<()> {
        self.checkpoint()?;
        if self.revision != hash_serializable(&revision) {
            return Err(invalid_stage(
                "placement revision changed after source scan",
            ));
        }
        Ok(())
    }
}

impl Drop for HostEffectStagePermit {
    fn drop(&mut self) {
        self.active_epoch.store(false, Ordering::Release);
        self.reservation.retire_invocation();
    }
}

/// Reserve the certified scan peak before opening a file. The final call must
/// use `select_host_effect_after_stage` with this exact permit; dropping an
/// abandoned or cancelled scan releases the reservation with no observation.
pub fn admit_host_effect_stage(
    runtime: &RuntimeContext,
    request: HostEffectRequest<'_>,
) -> BuiltinResult<HostEffectStagePermit> {
    try_admit_host_effect_stage(runtime, request)?
        .ok_or_else(|| insufficient("staged source has no complete bounded admission"))
}

/// A service with complete pre-open admission need not implement a staged
/// route. An offered stage is authoritative: invalid evidence is an error,
/// never a cue to fall back after opening the source.
pub fn try_admit_host_effect_stage(
    runtime: &RuntimeContext,
    request: HostEffectRequest<'_>,
) -> BuiltinResult<Option<HostEffectStagePermit>> {
    let (session_generation_owner, session_generation) = runtime.host_effect_session_generation();
    if runtime.is_cancelled() {
        return Err(cancelled());
    }
    let parallel = runtime
        .service_ports()
        .parallel()
        .ok_or_else(|| insufficient("staged scan requires an installed parallel resource owner"))?;
    let resources = parallel.placement_resources();
    let _available = resources
        .memory_available_bytes
        .ok_or_else(|| insufficient("staged scan requires an installed finite memory budget"))?;
    // A complete direct-admission service may decline staging. Capture any
    // managed owner before asking for a stage, but require it only if a stage
    // is actually offered.
    let managed = parallel.managed_host_effect_ledger();
    let before = managed.as_ref().map(ManagedHostEffectLedger::snapshot);
    let resource_service = runtime
        .service_ports()
        .host_effect_resources()
        .ok_or_else(|| insufficient("staged source resource service is absent"))?;
    let Some(admission) = resource_service.stage_admission(&request, runtime) else {
        return Ok(None);
    };
    let mut pending_retirement = PendingStageRetirement {
        reservation: admission.reservation.clone(),
        armed: true,
    };
    let managed =
        managed.ok_or_else(|| insufficient("staged scan has no managed allocation owner"))?;
    let before = before.expect("managed owner has a captured snapshot");
    if !before.current || before.epoch != resources.epoch {
        return Err(stale_resource());
    }
    let ResourceByteBound::Known(scratch_bytes) = admission.evidence.scratch_bytes else {
        return Err(insufficient("staged scan peak is unavailable"));
    };
    let reservation_owner = admission
        .reservation
        .managed_ledger()
        .ok_or_else(|| insufficient("staged reservation has no managed allocation owner"))?;
    let after = managed.snapshot();
    if !managed.same_owner(&reservation_owner)
        || !after.current
        || after.epoch != before.epoch
        || parallel.placement_resources().epoch != after.epoch
    {
        return Err(stale_resource());
    }
    if !admission.reservation.is_current() {
        return Err(stale_resource());
    }
    if session_generation_owner.load(Ordering::Acquire) != session_generation {
        return Err(stale_resource());
    }
    if admission.reservation.reserved_bytes() < scratch_bytes
        || admission.reservation.reserved_bytes() > admission.reservation.capacity_bytes()
        || admission.reservation.capacity_bytes() != after.capacity_bytes
    {
        return Err(insufficient(
            "staged scan reservation does not fit the memory budget",
        ));
    }
    let placement = runtime
        .service_ports()
        .placement()
        .ok_or_else(|| insufficient("placement service is absent"))?;
    let revision = placement
        .whole_work_host_service_revision_ref(
            runtime.program_revision(),
            admission.evidence.authority_revision,
        )
        .ok_or_else(|| insufficient("staged scan placement revision is unavailable"))?;
    if revision.authority
        != PlacementExecutionAuthorityRevision::HostService(admission.evidence.authority_revision)
    {
        return Err(invalid_stage(
            "staged scan has a foreign placement authority",
        ));
    }
    if runtime.is_cancelled() {
        return Err(cancelled());
    }
    if !admission.reservation.is_current() {
        return Err(stale_resource());
    }
    if session_generation_owner.load(Ordering::Acquire) != session_generation {
        return Err(stale_resource());
    }
    let slot_generation = admission.reservation.slot_generation();
    let active_epoch = admission
        .reservation
        .invocation_epoch()
        .unwrap_or_else(|| Arc::new(AtomicBool::new(true)));
    let permit = HostEffectStagePermit {
        cancellation: runtime.cancellation(),
        authority: runtime.capture_host_effect_stage_authority(),
        identity: request_identity(&request),
        filesystem: request.filesystem.clone(),
        authority_revision: admission.evidence.authority_revision,
        revision: hash_serializable(&revision),
        reservation: admission.reservation,
        slot_generation,
        active_epoch,
        session_generation_owner,
        session_generation,
        started: Instant::now(),
    };
    pending_retirement.armed = false;
    Ok(Some(permit))
}

pub fn select_host_effect_after_stage(
    runtime: &RuntimeContext,
    request: HostEffectRequest<'_>,
    final_facts: HostEffectFinalFacts<'_>,
    stage: HostEffectStagePermit,
) -> BuiltinResult<HostEffectReceipt> {
    stage.validate_runtime(runtime)?;
    stage.validate_request(&request)?;
    let receipt = select_host_effect_inner(Some(runtime), request, Some(stage), Some(final_facts))?
        .ok_or_else(|| insufficient("staged final complete-call selection is unavailable"))?;
    receipt
        ._stage
        .as_ref()
        .expect("selected staged receipt")
        .checkpoint()?;
    Ok(receipt)
}

pub(super) fn invalid_stage(detail: &str) -> crate::RuntimeError {
    build_runtime_error(detail)
        .with_identifier("RunMat:Placement:InvalidStage")
        .build()
}

fn cancelled() -> crate::RuntimeError {
    build_runtime_error("host effect scan was cancelled")
        .with_identifier("RunMat:Cancelled")
        .build()
}
