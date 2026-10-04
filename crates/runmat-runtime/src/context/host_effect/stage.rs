//! One pre-scan resource admission for an opened source. This is not a second
//! placement decision: final whole-work selection and its probe occur once,
//! after the retained source has supplied exact result geometry.

use super::*;
use runmat_execution::{PlacementRevision, ResourceByteBound};
use std::sync::atomic::{AtomicU64, Ordering};

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
    fn reserved_bytes(&self) -> u64;
    /// Capacity of the same managed-allocation ledger used for the scan and
    /// every exact output reservation. This is not a process-RSS certificate.
    fn capacity_bytes(&self) -> u64;
    /// Atomically charge this exact output against the scan's live charge and
    /// all other reservations on this authority. Implementations must reject
    /// revocation and keep the charge until the returned owner is dropped.
    /// The default deliberately makes staged returned values unavailable.
    fn reserve_output_exact(
        &self,
        _bytes: usize,
    ) -> Result<
        Arc<dyn runmat_value::ExactHostAllocationReservation>,
        runmat_value::ExactHostNumericAllocationError,
    > {
        Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
    }
    fn is_current(&self) -> bool;
}

struct StagedOutputPermit {
    reservation: Arc<dyn HostEffectStageReservation>,
    max_output_bytes: u64,
    live_output_bytes: Arc<AtomicU64>,
}

#[derive(Debug)]
struct StagedOutputReservation {
    _authority_reservation: Arc<dyn runmat_value::ExactHostAllocationReservation>,
    live_output_bytes: Arc<AtomicU64>,
    bytes: u64,
}

impl Drop for StagedOutputReservation {
    fn drop(&mut self) {
        self.live_output_bytes
            .fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

impl runmat_value::ExactHostAllocationReservation for StagedOutputReservation {}

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
        let requested = u64::try_from(bytes)
            .map_err(|_| runmat_value::ExactHostNumericAllocationError::ExceedsPermit)?;
        if !self.reservation.is_current() {
            return Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit);
        }
        self.live_output_bytes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |live| {
                live.checked_add(requested)
                    .filter(|total| *total <= self.max_output_bytes)
            })
            .map_err(|_| runmat_value::ExactHostNumericAllocationError::ExceedsPermit)?;
        let authority_reservation = match self.reservation.reserve_output_exact(bytes) {
            Ok(reservation) => reservation,
            Err(error) => {
                self.live_output_bytes
                    .fetch_sub(requested, Ordering::SeqCst);
                return Err(error);
            }
        };
        if !self.reservation.is_current() {
            self.live_output_bytes
                .fetch_sub(requested, Ordering::SeqCst);
            return Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit);
        }
        Ok(Arc::new(StagedOutputReservation {
            _authority_reservation: authority_reservation,
            live_output_bytes: self.live_output_bytes.clone(),
            bytes: requested,
        }))
    }
}

pub struct HostEffectStageAdmission {
    pub evidence: HostEffectStageEvidence,
    pub reservation: Arc<dyn HostEffectStageReservation>,
}

/// Retains the original filesystem owner, authority, and pre-scan reservation
/// across suspension. Only the matching complete call can consume it.
pub struct HostEffectStagePermit {
    runtime: RuntimeContext,
    cancellation: runmat_execution::cancellation::CancellationToken,
    identity: Digest,
    filesystem: Arc<dyn runmat_filesystem::FsProvider>,
    authority_revision: Digest,
    revision: PlacementRevision,
    reservation: Arc<dyn HostEffectStageReservation>,
    pub(super) started: Instant,
}

impl HostEffectStagePermit {
    pub fn filesystem(&self) -> &Arc<dyn runmat_filesystem::FsProvider> {
        &self.filesystem
    }

    pub fn reservation(&self) -> Arc<dyn HostEffectStageReservation> {
        self.reservation.clone()
    }

    pub(super) fn output_permit(
        &self,
        max_output_bytes: u64,
    ) -> Arc<dyn runmat_value::ExactHostAllocationPermit> {
        Arc::new(StagedOutputPermit {
            reservation: self.reservation.clone(),
            max_output_bytes,
            live_output_bytes: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn checkpoint(&self) -> BuiltinResult<()> {
        if !self
            .cancellation
            .same_generation(&self.runtime.cancellation())
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
        if self.runtime.local_identity() != runtime.local_identity()
            || !self
                .runtime
                .service_ports()
                .same_host_effect_owners(runtime.service_ports())
            || !std::rc::Rc::ptr_eq(self.runtime.execution(), runtime.execution())
            || self.runtime.program_revision() != runtime.program_revision()
            || !self.cancellation.same_generation(&runtime.cancellation())
        {
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
            || !matches!(evidence.scratch_bytes, ResourceByteBound::Known(bytes) if bytes >= self.reservation.reserved_bytes())
        {
            return Err(invalid_stage(
                "final route does not cover staged resource authority",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_revision(&self, revision: &PlacementRevision) -> BuiltinResult<()> {
        self.checkpoint()?;
        if &self.revision != revision {
            return Err(invalid_stage(
                "placement revision changed after source scan",
            ));
        }
        Ok(())
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
    if runtime.is_cancelled() {
        return Err(cancelled());
    }
    let available = runtime
        .service_ports()
        .parallel()
        .and_then(|port| port.placement_resources().memory_available_bytes)
        .ok_or_else(|| insufficient("staged scan requires an installed finite memory budget"))?;
    let resource_service = runtime
        .service_ports()
        .host_effect_resources()
        .ok_or_else(|| insufficient("staged source resource service is absent"))?;
    let Some(admission) = resource_service.stage_admission(&request, runtime) else {
        return Ok(None);
    };
    let ResourceByteBound::Known(scratch_bytes) = admission.evidence.scratch_bytes else {
        return Err(insufficient("staged scan peak is unavailable"));
    };
    if !admission.reservation.is_current() {
        return Err(stale_resource());
    }
    if admission.reservation.reserved_bytes() < scratch_bytes
        || admission.reservation.reserved_bytes() > admission.reservation.capacity_bytes()
        || admission.reservation.capacity_bytes() > available
        || admission.reservation.reserved_bytes() > available
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
        .whole_work_host_service_revision(
            runtime.program_revision().cloned(),
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
    Ok(Some(HostEffectStagePermit {
        cancellation: runtime.cancellation(),
        runtime: runtime.clone(),
        identity: request_identity(&request),
        filesystem: request.filesystem.clone(),
        authority_revision: admission.evidence.authority_revision,
        revision,
        reservation: admission.reservation,
        started: Instant::now(),
    }))
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

fn request_identity(request: &HostEffectRequest<'_>) -> Digest {
    // The complete call's public operation, source, options, and output mode
    // are fixed before scan. `domain_facts` is deliberately excluded because
    // it becomes the checked post-header geometry digest at final selection.
    Digest::sha256(
        serde_json::to_vec(&(request.operation, request.workload.identity_bytes()))
            .expect("typed stage request serializes"),
    )
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
