//! One pre-scan resource admission for an opened source. This is not a second
//! placement decision: final whole-work selection and its probe occur once,
//! after the retained source has supplied exact result geometry.

use super::*;
use runmat_execution::{PlacementRevision, ResourceByteBound};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostEffectStageEvidence {
    pub authority_revision: Digest,
    /// Complete peak for opened-source/header work before final selection.
    pub scratch_bytes: ResourceByteBound,
}

/// The installed service holds a real scratch reservation until the final
/// receipt is dropped. A size assertion without an enforcing reservation is
/// not a valid implementation of this trait.
pub trait HostEffectStageReservation: Send + Sync {
    fn reserved_bytes(&self) -> u64;
    fn is_current(&self) -> bool;
}

pub struct HostEffectStageAdmission {
    pub evidence: HostEffectStageEvidence,
    pub reservation: Arc<dyn HostEffectStageReservation>,
}

/// Retains the original filesystem owner, authority, and pre-scan reservation
/// across suspension. Only the matching complete call can consume it.
pub struct HostEffectStagePermit {
    runtime: RuntimeContext,
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

    pub fn checkpoint(&self) -> BuiltinResult<()> {
        if self.runtime.is_cancelled() {
            return Err(cancelled());
        }
        if !self.reservation.is_current() {
            return Err(stale_resource());
        }
        Ok(())
    }

    fn validate_runtime(&self, runtime: &RuntimeContext) -> BuiltinResult<()> {
        self.checkpoint()?;
        if self.runtime.local_identity() != runtime.local_identity() {
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
    let admission = resource_service
        .stage_admission(&request, runtime)
        .ok_or_else(|| insufficient("staged source has no complete bounded admission"))?;
    let ResourceByteBound::Known(scratch_bytes) = admission.evidence.scratch_bytes else {
        return Err(insufficient("staged scan peak is unavailable"));
    };
    if !admission.reservation.is_current() {
        return Err(stale_resource());
    }
    if admission.reservation.reserved_bytes() < scratch_bytes
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
    Ok(HostEffectStagePermit {
        runtime: runtime.clone(),
        identity: request_identity(&request),
        filesystem: request.filesystem.clone(),
        authority_revision: admission.evidence.authority_revision,
        revision,
        reservation: admission.reservation,
        started: Instant::now(),
    })
}

pub fn select_host_effect_after_stage(
    runtime: &RuntimeContext,
    request: HostEffectRequest<'_>,
    final_facts: &dyn std::any::Any,
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

fn invalid_stage(detail: &str) -> crate::RuntimeError {
    build_runtime_error(detail)
        .with_identifier("RunMat:Placement:InvalidStage")
        .build()
}

fn cancelled() -> crate::RuntimeError {
    build_runtime_error("host effect scan was cancelled")
        .with_identifier("RunMat:Cancelled")
        .build()
}
