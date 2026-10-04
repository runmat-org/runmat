//! Complete-call admission for host effects. The installed host service owns
//! resource and revision evidence; the existing placement service alone owns
//! selection and feedback. Missing evidence is never interpreted as zero.

use super::{RuntimeContext, RuntimeWholeWorkSelection};
use crate::{build_runtime_error, BuiltinResult};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementExecutionAuthorityRevision,
    PlacementResourceSnapshot, PlacementSignature, ResourceByteBound, WholeWorkCandidate,
    WholeWorkConstraint, WholeWorkObservationOutcome, WholeWorkPlacementRequest,
};
use runmat_time::{duration_ns_saturating, Instant};
use std::sync::Arc;

mod workload;
mod stage;
mod final_facts;
pub use final_facts::HostEffectFinalFacts;
pub use stage::{
    admit_host_effect_stage, select_host_effect_after_stage, try_admit_host_effect_stage,
    HostEffectStageAdmission, HostEffectStageEvidence, HostEffectStagePermit,
    HostEffectStageReservation,
};
pub use workload::{
    AudioDecodeOutput, AudioFrameRange, HostEffectOutput, HostEffectSource, HostEffectWorkload,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct HostEffectResourceEvidence {
    /// Revision of every installed service/codec participating in this route.
    pub authority_revision: Digest,
    /// Peak operation-created memory, including transport, codec, staging,
    /// and effect-only temporary result storage, excluding retained output.
    pub scratch_bytes: ResourceByteBound,
    /// Physical retained host Value bound when an output is requested.
    pub returned_value_bytes: ResourceByteBound,
}

/// Keeps the certifying authority alive and detects its revocation across an
/// await. This is only a liveness lease: it does not reserve memory, enforce
/// allocation sizes, or turn an estimate into a physical peak guarantee.
pub trait HostEffectResourceLivenessLease: Send + Sync {
    fn is_current(&self) -> bool;
}

#[cfg(test)]
pub(crate) struct TestHostEffectLivenessLease;

#[cfg(test)]
impl HostEffectResourceLivenessLease for TestHostEffectLivenessLease {
    fn is_current(&self) -> bool {
        true
    }
}

#[cfg(test)]
/// Synthetic selection tests only: this is not physical resource evidence.
pub(crate) fn test_admission(evidence: HostEffectResourceEvidence) -> HostEffectResourceAdmission {
    HostEffectResourceAdmission {
        evidence,
        liveness_lease: Arc::new(TestHostEffectLivenessLease),
        output_allocation_permit: None,
    }
}

pub struct HostEffectResourceAdmission {
    pub evidence: HostEffectResourceEvidence,
    /// Revocation guard for the installed source/codec authority. Allocation
    /// enforcement is a separate prerequisite of `evidence` being truthful.
    pub liveness_lease: Arc<dyn HostEffectResourceLivenessLease>,
    /// Optional exact output payload reservation authority. The operation
    /// owner may require it for a finite route; ordinary unbounded calls do
    /// not fabricate one. A retained output keeps its reservation until drop.
    pub output_allocation_permit: Option<Arc<dyn runmat_value::ExactHostAllocationPermit>>,
}

pub struct HostEffectRequest<'a> {
    pub operation: &'static str,
    pub workload: HostEffectWorkload<'a>,
    pub domain_facts: Digest,
    /// Exact filesystem authority later used by the effect; a digest of path
    /// text cannot prove that a different active provider has the same bounds.
    pub filesystem: &'a Arc<dyn runmat_filesystem::FsProvider>,
}

/// The caller must invoke this after pure semantic admission but before any
/// HTTP, filesystem, or codec effect. `None` means no measured route was
/// selected; a finite budget without service/revision evidence is an error.
pub fn select_host_effect(
    runtime: Option<&RuntimeContext>,
    request: HostEffectRequest<'_>,
) -> BuiltinResult<Option<HostEffectReceipt>> {
    select_host_effect_inner(runtime, request, None, None)
}

fn select_host_effect_inner(
    runtime: Option<&RuntimeContext>,
    request: HostEffectRequest<'_>,
    stage: Option<HostEffectStagePermit>,
    final_facts: Option<HostEffectFinalFacts<'_>>,
) -> BuiltinResult<Option<HostEffectReceipt>> {
    if let Some(stage) = stage.as_ref() {
        stage.validate_request(&request)?;
    }
    if let Some(facts) = final_facts {
        facts.validate_for(&request)?;
    }
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let finite = parallel.memory_available_bytes.is_some();
    let Some(service) = runtime.service_ports().placement() else {
        return if finite {
            Err(insufficient("placement service is absent"))
        } else {
            Ok(None)
        };
    };
    let Some(admission) = runtime
        .service_ports()
        .host_effect_resources()
        .and_then(|port| match (stage.as_ref(), final_facts) {
            (Some(_), Some(facts)) => port.final_admission_after_stage(&request, facts, runtime),
            (None, None) => port.admission(&request, runtime),
            _ => None,
        })
    else {
        return if finite {
            Err(insufficient(
                "host effect resource service is absent or has no complete bound",
            ))
        } else {
            Ok(None)
        };
    };
    if !admission.liveness_lease.is_current() {
        return Err(stale_resource());
    }
    let evidence = admission.evidence;
    if let Some(stage) = stage.as_ref() {
        stage.validate_final_evidence(evidence)?;
    }
    let Some(revision) = service.whole_work_host_service_revision(
        runtime.program_revision().cloned(),
        evidence.authority_revision,
    ) else {
        return if finite {
            Err(insufficient(
                "host effect placement revision is unavailable",
            ))
        } else {
            Ok(None)
        };
    };
    if revision.authority
        != PlacementExecutionAuthorityRevision::HostService(evidence.authority_revision)
    {
        return Err(
            build_runtime_error("placement returned a foreign host effect authority")
                .with_identifier("RunMat:Placement:InvalidRevision")
                .build(),
        );
    }
    if let Some(stage) = stage.as_ref() {
        stage.validate_revision(&revision)?;
    }
    let output = request.workload.output();
    let retained_bytes = match output {
        HostEffectOutput::EffectOnly => ResourceByteBound::Known(0),
        HostEffectOutput::ReturnedValue => evidence.returned_value_bytes,
    };
    if finite
        && (matches!(retained_bytes, ResourceByteBound::Unavailable(_))
            || matches!(evidence.scratch_bytes, ResourceByteBound::Unavailable(_)))
    {
        return Err(insufficient("complete-call peak memory is unavailable"));
    }
    let signature = PlacementSignature {
        region: None,
        operation: request.operation.into(),
        runtime_facts: Digest::sha256(
            serde_json::to_vec(&(
                request.workload.identity_bytes(),
                request.domain_facts,
                evidence,
            ))
            .expect("typed host effect facts serialize"),
        ),
        revision,
    };
    let candidate = WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Host,
        presentation: CandidateOutputPresentation::Direct,
        route_facts: Digest::sha256(
            serde_json::to_vec(&(request.workload.identity_bytes(), evidence))
                .expect("typed host effect route serializes"),
        ),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            retained_bytes,
            scratch_bytes: evidence.scratch_bytes,
            provider_host_workspace: None,
            provider_input_stages: None,
            queue_slots: 0,
            peak_live: None,
        },
        transactional_results: false,
    };
    let epoch_digest =
        Digest::sha256(serde_json::to_vec(&parallel.epoch).expect("parallel epoch serializes"));
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: Vec::new(),
        epoch: u64::from_le_bytes(epoch_digest.bytes()[..8].try_into().expect("digest length")),
    };
    let selected = service.select_whole_work(WholeWorkPlacementRequest {
        signature: signature.clone(),
        constraint: WholeWorkConstraint::AnyLegal,
        candidates: vec![candidate.clone()],
        resources,
        // Planner alternates are disabled for a side-effecting single route;
        // this is not a claim that filesystem or HTTP payloads repeat.
        deterministic: true,
        require_transactional_results: false,
    })?;
    if !admission.liveness_lease.is_current() {
        return Err(stale_resource());
    }
    if selected.decision.signature != signature
        || selected.decision.selected != candidate.selected(request.operation)
    {
        return Err(
            build_runtime_error("placement selected an unadmitted host effect route")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    Ok(Some(HostEffectReceipt {
        runtime: runtime.clone(),
        selected,
        started: stage
            .as_ref()
            .map_or_else(Instant::now, |stage| stage.started),
        resource_liveness_lease: admission.liveness_lease,
        output_allocation_permit: admission.output_allocation_permit,
        _stage: stage,
    }))
}

pub(crate) fn stale_resource() -> crate::RuntimeError {
    build_runtime_error("host effect resource authority changed before completion")
        .with_identifier("RunMat:Placement:StaleResourceAuthority")
        .build()
}

fn insufficient(detail: &str) -> crate::RuntimeError {
    build_runtime_error(format!(
        "host effect has insufficient resource evidence: {detail}"
    ))
    .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
    .build()
}

/// Owns the exploratory probe until all effects and output publication have
/// finished. Dropping an unpolled/cancelled call releases it without feedback.
pub struct HostEffectReceipt {
    runtime: RuntimeContext,
    selected: RuntimeWholeWorkSelection,
    started: Instant,
    resource_liveness_lease: Arc<dyn HostEffectResourceLivenessLease>,
    output_allocation_permit: Option<Arc<dyn runmat_value::ExactHostAllocationPermit>>,
    _stage: Option<HostEffectStagePermit>,
}

impl HostEffectReceipt {
    pub fn output_allocation_permit(
        &self,
    ) -> Option<Arc<dyn runmat_value::ExactHostAllocationPermit>> {
        self.output_allocation_permit.clone()
    }
    pub fn resource_liveness_lease(&self) -> Arc<dyn HostEffectResourceLivenessLease> {
        self.resource_liveness_lease.clone()
    }

    pub fn checkpoint(&self) -> BuiltinResult<()> {
        if let Some(stage) = self._stage.as_ref() {
            stage.checkpoint()?;
        }
        if self.resource_liveness_lease.is_current() {
            Ok(())
        } else {
            Err(stale_resource())
        }
    }

    pub fn finish(self, outcome: WholeWorkObservationOutcome) {
        if !self.resource_liveness_lease.is_current()
            || self
                ._stage
                .as_ref()
                .is_some_and(|stage| stage.checkpoint().is_err())
        {
            return;
        }
        let outcome = if self.runtime.is_cancelled() {
            WholeWorkObservationOutcome::Cancelled
        } else {
            outcome
        };
        if matches!(
            outcome,
            WholeWorkObservationOutcome::Cancelled | WholeWorkObservationOutcome::Abandoned
        ) {
            return;
        }
        if let Some(service) = self.runtime.service_ports().placement() {
            let _ = service.observe_selected_whole_work(
                &self.selected,
                duration_ns_saturating(self.started.elapsed()).max(1),
                outcome,
            );
        }
    }
}

#[cfg(test)]
#[path = "host_effect/tests.rs"]
mod tests;
