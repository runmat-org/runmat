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
    WholeWorkConstraint, WholeWorkObservation, WholeWorkObservationOutcome,
    WholeWorkPlacementRequest,
};
use runmat_time::{duration_ns_saturating, Instant};

/// The requested result is distinct from the committed side effects. No
/// output still needs a bound for the temporary filename list and codec work.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum HostEffectOutput {
    EffectOnly,
    ReturnedValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum HostEffectSource {
    LocalArchive,
    HttpArchive,
}

/// Typed operation identity at the service evidence boundary; no builtin
/// spelling switch or secondary scheduler is introduced here.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum HostEffectWorkload {
    ArchiveExtract {
        source: HostEffectSource,
        output: HostEffectOutput,
    },
}

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

pub struct HostEffectRequest {
    pub operation: &'static str,
    pub workload: HostEffectWorkload,
    pub domain_facts: Digest,
}

/// The caller must invoke this after pure semantic admission but before any
/// HTTP, filesystem, or codec effect. `None` means no measured route was
/// selected; a finite budget without service/revision evidence is an error.
pub fn select_host_effect(
    runtime: Option<&RuntimeContext>,
    request: HostEffectRequest,
) -> BuiltinResult<Option<HostEffectReceipt>> {
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
    let Some(evidence) = runtime
        .service_ports()
        .host_effect_resources()
        .and_then(|port| port.admission(&request))
    else {
        return if finite {
            Err(insufficient(
                "host effect resource service is absent or has no complete bound",
            ))
        } else {
            Ok(None)
        };
    };
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
    let output = match request.workload {
        HostEffectWorkload::ArchiveExtract { output, .. } => output,
    };
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
            serde_json::to_vec(&(request.workload, request.domain_facts, evidence))
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
            serde_json::to_vec(&(request.workload, evidence))
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
        deterministic: true,
        require_transactional_results: false,
    })?;
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
        started: Instant::now(),
    }))
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
}

impl HostEffectReceipt {
    pub fn finish(self, outcome: WholeWorkObservationOutcome) {
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
            let _ = service.observe_whole_work(WholeWorkObservation {
                signature: self.selected.decision.signature.clone(),
                selected: self.selected.decision.selected.clone(),
                elapsed_ns: duration_ns_saturating(self.started.elapsed()).max(1),
                outcome,
            });
        }
    }
}

#[cfg(test)]
#[path = "host_effect/tests.rs"]
mod tests;
