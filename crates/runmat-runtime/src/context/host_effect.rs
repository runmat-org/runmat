//! Complete-call admission for host effects. The installed host service owns
//! resource and revision evidence; the existing placement service alone owns
//! selection and feedback. Missing evidence is never interpreted as zero.

use super::{RuntimeContext, RuntimeManagedWholeWorkSelection, RuntimeWholeWorkSelection};
use crate::{build_runtime_error, BuiltinResult};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementExecutionAuthorityRevision,
    PlacementResourceSnapshot, PlacementSignature, PlacementSignatureRef, ResourceByteBound,
    WholeWorkCandidate, WholeWorkConstraint, WholeWorkObservationOutcome,
    WholeWorkPlacementRequest, WholeWorkPlacementRequestRef,
};
use runmat_time::{duration_ns_saturating, Instant};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

mod exact_bytes;
mod exact_diagnostic;
mod final_facts;
mod managed_ledger;
pub(crate) use managed_ledger::{
    owned_output_claim_metadata_bytes, retained_claim_host_upper_bytes,
};
mod output_permit;
mod stage;
pub(in crate::context) use stage::HostEffectStageAuthority;
mod workload;
pub(crate) fn hash_serializable<T: serde::Serialize>(facts: &T) -> Digest {
    Digest::sha256_serialized(facts).expect("typed host effect facts serialize")
}
pub use exact_bytes::ExactHostEffectBytes;
pub(crate) use exact_diagnostic::ExactHostEffectDiagnosticStrings;
pub use final_facts::HostEffectFinalFacts;
pub use managed_ledger::{
    ManagedHostEffectLedger, ManagedHostEffectLedgerSnapshot, ManagedHostEffectStagePool,
    ManagedHostEffectStageReservation, ManagedHostEffectStageSlot,
};
pub use stage::{
    admit_host_effect_stage, select_host_effect_after_stage, try_admit_host_effect_stage,
    HostEffectAllocationError, HostEffectLiveAllocationClaim, HostEffectLiveAllocationKind,
    HostEffectLiveAllocationReservation, HostEffectOutputHeadroom, HostEffectStageAdmission,
    HostEffectStageEvidence, HostEffectStagePermit, HostEffectStageReservation,
};
pub use workload::{
    AudioDecodeOutput, AudioFrameRange, DatasetHandleInput, DatasetHandleKey,
    DatasetHandleOperation, DatasetHandleRole, DatasetPathOperation, HostEffectOutput,
    HostEffectSource, HostEffectWorkload, MatAssignmentSuffix, MatDecodeOutput, MatDecodeSelection,
    MatEncodeMode, MatEncodeOutput, MatHandleControl, MatStructSelection,
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
    /// Direct/a-priori exact output authority only. A staged call must leave
    /// this empty: its output permit is derived from the captured scan
    /// reservation so scan and output compete in one managed byte ledger.
    /// A retained output keeps its reservation until drop.
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
    if !request.workload.identity_complete() {
        return if finite {
            Err(insufficient(
                "host effect control or input identity is incomplete",
            ))
        } else {
            Ok(None)
        };
    }
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
        if admission.output_allocation_permit.is_some() {
            return Err(stage::invalid_stage(
                "final route supplied a separate output allocation authority",
            ));
        }
    }
    let managed_revision = if stage.is_some() {
        service.whole_work_host_service_revision_ref(
            runtime.program_revision(),
            evidence.authority_revision,
        )
    } else {
        None
    };
    let mut portable_revision = if stage.is_none() {
        service.whole_work_host_service_revision(
            runtime.program_revision().cloned(),
            evidence.authority_revision,
        )
    } else {
        None
    };
    let Some(revision) = managed_revision.or_else(|| portable_revision.as_ref().map(Into::into))
    else {
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
        stage.validate_revision(revision)?;
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
    let runtime_facts = match request.workload {
        HostEffectWorkload::ReadMatrixText {
            source,
            controls,
            output,
        } => hash_serializable(&(
            source.as_os_str().as_encoded_bytes(),
            controls,
            output,
            request.domain_facts,
            evidence,
        )),
        _ => Digest::sha256(
            serde_json::to_vec(&(
                request.workload.identity_bytes(),
                request.domain_facts,
                evidence,
            ))
            .expect("typed host effect facts serialize"),
        ),
    };
    let portable_signature = portable_revision.take().map(|revision| PlacementSignature {
        region: None,
        operation: request.operation.into(),
        runtime_facts,
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    });
    let signature = PlacementSignatureRef {
        region: None,
        operation: request.operation,
        runtime_facts,
        provider_provenance: None,
        graph_output_authority: None,
        revision: managed_revision
            .or_else(|| {
                portable_signature
                    .as_ref()
                    .map(|signature| (&signature.revision).into())
            })
            .expect("placement revision was checked"),
    };
    // A staged scratch reservation remains charged while the returned value
    // is allocated. The final measured scratch bound may be lower than that
    // still-live precharge; placement must account for the larger amount.
    let placement_scratch = match (stage.as_ref(), evidence.scratch_bytes) {
        (Some(stage), ResourceByteBound::Known(final_bytes)) => {
            ResourceByteBound::Known(final_bytes.max(stage.reserved_bytes()))
        }
        (_, bound) => bound,
    };
    let route_facts = match request.workload {
        HostEffectWorkload::ReadMatrixText {
            source,
            controls,
            output,
        } => hash_serializable(&(
            source.as_os_str().as_encoded_bytes(),
            controls,
            output,
            evidence,
        )),
        _ => Digest::sha256(
            serde_json::to_vec(&(request.workload.identity_bytes(), evidence))
                .expect("typed host effect route serializes"),
        ),
    };
    let candidate = WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Host,
        presentation: CandidateOutputPresentation::Direct,
        route_facts,
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            retained_bytes,
            scratch_bytes: placement_scratch,
            provider_host_workspace: None,
            provider_input_stages: None,
            queue_slots: 0,
            peak_live: None,
        },
        transactional_results: false,
    };
    let epoch_digest =
        Digest::sha256(serde_json::to_vec(&parallel.epoch).expect("parallel epoch serializes"));
    // The staged reservation has already atomically debited the installed
    // parallel owner's free bytes. Return only this invocation's own precharge
    // to the placement comparison; other live stages remain unavailable.
    let placement_memory = match stage.as_ref() {
        Some(stage) => {
            let owner = stage
                .managed_ledger()
                .ok_or_else(|| insufficient("staged allocation owner is absent"))?;
            let parallel_owner = runtime
                .service_ports()
                .parallel()
                .and_then(|parallel| parallel.managed_host_effect_ledger())
                .ok_or_else(|| insufficient("parallel allocation owner is absent"))?;
            if !owner.same_owner(&parallel_owner) {
                return Err(stale_resource());
            }
            let snapshot = owner.snapshot();
            if !snapshot.current || snapshot.epoch != parallel.epoch {
                return Err(stale_resource());
            }
            Some(
                snapshot
                    .free_bytes()
                    .checked_add(stage.reserved_bytes())
                    .ok_or_else(|| insufficient("staged resource snapshot overflow"))?,
            )
        }
        None => parallel.memory_available_bytes,
    };
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: placement_memory,
        cancellation_requested: runtime.is_cancelled(),
        providers: Vec::new(),
        epoch: u64::from_le_bytes(epoch_digest.bytes()[..8].try_into().expect("digest length")),
    };
    let expected_candidate_digest = candidate.identity_digest();
    let expected_candidate_kind = candidate.kind;
    let selected = if stage.is_some() {
        let candidates = [candidate];
        SelectedHostEffect::Managed(service.select_whole_work_borrowed_managed(
            WholeWorkPlacementRequestRef {
                signature,
                constraint: WholeWorkConstraint::AnyLegal,
                candidates: &candidates,
                resources: &resources,
                // Planner alternates are disabled for a side-effecting single
                // route; this does not assert that the effect can be replayed.
                deterministic: true,
                require_transactional_results: false,
            },
        )?)
    } else {
        SelectedHostEffect::Portable(
            service.select_whole_work(WholeWorkPlacementRequest {
                signature: portable_signature
                    .as_ref()
                    .expect("portable revision")
                    .clone(),
                constraint: WholeWorkConstraint::AnyLegal,
                candidates: vec![candidate],
                resources,
                deterministic: true,
                require_transactional_results: false,
            })?,
        )
    };
    if !admission.liveness_lease.is_current() {
        return Err(stale_resource());
    }
    let valid_selection = match &selected {
        SelectedHostEffect::Portable(selected) => {
            &selected.decision.signature == portable_signature.as_ref().expect("portable signature")
                && selected.decision.selected.node == request.operation
                && selected.decision.selected.kind == expected_candidate_kind
                && Digest::parse_canonical(&selected.decision.selected.candidate)
                    == Some(expected_candidate_digest)
        }
        SelectedHostEffect::Managed(selected) => {
            selected.identity
                == signature.permit_identity().map_err(|source| {
                    build_runtime_error("host effect signature identity is invalid")
                        .with_identifier("RunMat:Placement:InvalidDecision")
                        .with_source(source)
                        .build()
                })?
                && selected.selected.node.as_str() == request.operation
                && selected.selected.kind == expected_candidate_kind
                && selected.selected.candidate == expected_candidate_digest
                && selected.observation_is_current()
                && stage.as_ref().is_some_and(|stage| {
                    stage.owns_managed_observation(&selected.observation_lease)
                })
        }
    };
    if !valid_selection {
        return Err(
            build_runtime_error("placement selected an unadmitted host effect route")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let (output_allocation_permit, output_headroom_to_retire) = match stage.as_ref() {
        Some(stage) => match retained_bytes {
            ResourceByteBound::Known(bytes) => {
                let output = stage.output_permit(bytes).map_err(selected_output_error)?;
                (Some(output.permit), output.headroom)
            }
            ResourceByteBound::Unavailable(_) => (None, None),
        },
        None => (admission.output_allocation_permit, None),
    };
    Ok(Some(HostEffectReceipt {
        runtime: runtime.clone(),
        selected,
        started: stage
            .as_ref()
            .map_or_else(Instant::now, |stage| stage.started),
        resource_liveness_lease: admission.liveness_lease,
        output_allocation_permit,
        output_headroom_to_retire,
        active_epoch: stage.as_ref().map_or_else(
            || Arc::new(AtomicBool::new(true)),
            HostEffectStagePermit::active_epoch,
        ),
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

fn selected_output_error(source: HostEffectAllocationError) -> crate::RuntimeError {
    let cancelled = matches!(source, HostEffectAllocationError::Cancelled);
    let identifier = match source {
        HostEffectAllocationError::Revoked | HostEffectAllocationError::InvalidInvocation => {
            "RunMat:Placement:StaleResourceAuthority"
        }
        HostEffectAllocationError::Cancelled => "RunMat:Cancelled",
        HostEffectAllocationError::ExceedsCapacity
        | HostEffectAllocationError::ExceedsStageEnvelope
        | HostEffectAllocationError::Overflow
        | HostEffectAllocationError::AllocationFailed => "RunMat:Placement:ResourceLimit",
        HostEffectAllocationError::Unavailable => "RunMat:Placement:InsufficientResourceEvidence",
    };
    let mut error = build_runtime_error("selected output headroom could not be reserved")
        .with_identifier(identifier)
        .with_source(source)
        .build();
    if cancelled {
        error.failure_origin =
            crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation;
    }
    error
}

/// Owns the exploratory probe until all effects and output publication have
/// finished. Dropping an unpolled/cancelled call releases it without feedback.
pub struct HostEffectReceipt {
    runtime: RuntimeContext,
    selected: SelectedHostEffect,
    started: Instant,
    resource_liveness_lease: Arc<dyn HostEffectResourceLivenessLease>,
    output_allocation_permit: Option<Arc<dyn runmat_value::ExactHostAllocationPermit>>,
    output_headroom_to_retire: Option<Arc<dyn HostEffectOutputHeadroom>>,
    active_epoch: Arc<AtomicBool>,
    _stage: Option<HostEffectStagePermit>,
}

enum SelectedHostEffect {
    Portable(RuntimeWholeWorkSelection),
    Managed(RuntimeManagedWholeWorkSelection),
}

impl HostEffectReceipt {
    pub fn reserve_diagnostic_claim(
        &self,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        if self.runtime.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !self.resource_liveness_lease.is_current() {
            return Err(HostEffectAllocationError::Revoked);
        }
        let claim = self
            ._stage
            .as_ref()
            .ok_or(HostEffectAllocationError::Unavailable)?
            .reserve_diagnostic_claim(bytes)?;
        if self.runtime.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !self.resource_liveness_lease.is_current() {
            return Err(HostEffectAllocationError::Revoked);
        }
        Ok(claim)
    }

    pub fn reserve_live_exact(
        &self,
        kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<Arc<dyn HostEffectLiveAllocationReservation>, HostEffectAllocationError> {
        if self.runtime.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !self.resource_liveness_lease.is_current() {
            return Err(HostEffectAllocationError::Revoked);
        }
        self._stage
            .as_ref()
            .ok_or(HostEffectAllocationError::Unavailable)?
            .reserve_live_exact(kind, bytes)
    }

    pub fn output_allocation_permit(
        &self,
    ) -> Option<Arc<dyn runmat_value::ExactHostAllocationPermit>> {
        // The staged authority already checks this receipt's epoch,
        // cancellation, session generation, and captured reservation. Reuse
        // its owner instead of allocating a second forwarding Arc per call.
        if self._stage.is_some() {
            return self.output_allocation_permit.clone();
        }
        self.output_allocation_permit.as_ref().map(|permit| {
            Arc::new(output_permit::ReceiptOutputPermit::new(
                Arc::downgrade(permit),
                self.active_epoch.clone(),
                self.runtime.cancellation(),
            )) as Arc<dyn runmat_value::ExactHostAllocationPermit>
        })
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
        // Observation is an external callback. The measured invocation is
        // already terminal, so no retained permit can admit work inside it.
        self.active_epoch.store(false, Ordering::Release);
        if let Some(headroom) = self.output_headroom_to_retire.as_ref() {
            headroom.retire_invocation();
        }
        if let Some(stage) = self._stage.as_ref() {
            stage.terminate_allocations();
        }
        if let Some(service) = self.runtime.service_ports().placement() {
            let elapsed = duration_ns_saturating(self.started.elapsed()).max(1);
            let _ = match &self.selected {
                SelectedHostEffect::Portable(selected) => {
                    service.observe_selected_whole_work(selected, elapsed, outcome)
                }
                SelectedHostEffect::Managed(selected) => {
                    service.observe_managed_whole_work(selected, elapsed, outcome)
                }
            };
        }
    }
}

impl Drop for HostEffectReceipt {
    fn drop(&mut self) {
        self.active_epoch.store(false, Ordering::Release);
        if let Some(headroom) = self.output_headroom_to_retire.as_ref() {
            headroom.retire_invocation();
        }
    }
}

#[cfg(test)]
#[path = "host_effect/tests.rs"]
mod tests;
