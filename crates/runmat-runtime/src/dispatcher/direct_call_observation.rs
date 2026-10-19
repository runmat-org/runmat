//! Request-local, opt-in complete-call feedback for admitted runtime bindings.
//! A domain certifies its own route; dispatch owns the full outer-call clock.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::workspace_effect::PostCallObservationOutcome;
use crate::builtin::registration::RuntimeBindingRegistration;
use crate::builtin::RuntimeBuiltinBinding;
use crate::context::{
    RuntimeBuiltinService, RuntimeContext, RuntimePlacementService, RuntimeWholeWorkSelection,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementExecutionAuthorityRevision,
    PlacementResourceSnapshot, PlacementSignature, ResourceBoundUnavailable, ResourceByteBound,
    WholeWorkCandidate, WholeWorkConstraint, WholeWorkObservationOutcome,
    WholeWorkPlacementRequest,
};
use runmat_time::{duration_ns_saturating, Instant};

mod prepared;
mod prepared_host;
use prepared::PreparedSelection;
pub use prepared::{PreparedBindingAuthority, PreparedCompletionRight};
pub use prepared_host::PreparedHostExecution;

/// Leaf-owned post-parse certificate for a serial, ready, host-only subroute
/// with direct host output. The implementer must have ruled out provider
/// materialization, transferred presentation, and nested timed ownership;
/// this generic candidate assumes one CPU core and unknown byte bounds.
/// Admission after parsing is not permission to label an arbitrary host
/// wrapper, asynchronous suspension, or workspace/provider effect as this
/// complete route. The observer never reconstructs builtin input domains.
pub(crate) trait HostCallDomainFacts: serde::Serialize {
    const ROUTE_ID: &'static [u8];
    fn requested_outputs(&self) -> Option<usize>;
}

#[derive(Clone)]
pub(crate) enum RegistrationAuthority {
    Global(RuntimeBindingRegistration),
    Scoped {
        service: Rc<dyn RuntimeBuiltinService>,
        generation: Digest,
    },
}

impl RegistrationAuthority {
    fn generation(&self) -> Digest {
        match self {
            Self::Global(registration) => registration.session_generation(),
            Self::Scoped { generation, .. } => *generation,
        }
    }

    fn certifies(&self, binding: RuntimeBuiltinBinding) -> bool {
        match self {
            Self::Global(registration) => registration.certifies(binding),
            Self::Scoped {
                service,
                generation,
            } => service.binding_registration_revision(binding) == Some(*generation),
        }
    }

    fn is_current(&self, binding: RuntimeBuiltinBinding) -> bool {
        match self {
            Self::Global(registration) => registration.is_current(),
            Self::Scoped {
                service,
                generation,
            } => service.binding_registration_revision(binding) == Some(*generation),
        }
    }
}

struct SelectedObservation {
    service: Rc<dyn RuntimePlacementService>,
    selected: RuntimeWholeWorkSelection,
    authority: RegistrationAuthority,
    binding: RuntimeBuiltinBinding,
}

struct State {
    started: Instant,
    selected: Option<SelectedObservation>,
    prepared: Option<PreparedSelection>,
    prepared_claimed: bool,
    host_branch: prepared_host::HostBranchState,
    invalidated: bool,
}

pub(super) struct CallObservation {
    runtime: RuntimeContext,
    state: Rc<RefCell<State>>,
}

impl CallObservation {
    pub(super) fn capture(runtime: Option<&RuntimeContext>) -> Option<Self> {
        let runtime = runtime?;
        runtime.service_ports().placement()?;
        Some(Self {
            runtime: runtime.clone(),
            state: Rc::new(RefCell::new(State {
                started: Instant::now(),
                selected: None,
                prepared: None,
                prepared_claimed: false,
                host_branch: Default::default(),
                invalidated: false,
            })),
        })
    }

    pub(super) fn sink(
        &self,
        binding: RuntimeBuiltinBinding,
        authority: RegistrationAuthority,
        requested_outputs: Option<usize>,
    ) -> Option<CallAdmissionSink> {
        if !authority.certifies(binding) || self.state.borrow().invalidated {
            return None;
        }
        Some(CallAdmissionSink {
            runtime: self.runtime.clone(),
            binding,
            authority,
            requested_outputs,
            state: Rc::downgrade(&self.state),
        })
    }

    pub(super) fn invalidate_retry(&self) {
        let mut state = self.state.borrow_mut();
        state.invalidated = true;
        state.selected = None;
        state.prepared = None;
    }

    /// Only a fully validated, workspace-reconciled call is attributable to
    /// this candidate. A finalization error drops the selection without
    /// recording a misleading execution failure.
    pub(super) fn finish_completed(&self, outcome: PostCallObservationOutcome) {
        let (started, elapsed, observation, prepared) = {
            let mut state = self.state.borrow_mut();
            if state.invalidated {
                return;
            }
            (
                state.started,
                duration_ns_saturating(state.started.elapsed()).max(1),
                state.selected.take(),
                state.prepared.take(),
            )
        };
        if self.runtime.is_cancelled() {
            return;
        }
        if let Some(observation) = observation {
            if outcome == PostCallObservationOutcome::CompletedSuccess
                && observation.authority.is_current(observation.binding)
                && observation.selected.observation_is_current()
            {
                let _ = observation.service.observe_selected_whole_work(
                    &observation.selected,
                    elapsed,
                    WholeWorkObservationOutcome::Success,
                );
            }
        }
        if let Some(prepared) = prepared {
            let status_matches = matches!(
                (prepared.completion.status(), outcome),
                (
                    crate::context::preparation::PreparedCallStatus::Succeeded,
                    PostCallObservationOutcome::CompletedSuccess
                ) | (
                    crate::context::preparation::PreparedCallStatus::CompletedFailure,
                    PostCallObservationOutcome::CompletedFailure
                )
            );
            if status_matches && prepared.authority.is_current(prepared.binding) {
                prepared.completion.publish(started, elapsed);
                // The scope's Ready drain ran before this outer fence. A
                // published asynchronous receipt belongs to this later drain.
                if let Some(service) = self.runtime.service_ports().placement() {
                    service.drain_feedback();
                }
            }
        }
    }
}

/// Non-Clone admission right for one selected canonical invocation. A dropped
/// or unpolled outer future cannot publish through its weak state handle.
pub(crate) struct CallAdmissionSink {
    runtime: RuntimeContext,
    binding: RuntimeBuiltinBinding,
    authority: RegistrationAuthority,
    requested_outputs: Option<usize>,
    state: Weak<RefCell<State>>,
}

impl CallAdmissionSink {
    pub(crate) fn admit<T: HostCallDomainFacts>(self, workload: T) {
        let Some(state) = self.state.upgrade() else {
            return;
        };
        if state.borrow().invalidated
            || state.borrow().selected.is_some()
            || state.borrow().prepared_claimed
        {
            return;
        }
        if !self.authority.certifies(self.binding) || self.runtime.is_cancelled() {
            return;
        }
        let requested_outputs = workload.requested_outputs();
        if requested_outputs != self.requested_outputs {
            return;
        }
        let Some(service) = self.runtime.service_ports().placement().cloned() else {
            return;
        };
        let parallel = self
            .runtime
            .service_ports()
            .parallel()
            .map(|port| port.placement_resources())
            .unwrap_or_default();
        // No exact peak for String clones/Vec storage is certified. A finite
        // memory policy must not turn this unknown into an admitted zero.
        if parallel.memory_available_bytes.is_some() {
            return;
        }
        let Some(revision) = service.whole_work_runtime_binding_session_revision(
            self.runtime.program_revision().cloned(),
            self.binding.identity,
            self.authority.generation(),
        ) else {
            return;
        };
        if !matches!(
            revision.authority,
            PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
        ) {
            return;
        }
        let runtime_facts =
            Digest::sha256_serialized(&(self.binding.identity, workload, T::ROUTE_ID))
                .expect("typed direct-call facts serialize");
        let signature = PlacementSignature {
            region: None,
            operation: self.binding.identity.builtin.name.to_owned(),
            runtime_facts,
            provider_provenance: None,
            graph_output_authority: None,
            revision,
        };
        let candidate = WholeWorkCandidate {
            kind: ExecutionCandidateKind::SharedRuntime,
            execution_location: CandidateExecutionLocation::Host,
            preparation: CandidatePreparationState::Ready,
            output_residency: CandidateOutputResidency::Host,
            presentation: CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256(T::ROUTE_ID),
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
            resources: CandidateResourceDemand {
                cpu_millicores: 1_000,
                retained_bytes: ResourceByteBound::Unavailable(
                    ResourceBoundUnavailable::NoRetainedValueBound,
                ),
                scratch_bytes: ResourceByteBound::Unavailable(
                    ResourceBoundUnavailable::NoScratchBound,
                ),
                provider_host_workspace: None,
                provider_input_stages: None,
                queue_slots: 0,
                peak_live: None,
            },
            transactional_results: false,
        };
        let candidate_digest = candidate.identity_digest();
        let request = WholeWorkPlacementRequest {
            signature: signature.clone(),
            constraint: WholeWorkConstraint::AnyLegal,
            candidates: vec![candidate],
            resources: PlacementResourceSnapshot {
                cpu_millicores_available: parallel.cpu_millicores_available,
                memory_available_bytes: None,
                cancellation_requested: false,
                providers: Vec::new(),
                epoch: parallel.epoch,
            },
            deterministic: true,
            require_transactional_results: false,
        };
        let Ok(selected) = service.select_whole_work(request) else {
            return;
        };
        if selected.decision.signature != signature
            || selected.decision.selected.node != signature.operation
            || selected.decision.selected.kind != ExecutionCandidateKind::SharedRuntime
            || selected.decision.selected.candidate != candidate_digest.to_string()
            || !selected.observation_is_current()
            || !self.authority.is_current(self.binding)
        {
            return;
        }
        let mut state = state.borrow_mut();
        if !state.invalidated && state.selected.is_none() && !state.prepared_claimed {
            state.selected = Some(SelectedObservation {
                service,
                selected,
                authority: self.authority,
                binding: self.binding,
            });
        }
    }
}

#[cfg(test)]
mod tests;
