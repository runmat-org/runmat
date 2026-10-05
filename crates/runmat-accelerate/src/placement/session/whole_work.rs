//! One complete invocation enters shared placement only after route and
//! resource admission. Unknown durations are learned by executing a real call,
//! never by replaying effects or inserting a synthetic graph cost.

use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateObservationEvidence,
    CompactSelectedExecutionCandidate, Digest, ExecutionCandidateDescriptor,
    ExecutionCandidateKind, PlacementGraph, PlacementGraphCandidate, PlacementGraphLimits,
    PlacementGraphNode, PlacementOperationId, PlacementPlanRequest, ResourceByteBound,
    WholeWorkCandidate, WholeWorkConstraint, WholeWorkDecision, WholeWorkDecisionKind,
    WholeWorkPlacementRequest, WholeWorkPlacementRequestRef,
};
use runmat_runtime::context::{
    RuntimeManagedWholeWorkSelection, RuntimeWholeWorkProbeLease, RuntimeWholeWorkSelection,
};

use super::{PlacementPlanError, PlacementSession};
use crate::placement::partition::{
    single_candidate_resources_admitted, CandidateResourceFacts, ResourceAdmissionError,
};

pub(super) type ProbeKey = (Digest, Digest, ExecutionCandidateKind);

struct ProbeLease {
    active: Arc<Mutex<BTreeSet<ProbeKey>>>,
    key: ProbeKey,
}

impl RuntimeWholeWorkProbeLease for ProbeLease {}

impl Drop for ProbeLease {
    fn drop(&mut self) {
        lock_probes(&self.active).remove(&self.key);
    }
}

fn lock_probes(
    probes: &Mutex<BTreeSet<ProbeKey>>,
) -> std::sync::MutexGuard<'_, BTreeSet<ProbeKey>> {
    probes
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Admitted<'a> {
    candidate: &'a WholeWorkCandidate,
    identity: Digest,
    cost: CandidateCostEvidence,
    observation: CandidateObservationEvidence,
}

enum SelectionDraft {
    Direct {
        selected: CompactSelectedExecutionCandidate,
        kind: WholeWorkDecisionKind,
        predicted_total_ns: Option<u64>,
        probe: Option<Box<dyn RuntimeWholeWorkProbeLease>>,
    },
    /// A compared route waits for the caller to transfer its original
    /// signature into the graph planner. A borrowed caller clones only here.
    Compared { graph: PlacementGraph },
}

impl PlacementSession {
    pub fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, PlacementPlanError> {
        let draft = self.select_whole_work_untracked(request.as_ref(), false)?;
        self.finish_whole_work_selection(
            draft,
            request.signature,
            || request.resources,
            request.deterministic,
            request.require_transactional_results,
        )
    }

    pub fn select_whole_work_borrowed(
        &self,
        request: WholeWorkPlacementRequestRef<'_>,
    ) -> Result<RuntimeWholeWorkSelection, PlacementPlanError> {
        let draft = self.select_whole_work_untracked(request, false)?;
        self.finish_whole_work_selection(
            draft,
            (*request.signature).clone(),
            || (*request.resources).clone(),
            request.deterministic,
            request.require_transactional_results,
        )
    }

    pub fn select_whole_work_borrowed_managed(
        &self,
        request: WholeWorkPlacementRequestRef<'_>,
    ) -> Result<RuntimeManagedWholeWorkSelection, PlacementPlanError> {
        if request.candidates.len() != 1 {
            return Err(PlacementPlanError {
                code: "RunMat:Placement:InsufficientResourceEvidence",
                message: "managed single-route placement cannot retain a compared graph".into(),
            });
        }
        if self.managed_observations.is_none() {
            return Err(PlacementPlanError {
                code: "RunMat:Placement:InsufficientResourceEvidence",
                message: "managed observation backing is not installed".into(),
            });
        }
        let SelectionDraft::Direct {
            selected,
            kind,
            predicted_total_ns,
            probe: None,
        } = self.select_whole_work_untracked(request, true)?
        else {
            unreachable!("a managed one-route policy has no graph or dynamic probe")
        };
        let identity =
            request
                .signature
                .permit_identity()
                .map_err(|source| PlacementPlanError {
                    code: "RunMat:Placement:InvalidRequest",
                    message: source.to_string(),
                })?;
        let seed = self
            .state
            .borrow()
            .feedback
            .get(&(
                identity.signature,
                super::FeedbackCandidateKey::Canonical(selected.candidate),
                selected.kind,
            ))
            .map(super::evidence::FeedbackNumbers::from_series);
        let lease = self
            .managed_observations
            .as_ref()
            .expect("checked managed owner")
            .acquire(identity, selected.candidate)
            .map_err(|source| PlacementPlanError {
                code: "RunMat:Placement:InsufficientResourceEvidence",
                message: source.to_string(),
            })?;
        self.managed_feedback
            .as_ref()
            .expect("checked managed feedback owner")
            .ensure(request.signature, selected.candidate, selected.kind, seed)
            .map_err(|source| PlacementPlanError {
                code: "RunMat:Placement:InsufficientResourceEvidence",
                message: source.to_string(),
            })?;
        Ok(RuntimeManagedWholeWorkSelection {
            identity,
            selected,
            kind,
            predicted_total_ns,
            observation_lease: lease,
        })
    }

    fn finish_whole_work_selection(
        &self,
        draft: SelectionDraft,
        signature: runmat_execution::PlacementSignature,
        resources: impl FnOnce() -> runmat_execution::PlacementResourceSnapshot,
        deterministic: bool,
        require_transactional_results: bool,
    ) -> Result<RuntimeWholeWorkSelection, PlacementPlanError> {
        let (signature, selected_candidate, kind, predicted_total_ns, probe) = match draft {
            SelectionDraft::Compared { graph } => {
                let plan = self.plan(PlacementPlanRequest {
                    signature,
                    graph,
                    limits: PlacementGraphLimits::default(),
                    resources: resources(),
                    deterministic,
                    require_transactional_results,
                })?;
                (
                    plan.signature,
                    plan.selections.into_iter().next().expect("one-node graph"),
                    WholeWorkDecisionKind::Compared,
                    Some(plan.predicted_total_ns),
                    None,
                )
            }
            SelectionDraft::Direct {
                selected,
                kind,
                predicted_total_ns,
                probe,
            } => (
                signature,
                selected.portable(),
                kind,
                predicted_total_ns,
                probe,
            ),
        };
        let mut selected = RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature,
                selected: selected_candidate,
                kind,
                predicted_total_ns,
            },
            probe,
            observation_permit: None,
        };
        selected.observation_permit = Some(
            runmat_runtime::context::RuntimeWholeWorkObservationAuthority::Dynamic(
                self.observation_permits
                    .register(&selected.decision.signature)
                    .map_err(|source| PlacementPlanError {
                        code: "RunMat:Placement:InvalidRequest",
                        message: source.to_string(),
                    })?,
            ),
        );
        Ok(selected)
    }

    fn select_whole_work_untracked(
        &self,
        request: WholeWorkPlacementRequestRef<'_>,
        managed: bool,
    ) -> Result<SelectionDraft, PlacementPlanError> {
        request.validate().map_err(invalid_request)?;
        let operation =
            PlacementOperationId::new(&request.signature.operation).map_err(invalid_request)?;
        if request.resources.cancellation_requested {
            return Err(PlacementPlanError {
                code: "RunMat:Placement:Cancelled",
                message: "placement cancelled before dispatch".into(),
            });
        }
        self.drain_observations();
        let mut admitted: SmallVec<[Admitted<'_>; 1]> = SmallVec::new();
        let mut insufficient_resource_evidence = false;
        for candidate in request.candidates {
            let demand = candidate.resources;
            if let Err(reason) = single_candidate_resources_admitted(
                CandidateResourceFacts {
                    location: candidate.execution_location,
                    output_residency: candidate.output_residency,
                    presentation: &candidate.presentation,
                    kind: candidate.kind,
                    demand,
                    transactional_results: candidate.transactional_results,
                },
                request.resources,
                request.require_transactional_results,
            ) {
                insufficient_resource_evidence |=
                    matches!(reason, ResourceAdmissionError::InsufficientEvidence { .. });
                continue;
            }
            let identity = candidate.identity_digest();
            let observed = self.observed_candidate_duration_digest(
                request.signature,
                identity,
                candidate.kind,
            );
            if matches!(
                observed,
                CandidateObservationEvidence::Unavailable(
                    CandidateCostUnavailable::RepeatedFailures { .. }
                )
            ) && candidate.kind.is_provider()
            {
                continue;
            }
            let cost = match observed {
                CandidateObservationEvidence::Observed(_) => {
                    let scratch_bytes = match demand.scratch_bytes {
                        ResourceByteBound::Known(bytes) => Some(bytes),
                        ResourceByteBound::Unavailable(_) => None,
                    };
                    CandidateCostEvidence::from_observation(observed, scratch_bytes)
                }
                CandidateObservationEvidence::Unavailable(_) => candidate.cost,
            };
            let cost = match cost {
                CandidateCostEvidence::Estimated(estimate)
                    if estimate.checked_risk_adjusted_ns().is_none() =>
                {
                    CandidateCostEvidence::Unavailable(
                        CandidateCostUnavailable::UnrepresentableEstimate,
                    )
                }
                other => other,
            };
            admitted.push(Admitted {
                candidate,
                identity,
                cost,
                observation: observed,
            });
        }
        if admitted.is_empty() {
            return Err(if insufficient_resource_evidence {
                PlacementPlanError {
                    code: "RunMat:Placement:InsufficientResourceEvidence",
                    message: "a resource budget requires a matching route memory bound".into(),
                }
            } else {
                no_legal_candidate()
            });
        }

        if managed {
            // A single admitted route has no comparison graph. Keep the same
            // validated candidate and observation cost; an occupied managed
            // slot suppresses a second exploratory probe without replay.
            let item = &admitted[0];
            let active = self.managed_observations.as_ref().is_some_and(|pool| {
                request
                    .signature
                    .permit_identity()
                    .ok()
                    .is_some_and(|identity| pool.has_active(identity, item.identity))
            });
            let kind = if matches!(
                request.constraint,
                WholeWorkConstraint::ProviderRequired { .. }
            ) {
                WholeWorkDecisionKind::ProviderRequired
            } else if matches!(item.cost, CandidateCostEvidence::Unavailable(_)) {
                if active {
                    WholeWorkDecisionKind::Unmeasured
                } else {
                    WholeWorkDecisionKind::Exploratory
                }
            } else {
                WholeWorkDecisionKind::Uncompared
            };
            return Ok(SelectionDraft::Direct {
                selected: CompactSelectedExecutionCandidate {
                    node: operation,
                    candidate: item.identity,
                    kind: item.candidate.kind,
                },
                kind,
                predicted_total_ns: item
                    .cost
                    .estimate()
                    .and_then(|cost| cost.checked_total_ns()),
                probe: None,
            });
        }

        let provider_only = admitted
            .iter()
            .all(|item| item.candidate.kind.is_provider());
        if matches!(
            request.constraint,
            WholeWorkConstraint::ProviderRequired { .. }
        ) || provider_only
        {
            let item = admitted
                .iter()
                .filter(|item| matches!(item.cost, CandidateCostEvidence::Estimated(_)))
                .min_by_key(|item| match item.cost {
                    CandidateCostEvidence::Estimated(estimate) => {
                        (estimate.checked_risk_adjusted_ns(), &item.identity)
                    }
                    CandidateCostEvidence::Unavailable(_) => unreachable!(),
                })
                .or_else(|| admitted.first())
                .expect("nonempty admitted whole-work candidates");
            return Ok(SelectionDraft::Direct {
                selected: CompactSelectedExecutionCandidate {
                    node: operation,
                    candidate: item.identity,
                    kind: item.candidate.kind,
                },
                kind: if matches!(
                    request.constraint,
                    WholeWorkConstraint::ProviderRequired { .. }
                ) {
                    WholeWorkDecisionKind::ProviderRequired
                } else {
                    WholeWorkDecisionKind::Uncompared
                },
                predicted_total_ns: item
                    .cost
                    .estimate()
                    .and_then(|cost| cost.checked_total_ns()),
                probe: None,
            });
        }

        if matches!(request.constraint, WholeWorkConstraint::AnyLegal) {
            if let Some(index) = exploration_candidate(&admitted, &self.whole_work_probes, request)
            {
                let item = &admitted[index];
                let key = (
                    request.signature.cache_key(),
                    item.identity,
                    item.candidate.kind,
                );
                lock_probes(&self.whole_work_probes).insert(key);
                return Ok(SelectionDraft::Direct {
                    selected: CompactSelectedExecutionCandidate {
                        node: operation,
                        candidate: item.identity,
                        kind: item.candidate.kind,
                    },
                    kind: WholeWorkDecisionKind::Exploratory,
                    predicted_total_ns: None,
                    probe: Some(Box::new(ProbeLease {
                        active: Arc::clone(&self.whole_work_probes),
                        key,
                    })),
                });
            }
        }

        let estimated: Vec<_> = admitted
            .iter()
            .filter_map(|item| match item.cost {
                CandidateCostEvidence::Estimated(cost) => Some((item, cost)),
                CandidateCostEvidence::Unavailable(_) => None,
            })
            .collect();
        if estimated.is_empty() {
            return unmeasured_host(&admitted, operation);
        }
        // A measured host route is required before the ordinary numeric graph
        // can compare it with provider candidates. Unmeasured host routes are
        // handled by the bounded single-invocation exploration above.
        if !estimated
            .iter()
            .any(|(item, _)| !item.candidate.kind.is_provider())
        {
            return unmeasured_host(&admitted, operation);
        }
        let node = PlacementGraphNode {
            identity: request.signature.operation.clone(),
            candidates: estimated
                .into_iter()
                .map(|(item, cost)| PlacementGraphCandidate {
                    descriptor: ExecutionCandidateDescriptor {
                        identity: item.identity.to_string(),
                        region: request.signature.region,
                        kind: item.candidate.kind,
                        execution_location: item.candidate.execution_location,
                        preparation: item.candidate.preparation,
                        cost,
                        output_residency: item.candidate.output_residency,
                        presentation: item.candidate.presentation.clone(),
                        guards: Vec::new(),
                    },
                    resources: item.candidate.resources,
                    transactional_results: item.candidate.transactional_results,
                })
                .collect(),
        };
        Ok(SelectionDraft::Compared {
            graph: PlacementGraph {
                nodes: vec![node],
                edges: Vec::new(),
            },
        })
    }
}

fn unmeasured_host(
    admitted: &[Admitted<'_>],
    operation: PlacementOperationId,
) -> Result<SelectionDraft, PlacementPlanError> {
    let item = admitted
        .iter()
        .find(|item| !item.candidate.kind.is_provider())
        .ok_or_else(no_legal_candidate)?;
    Ok(SelectionDraft::Direct {
        selected: CompactSelectedExecutionCandidate {
            node: operation,
            candidate: item.identity,
            kind: item.candidate.kind,
        },
        kind: WholeWorkDecisionKind::Unmeasured,
        predicted_total_ns: None,
        probe: None,
    })
}

fn exploration_candidate(
    admitted: &[Admitted<'_>],
    active: &Arc<Mutex<BTreeSet<ProbeKey>>>,
    request: WholeWorkPlacementRequestRef<'_>,
) -> Option<usize> {
    let active = lock_probes(active);
    let signature = request.signature.cache_key();
    let eligible = |item: &Admitted<'_>| {
        matches!(item.cost, CandidateCostEvidence::Unavailable(_))
            && (!item.candidate.kind.is_provider()
                || !request.require_transactional_results
                || item.candidate.transactional_results)
            && !active.contains(&(signature, item.identity, item.candidate.kind))
    };
    let host = admitted
        .iter()
        .position(|item| !item.candidate.kind.is_provider() && eligible(item));
    let provider = admitted
        .iter()
        .position(|item| item.candidate.kind.is_provider() && eligible(item));
    // Establish a legal host baseline, then admit an unmeasured provider. If
    // no host route exists, provider execution is not compared to fake CPU work.
    if admitted.iter().any(|item| {
        !item.candidate.kind.is_provider()
            && matches!(item.cost, CandidateCostEvidence::Unavailable(_))
    }) {
        let failed_host = admitted.iter().any(|item| {
            !item.candidate.kind.is_provider()
                && matches!(
                    item.observation,
                    CandidateObservationEvidence::Unavailable(
                        CandidateCostUnavailable::InsufficientObservations { failed: 2.., .. }
                    )
                )
        });
        if failed_host {
            provider.or(host)
        } else {
            host.or(provider)
        }
    } else {
        provider.or(host)
    }
}

fn invalid_request(message: &'static str) -> PlacementPlanError {
    PlacementPlanError {
        code: "RunMat:Placement:InvalidRequest",
        message: message.into(),
    }
}

fn no_legal_candidate() -> PlacementPlanError {
    PlacementPlanError {
        code: "RunMat:Placement:NoLegalCandidate",
        message: "no complete route has admissible resource evidence".into(),
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
#[path = "whole_work/host_effect_tests.rs"]
mod host_effect_tests;
