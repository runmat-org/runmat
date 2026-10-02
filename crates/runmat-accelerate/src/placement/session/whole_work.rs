//! One complete invocation enters shared placement only after route and
//! resource admission. Unknown durations are learned by executing a real call,
//! never by replaying effects or inserting a synthetic graph cost.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateObservationEvidence, Digest,
    ExecutionCandidateDescriptor, ExecutionCandidateKind, PlacementGraph, PlacementGraphCandidate,
    PlacementGraphLimits, PlacementGraphNode, PlacementPlanRequest, ResourceByteBound,
    SelectedExecutionCandidate, WholeWorkCandidate, WholeWorkConstraint, WholeWorkDecision,
    WholeWorkDecisionKind, WholeWorkPlacementRequest,
};
use runmat_runtime::context::{RuntimeWholeWorkProbeLease, RuntimeWholeWorkSelection};

use super::{PlacementPlanError, PlacementSession};
use crate::placement::partition::{single_candidate_resources_admitted, ResourceAdmissionError};

type ProbeKey = (Digest, String, ExecutionCandidateKind);

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
    selected: SelectedExecutionCandidate,
    cost: CandidateCostEvidence,
    observation: CandidateObservationEvidence,
}

impl PlacementSession {
    pub fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, PlacementPlanError> {
        request.validate().map_err(invalid_request)?;
        if request.resources.cancellation_requested {
            return Err(PlacementPlanError {
                code: "RunMat:Placement:Cancelled",
                message: "placement cancelled before dispatch".into(),
            });
        }
        self.drain_observations();
        let mut admitted = Vec::new();
        let mut insufficient_resource_evidence = false;
        for candidate in &request.candidates {
            let demand = candidate.resources;
            if let Err(reason) = single_candidate_resources_admitted(
                candidate.execution_location,
                candidate.output_residency,
                candidate.presentation,
                candidate.kind,
                demand,
                candidate.transactional_results,
                &request.resources,
                request.require_transactional_results,
            ) {
                insufficient_resource_evidence |=
                    matches!(reason, ResourceAdmissionError::InsufficientEvidence { .. });
                continue;
            }
            let selected = candidate.selected(&request.signature.operation);
            let observed = self.observed_candidate_duration(
                &request.signature,
                &selected.candidate,
                selected.kind,
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
                selected,
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
                    CandidateCostEvidence::Estimated(estimate) => (
                        estimate.checked_risk_adjusted_ns(),
                        &item.selected.candidate,
                    ),
                    CandidateCostEvidence::Unavailable(_) => unreachable!(),
                })
                .or_else(|| admitted.first())
                .expect("nonempty admitted whole-work candidates");
            return Ok(RuntimeWholeWorkSelection {
                decision: WholeWorkDecision {
                    signature: request.signature,
                    selected: item.selected.clone(),
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
                },
                probe: None,
            });
        }

        if matches!(request.constraint, WholeWorkConstraint::AnyLegal) {
            if let Some(index) = exploration_candidate(&admitted, &self.whole_work_probes, &request)
            {
                let item = &admitted[index];
                let key = (
                    request.signature.cache_key(),
                    item.selected.candidate.clone(),
                    item.selected.kind,
                );
                lock_probes(&self.whole_work_probes).insert(key.clone());
                return Ok(RuntimeWholeWorkSelection {
                    decision: WholeWorkDecision {
                        signature: request.signature,
                        selected: item.selected.clone(),
                        kind: WholeWorkDecisionKind::Exploratory,
                        predicted_total_ns: None,
                    },
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
            return unmeasured_host(&request, &admitted);
        }
        // A measured host route is required before the ordinary numeric graph
        // can compare it with provider candidates. Unmeasured host routes are
        // handled by the bounded single-invocation exploration above.
        if !estimated
            .iter()
            .any(|(item, _)| !item.candidate.kind.is_provider())
        {
            return unmeasured_host(&request, &admitted);
        }
        let node = PlacementGraphNode {
            identity: request.signature.operation.clone(),
            candidates: estimated
                .into_iter()
                .map(|(item, cost)| PlacementGraphCandidate {
                    descriptor: ExecutionCandidateDescriptor {
                        identity: item.selected.candidate.clone(),
                        region: request.signature.region,
                        kind: item.selected.kind,
                        execution_location: item.candidate.execution_location,
                        preparation: item.candidate.preparation,
                        cost,
                        output_residency: item.candidate.output_residency,
                        presentation: item.candidate.presentation,
                        guards: Vec::new(),
                    },
                    resources: item.candidate.resources,
                    transactional_results: item.candidate.transactional_results,
                })
                .collect(),
        };
        let plan = self.plan(PlacementPlanRequest {
            signature: request.signature,
            graph: PlacementGraph {
                nodes: vec![node],
                edges: Vec::new(),
            },
            limits: PlacementGraphLimits::default(),
            resources: request.resources,
            deterministic: request.deterministic,
            require_transactional_results: request.require_transactional_results,
        })?;
        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: plan.signature,
                selected: plan.selections.into_iter().next().expect("one-node graph"),
                kind: WholeWorkDecisionKind::Compared,
                predicted_total_ns: Some(plan.predicted_total_ns),
            },
            probe: None,
        })
    }
}

fn unmeasured_host(
    request: &WholeWorkPlacementRequest,
    admitted: &[Admitted<'_>],
) -> Result<RuntimeWholeWorkSelection, PlacementPlanError> {
    let item = admitted
        .iter()
        .find(|item| !item.candidate.kind.is_provider())
        .ok_or_else(no_legal_candidate)?;
    Ok(RuntimeWholeWorkSelection {
        decision: WholeWorkDecision {
            signature: request.signature.clone(),
            selected: item.selected.clone(),
            kind: WholeWorkDecisionKind::Unmeasured,
            predicted_total_ns: None,
        },
        probe: None,
    })
}

fn exploration_candidate(
    admitted: &[Admitted<'_>],
    active: &Arc<Mutex<BTreeSet<ProbeKey>>>,
    request: &WholeWorkPlacementRequest,
) -> Option<usize> {
    let active = lock_probes(active);
    let signature = request.signature.cache_key();
    let eligible = |item: &Admitted<'_>| {
        matches!(item.cost, CandidateCostEvidence::Unavailable(_))
            && (!item.candidate.kind.is_provider()
                || !request.require_transactional_results
                || item.candidate.transactional_results)
            && !active.contains(&(
                signature,
                item.selected.candidate.clone(),
                item.selected.kind,
            ))
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
