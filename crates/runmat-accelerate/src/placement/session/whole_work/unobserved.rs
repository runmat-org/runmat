//! Context-free entry to the same whole-work choice, without learning rights.
use super::*;

/// The same provider-only policy for callers without an observation session.
/// Missing prices never authorize exploration or a synthetic numeric graph.
/// Resource legality is independent of observation/session authority.
pub(crate) fn select_unobserved(
    operation: &str,
    candidates: &[WholeWorkCandidate],
    constraint: WholeWorkConstraint,
    resources: &runmat_execution::PlacementResourceSnapshot,
    require_transactional_results: bool,
) -> Result<
    (
        runmat_execution::SelectedExecutionCandidate,
        WholeWorkDecisionKind,
        Option<u64>,
    ),
    PlacementPlanError,
> {
    if candidates.is_empty() {
        return Err(no_legal_candidate());
    }
    constraint
        .validate_candidates(candidates, resources)
        .map_err(invalid_request)?;
    if resources.cancellation_requested {
        return Err(PlacementPlanError {
            code: "RunMat:Placement:Cancelled",
            message: "placement cancelled before dispatch".into(),
        });
    }
    let operation = PlacementOperationId::new(operation).map_err(invalid_request)?;
    let mut insufficient_resource_evidence = false;
    let admitted = candidates
        .iter()
        .filter(|candidate| {
            match candidate_resources_admitted(candidate, resources, require_transactional_results)
            {
                Ok(()) => true,
                Err(reason) => {
                    insufficient_resource_evidence |=
                        matches!(reason, ResourceAdmissionError::InsufficientEvidence { .. });
                    false
                }
            }
        })
        .map(|candidate| Admitted {
            candidate,
            identity: candidate.identity_digest(),
            cost: candidate.cost,
            observation: CandidateObservationEvidence::Unavailable(
                CandidateCostUnavailable::NoObservations,
            ),
        })
        .collect::<Vec<_>>();
    if admitted.is_empty() {
        return Err(no_resource_candidate(insufficient_resource_evidence));
    }
    let draft = if admitted
        .iter()
        .all(|item| item.candidate.kind.is_provider())
    {
        let item = route_choice(&admitted, true).ok_or_else(no_legal_candidate)?;
        SelectionDraft::Direct {
            selected: CompactSelectedExecutionCandidate {
                node: operation,
                candidate: item.identity,
                kind: item.candidate.kind,
            },
            kind: if matches!(constraint, WholeWorkConstraint::ProviderRequired { .. }) {
                WholeWorkDecisionKind::ProviderRequired
            } else {
                WholeWorkDecisionKind::Uncompared
            },
            predicted_total_ns: item
                .cost
                .estimate()
                .and_then(|cost| cost.checked_total_ns()),
            probe: None,
        }
    } else {
        host_selection(
            &admitted,
            operation,
            if admitted.len() == 1 && admitted[0].cost.estimate().is_some() {
                WholeWorkDecisionKind::Uncompared
            } else {
                WholeWorkDecisionKind::Unmeasured
            },
        )?
    };
    let SelectionDraft::Direct {
        selected,
        kind,
        predicted_total_ns,
        ..
    } = draft
    else {
        unreachable!("unobserved missing-price policy never creates a numeric graph")
    };
    Ok((selected.portable(), kind, predicted_total_ns))
}
