//! Preserve unavailable quotes until the existing whole-work policy selects.
//! No numeric descriptor, runtime service, or observation right is invented.
use super::*;
use decision::DecisionRequest;
use runmat_execution::{WholeWorkCandidate, WholeWorkPlacementRequestRef};

pub(super) fn select(
    request: DecisionRequest<'_>,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    let DecisionRequest {
        constraint,
        provider,
        operation,
        semantic_identity,
        observation_contract,
        prepared_authority,
        vm_right,
        cpu,
        cpu_resources,
        cpu_cost,
        mut admitted,
        rejected,
        source_digest,
        residency,
        physical_facts,
        compare_profitability,
        runtime,
        policy,
    } = request;
    admitted.sort_by(|left, right| left.identity.cmp(&right.identity));
    if admitted
        .windows(2)
        .any(|pair| pair[0].identity == pair[1].identity)
    {
        return Err(Box::new(
            runmat_runtime::build_runtime_error("duplicate physical candidate identity")
                .with_identifier("RunMat:Placement:InvalidRequest")
                .build(),
        ));
    }
    // Snapshot admission is mandatory even without a service, a FullCall
    // revision, or an observation permit. These are physical facts only.
    let (resources, digest, host_epoch, transactional) = physical_facts;
    let mut candidates = Vec::with_capacity(admitted.len() + 1);
    if compare_profitability {
        if let Some(resources) = cpu_resources {
            let mut host = cpu.route.clone();
            host.resources = resources;
            candidates.push(host);
        }
    }
    let provider_offset = candidates.len();
    for entry in &admitted {
        candidates.push(WholeWorkCandidate {
            kind: entry.kind,
            execution_location: CandidateExecutionLocation::Provider {
                device_id: provider.device_id(),
            },
            preparation: entry.cost_query.preparation,
            output_residency: CandidateOutputResidency::Provider {
                device_id: provider.device_id(),
            },
            presentation: runmat_execution::CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256_serialized(&(&entry.identity, &entry.cost_query))
                .expect("physical route facts serialize"),
            cost: entry.cost,
            resources: candidate_resource(entry),
            transactional_results: transactional,
        });
    }
    // Match the numeric selector's price domain before whole-set validation.
    // A malformed estimate is not unavailable evidence and cannot invalidate
    // otherwise legal routes. Preserve the original sequence for receipts.
    let original_candidates = candidates;
    let candidates = original_candidates
        .iter()
        .filter(|candidate| {
            candidate
                .cost
                .estimate()
                .is_none_or(|estimate| estimate.checked_risk_adjusted_ns().is_some())
        })
        .cloned()
        .collect::<Vec<_>>();
    // Apply the same physical owner before either local policy or service
    // selection. Keep the original sequence for receipt/index projection.
    if !candidates.is_empty() {
        constraint
            .validate_candidates(&candidates, &resources)
            .map_err(|message| {
                plan_error(crate::placement::PlacementPlanError {
                    code: "RunMat:Placement:InvalidRequest",
                    message: message.into(),
                })
            })?;
    }
    if resources.cancellation_requested {
        return Err(plan_error(crate::placement::PlacementPlanError {
            code: "RunMat:Placement:Cancelled",
            message: "placement cancelled before dispatch".into(),
        }));
    }
    let mut insufficient = false;
    let candidates = candidates.into_iter().filter(|candidate| {
        match crate::placement::session::candidate_resources_admitted(candidate, &resources, true) {
            Ok(()) => true,
            Err(reason) => {
                insufficient |= matches!(reason, crate::placement::partition::ResourceAdmissionError::InsufficientEvidence { .. });
                false
            }
        }
    }).collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(plan_error(
            crate::placement::session::no_resource_candidate(insufficient),
        ));
    }
    // Only the installed complete-call right can admit learning. A partial
    // legacy observation is not complete-route whole-work evidence.
    let authority = runtime
        .filter(|_| !candidates.is_empty())
        .filter(|_| {
            matches!(
                observation_contract,
                TotalCandidateObservationContract::FullCall
            )
        })
        .filter(|runtime| runtime.service_ports().placement().is_some())
        .and_then(|runtime| {
            let revision = observation_revision(
                runtime,
                observation_contract,
                prepared_authority,
                vm_right,
                digest,
                policy,
            )?;
            let facts = signature::candidate_set_facts(
                &admitted
                    .iter()
                    .map(|entry| (&entry.identity, &entry.cost_query, entry.output_bytes))
                    .collect::<Vec<_>>(),
                &rejected,
                source_digest,
                cpu.route.preparation,
                &residency,
                &resources,
                host_epoch,
            );
            let signature = PlacementSignature {
                region: cpu.region,
                operation: operation.into(),
                runtime_facts: signature::with_total_candidate_observation_contract(
                    signature::with_semantic_identity(
                        facts,
                        observation_semantic_identity(
                            observation_contract,
                            semantic_identity,
                            digest,
                        ),
                    ),
                    observation_contract,
                ),
                provider_provenance: super::super::session_provider_provenance(&revision, digest),
                graph_output_authority: None,
                revision,
            };
            Some((runtime, signature))
        });
    let mut feedback = None;
    let (selected, decision_kind, predicted_total_ns) =
        if let Some((runtime, signature)) = authority {
            let service = runtime
                .service_ports()
                .placement()
                .expect("revision requires selected placement service");
            let selection = service
                .select_whole_work_borrowed(WholeWorkPlacementRequestRef {
                    signature: (&signature).into(),
                    constraint,
                    candidates: &candidates,
                    resources: &resources,
                    deterministic: false,
                    require_transactional_results: true,
                })
                .map_err(Box::new)?;
            if selection.decision.signature != signature {
                return Err(Box::new(
                    runmat_runtime::build_runtime_error(
                        "placement returned a different whole-work signature",
                    )
                    .with_identifier("RunMat:Placement:InvalidDecision")
                    .build(),
                ));
            }
            let decision = &selection.decision;
            let result = (
                decision.selected.clone(),
                decision.kind,
                decision.predicted_total_ns,
            );
            if selection.observation_permit.is_some() && selection.observation_is_current() {
                feedback = Some(Box::new(LocalPlacementFeedback {
                    runtime: runtime.clone(),
                    signature: decision.signature.clone(),
                    candidate: decision.selected.candidate.clone(),
                    kind: decision.selected.kind,
                    authority: LocalPlacementFeedbackAuthority::WholeWork(std::sync::Arc::new(
                        std::sync::Mutex::new(Some(selection)),
                    )),
                }));
            }
            result
        } else {
            crate::placement::session::select_unobserved(
                operation,
                &candidates,
                constraint,
                &resources,
                true,
            )
            .map_err(plan_error)?
        };
    if !candidates
        .iter()
        .any(|candidate| candidate.selected(operation) == selected)
    {
        return Err(Box::new(
            runmat_runtime::build_runtime_error(
                "placement selected a resource-ineligible physical candidate",
            )
            .with_identifier("RunMat:Placement:InvalidDecision")
            .build(),
        ));
    }
    let ordinal = original_candidates
        .iter()
        .position(|candidate| candidate.selected(operation) == selected)
        .ok_or_else(|| {
            Box::new(
                runmat_runtime::build_runtime_error(
                    "placement selected an unknown physical candidate",
                )
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
            )
        })?;
    let chosen = ordinal
        .checked_sub(provider_offset)
        .and_then(|index| admitted.get(index));
    let receipt = chosen.or_else(|| admitted.first());
    let provider_query = receipt
        .map(|entry| &entry.cost_query.operation)
        .or_else(|| rejected.first().map(|(_, query, _)| query))
        .expect("original physical record has an admitted or rejected receipt");
    Ok(LocalPlacementOutcome::Selected {
        selected: Box::new(selected),
        cpu_cost,
        provider_cost: receipt.and_then(|entry| entry.cost.estimate()),
        provider_rejection: admitted
            .is_empty()
            .then(|| rejected.first().map(|(_, _, code)| *code))
            .flatten(),
        decision_kind,
        predicted_total_ns,
        provider_observation: feedback.is_some()
            && chosen.is_some_and(|entry| entry.cost_query.context.observation.is_some()),
        feedback,
        provider_query: Box::new(provider_query.clone()),
        provider_index: chosen.map(|entry| entry.original_index),
    })
}
