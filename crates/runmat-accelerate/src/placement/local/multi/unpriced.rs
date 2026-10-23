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
        vm_decline,
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
        publication,
        policy,
        bounded_host,
    } = request;
    let require_transactional_results = publication == LocalPublicationSemantics::Atomic;
    admitted.sort_unstable_by(|left, right| left.identity.cmp(&right.identity));
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
    let mut candidates = Vec::new();
    candidates
        .try_reserve_exact(admitted.len().checked_add(3).ok_or_else(|| {
            plan_error(crate::placement::PlacementPlanError {
                code: "RunMat:Placement:ResourceOverflow",
                message: "physical candidate count overflows".into(),
            })
        })?)
        .map_err(|_| {
            plan_error(crate::placement::PlacementPlanError {
                code: "RunMat:Placement:ResourceOverflow",
                message: "physical candidate allocation failed".into(),
            })
        })?;
    if compare_profitability {
        if let Some(resources) = cpu_resources {
            let mut host = cpu.route.clone();
            host.resources = resources;
            candidates.push(host);
        }
        let legal_location = |location| match constraint {
            runmat_execution::WholeWorkConstraint::AnyLegal => true,
            runmat_execution::WholeWorkConstraint::ProviderRequired { device_id } => matches!(
                location,
                CandidateExecutionLocation::Provider { device_id: selected }
                    | CandidateExecutionLocation::NativeRegion { device_id: selected }
                    | CandidateExecutionLocation::VmRegion { device_id: selected }
                    if selected == device_id
            ),
        };
        if let Some(alternative) = cpu
            .native_alternative
            .as_ref()
            .filter(|alternative| legal_location(alternative.descriptor.execution_location))
        {
            candidates.push(WholeWorkCandidate {
                kind: alternative.descriptor.kind,
                execution_location: alternative.descriptor.execution_location,
                preparation: alternative.descriptor.preparation,
                output_residency: alternative.descriptor.output_residency,
                presentation: alternative.descriptor.presentation.clone(),
                route_facts: Digest::sha256_serialized(&(
                    "native-host-alternative-v2",
                    &alternative.descriptor,
                ))
                .expect("native alternative serializes"),
                cost: runmat_execution::CandidateCostEvidence::Estimated(
                    alternative.descriptor.cost,
                ),
                resources: alternative.resources,
                transactional_results: alternative.transactional_results,
            });
        }
        if let Some(alternative) = cpu
            .native_host_alternative
            .as_ref()
            .filter(|alternative| legal_location(alternative.execution_location))
        {
            // Preserve unavailable complete-route evidence. Only this same
            // selected full-call owner may replace it with an observation.
            candidates.push(alternative.clone());
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
            output_residency: entry
                .exact_residency
                .unwrap_or(CandidateOutputResidency::Provider {
                    device_id: provider.device_id(),
                }),
            presentation: runmat_execution::CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256_serialized(&(&entry.identity, &entry.cost_query))
                .expect("physical route facts serialize"),
            cost: entry.cost,
            resources: candidate_resource(entry),
            transactional_results: transactional && require_transactional_results,
        });
    }
    // Match the numeric selector's price domain before whole-set validation.
    // A malformed estimate is not unavailable evidence and cannot invalidate
    // otherwise legal routes. Preserve the original sequence for receipts.
    let original_candidates = candidates;
    let mut candidates = Vec::new();
    candidates
        .try_reserve_exact(original_candidates.len())
        .map_err(|_| {
            plan_error(crate::placement::PlacementPlanError {
                code: "RunMat:Placement:ResourceOverflow",
                message: "filtered candidate allocation failed".into(),
            })
        })?;
    candidates.extend(
        original_candidates
            .iter()
            .filter(|candidate| {
                candidate
                    .cost
                    .estimate()
                    .is_none_or(|estimate| estimate.checked_risk_adjusted_ns().is_some())
            })
            .cloned(),
    );
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
    candidates.retain(|candidate| {
        match crate::placement::session::candidate_resources_admitted(
            candidate,
            &resources,
            require_transactional_results,
        ) {
            Ok(()) => true,
            Err(reason) => {
                insufficient |= matches!(
                    reason,
                    crate::placement::partition::ResourceAdmissionError::InsufficientEvidence { .. }
                );
                false
            }
        }
    });
    if candidates.is_empty() {
        if vm_decline.is_some_and(|right| runtime.is_some_and(|context| right.is_current(context)))
            || cpu
                .native_origin
                .as_ref()
                .is_some_and(|right| runtime.is_some_and(|context| right.is_current(context)))
        {
            return Ok(LocalPlacementOutcome::ProviderRejected {
                code: rejected
                    .first()
                    .map_or(ProviderRejectionCode::ResourceLimit, |entry| entry.2),
                cpu_cost,
                feedback: None,
            });
        }
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
                    | TotalCandidateObservationContract::NativeFullCall
            )
        })
        .filter(|runtime| {
            runtime
                .service_ports()
                .placement()
                .is_some_and(|service| !bounded_host || service.managed_whole_work_available())
        })
        .and_then(|runtime| {
            let revision = observation_revision(
                runtime,
                observation_contract,
                prepared_authority,
                vm_right,
                cpu.native_origin.as_ref(),
                digest,
                policy,
            )?;
            let facts = if let [entry] = admitted.as_slice() {
                let source = (&entry.identity, &entry.cost_query, entry.output_bytes);
                signature::candidate_set_facts(
                    std::slice::from_ref(&source),
                    &rejected,
                    source_digest,
                    cpu.route.preparation,
                    &residency,
                    &resources,
                    host_epoch,
                )
            } else {
                let sources = admitted
                    .iter()
                    .map(|entry| (&entry.identity, &entry.cost_query, entry.output_bytes))
                    .collect::<Vec<_>>();
                signature::candidate_set_facts(
                    &sources,
                    &rejected,
                    source_digest,
                    cpu.route.preparation,
                    &residency,
                    &resources,
                    host_epoch,
                )
            };
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
    let (selected, decision_kind, predicted_total_ns) = if bounded_host {
        if let Some((runtime, signature)) = authority.as_ref().filter(|(runtime, _)| {
            runtime
                .service_ports()
                .placement()
                .is_some_and(|service| service.managed_whole_work_available())
        }) {
            let service = runtime
                .service_ports()
                .placement()
                .expect("checked managed placement service");
            let managed = service
                .select_whole_work_borrowed_managed(WholeWorkPlacementRequestRef {
                    signature: signature.into(),
                    constraint,
                    candidates: &candidates,
                    resources: &resources,
                    deterministic: false,
                    require_transactional_results,
                })
                .map_err(Box::new)?;
            if signature.permit_identity().ok() != Some(managed.identity) {
                return Err(Box::new(
                    runmat_runtime::build_runtime_error(
                        "managed placement returned a different whole-work identity",
                    )
                    .with_identifier("RunMat:Placement:InvalidDecision")
                    .build(),
                ));
            }
            let result = (
                managed.selected.portable(),
                managed.kind,
                managed.predicted_total_ns,
            );
            if managed.observation_is_current() {
                feedback = Some(Box::new(LocalPlacementFeedback {
                    runtime: (**runtime).clone(),
                    signature: (*signature).clone(),
                    candidate: result.0.candidate.clone(),
                    kind: result.0.kind,
                    authority: LocalPlacementFeedbackAuthority::Managed(std::sync::Arc::new(
                        std::sync::Mutex::new(Some(managed)),
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
                require_transactional_results,
            )
            .map_err(plan_error)?
        }
    } else if let Some((runtime, signature)) = authority {
        let service = runtime
            .service_ports()
            .placement()
            .expect("revision requires selected placement service");
        let selection = if observation_contract == TotalCandidateObservationContract::NativeFullCall
        {
            // Reuse the existing typed optional boundary for graph-only
            // services. This owned metadata path is unbounded only; finite
            // execution uses the managed borrowed branch above.
            service
                .try_select_whole_work(runmat_execution::WholeWorkPlacementRequest {
                    signature: signature.clone(),
                    constraint,
                    candidates: candidates.clone(),
                    resources: resources.clone(),
                    deterministic: false,
                    require_transactional_results,
                })
                .map_err(Box::new)?
        } else {
            Some(
                service
                    .select_whole_work_borrowed(WholeWorkPlacementRequestRef {
                        signature: (&signature).into(),
                        constraint,
                        candidates: &candidates,
                        resources: &resources,
                        deterministic: false,
                        require_transactional_results,
                    })
                    .map_err(Box::new)?,
            )
        };
        match selection {
            Some(selection) => {
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
            }
            None => crate::placement::session::select_unobserved(
                operation,
                &candidates,
                constraint,
                &resources,
                require_transactional_results,
            )
            .map_err(plan_error)?,
        }
    } else {
        crate::placement::session::select_unobserved(
            operation,
            &candidates,
            constraint,
            &resources,
            require_transactional_results,
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
