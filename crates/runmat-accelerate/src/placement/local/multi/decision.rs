//! One service graph and one selected physical receipt for the whole candidate set.
use super::*;
use crate::placement::partition::{
    single_candidate_resources_admitted, CandidateResourceFacts, ResourceAdmissionError,
};

pub(super) struct DecisionRequest<'a> {
    pub(super) constraint: runmat_execution::WholeWorkConstraint,
    pub(super) provider: &'a dyn AccelProvider,
    pub(super) operation: &'a str,
    pub(super) semantic_identity: Option<Digest>,
    pub(super) observation_contract: TotalCandidateObservationContract,
    pub(super) prepared_authority:
        Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
    pub(super) vm_right: Option<&'a runmat_runtime::context::VmFusionCompletionRight>,
    pub(super) vm_decline: Option<&'a runmat_runtime::context::VmFusionDeclineRight>,
    pub(super) cpu: LocalCpuCandidate,
    pub(super) cpu_resources: Option<CandidateResourceDemand>,
    pub(super) cpu_cost: runmat_execution::CandidateCostEvidence,
    pub(super) admitted: Vec<Admitted>,
    pub(super) rejected: Vec<(
        Option<Digest>,
        ProviderFeasibilityQuery,
        ProviderRejectionCode,
    )>,
    pub(super) source_digest: Digest,
    pub(super) residency: crate::placement::residency::ResidencySummary,
    pub(super) physical_facts: (
        runmat_execution::PlacementResourceSnapshot,
        Digest,
        u64,
        bool,
    ),
    pub(super) compare_profitability: bool,
    pub(super) runtime: Option<&'a runmat_runtime::context::RuntimeContext>,
    pub(super) publication: LocalPublicationSemantics,
    pub(super) policy: PlacementPolicy,
    pub(super) bounded_host: bool,
}

pub(super) fn select_decision(
    request: DecisionRequest<'_>,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    if matches!(
        request.publication,
        LocalPublicationSemantics::SourceOrderedVm | LocalPublicationSemantics::SourceOrderedNative
    ) || request.cpu.native_host_alternative.is_some()
        || request.cpu.route.cost.estimate().is_none()
        || request
            .admitted
            .iter()
            .any(|entry| entry.descriptor.is_none())
    {
        return super::unpriced::select(request);
    }
    let DecisionRequest {
        constraint,
        provider,
        operation,
        semantic_identity,
        observation_contract,
        prepared_authority,
        vm_right,
        vm_decline: _,
        mut cpu,
        cpu_resources,
        cpu_cost,
        mut admitted,
        rejected,
        source_digest,
        residency,
        physical_facts,
        compare_profitability,
        runtime,
        publication: _,
        policy,
        bounded_host: _,
    } = request;
    let native_origin = cpu.native_origin.take();
    let legal_native = |candidate: &runmat_execution::PlacementGraphCandidate| match constraint {
        runmat_execution::WholeWorkConstraint::AnyLegal => true,
        runmat_execution::WholeWorkConstraint::ProviderRequired { device_id } => matches!(
            candidate.descriptor.execution_location,
            CandidateExecutionLocation::Provider { device_id: selected }
                | CandidateExecutionLocation::VmRegion { device_id: selected }
                | CandidateExecutionLocation::NativeRegion { device_id: selected }
                if selected == device_id
        ),
    };
    let native_alternative = cpu.native_alternative.take().filter(legal_native);
    let cpu = cpu
        .into_priced()
        .expect("numeric branch requires an estimated host carrier");
    let native_set_identity = native_alternative.is_some().then(|| {
        Digest::sha256_serialized(&("native-numeric-cpu-set-v3", &cpu, &native_alternative))
            .expect("typed native candidates serialize")
    });
    let (resources, digest, host_epoch, _) = physical_facts;
    resources.validate().map_err(|message| {
        plan_error(crate::placement::PlacementPlanError {
            code: "RunMat:Placement:InvalidRequest",
            message: message.into(),
        })
    })?;
    // Stable order makes the set signature and the service graph independent
    // of provider candidate enumeration order. Original indices still map to
    // the caller's exact checked selection/recipe for execution.
    admitted.sort_by(|left, right| left.priced().identity.cmp(&right.priced().identity));
    if admitted
        .windows(2)
        .any(|pair| pair[0].priced().identity == pair[1].priced().identity)
    {
        return Err(Box::new(
            runmat_runtime::build_runtime_error("duplicate physical candidate identity")
                .with_identifier("RunMat:Placement:InvalidRequest")
                .build(),
        ));
    }
    let mut graph_candidates =
        Vec::with_capacity(admitted.len() + 1 + usize::from(native_alternative.is_some()));
    if let Some(resources) = cpu_resources.filter(|_| compare_profitability) {
        graph_candidates.push(PlacementGraphCandidate {
            resources,
            descriptor: cpu.clone(),
            output_facts: None,
            transactional_results: true,
        });
    }
    if compare_profitability {
        if let Some(alternative) = native_alternative {
            graph_candidates.push(alternative);
        }
    }
    for entry in &admitted {
        graph_candidates.push(PlacementGraphCandidate {
            resources: candidate_resource(entry),
            descriptor: entry.priced().clone(),
            output_facts: None,
            transactional_results: entry.transactional,
        });
    }
    let mut insufficient_resource_evidence = false;
    graph_candidates.retain(|candidate| {
        let descriptor = &candidate.descriptor;
        // Match the canonical numeric selector's admissible price domain
        // before graph publication or cross-route classification.
        if descriptor.validate().is_err() || descriptor.cost.checked_risk_adjusted_ns().is_none() {
            return false;
        }
        match single_candidate_resources_admitted(
            CandidateResourceFacts {
                location: descriptor.execution_location,
                output_residency: descriptor.output_residency,
                presentation: &descriptor.presentation,
                kind: descriptor.kind,
                demand: candidate.resources,
                transactional_results: candidate.transactional_results,
            },
            &resources,
            true,
        ) {
            Ok(()) => true,
            Err(reason) => {
                insufficient_resource_evidence |=
                    matches!(reason, ResourceAdmissionError::InsufficientEvidence { .. });
                false
            }
        }
    });
    if resources.cancellation_requested {
        return Err(plan_error(crate::placement::PlacementPlanError {
            code: "RunMat:Placement:Cancelled",
            message: "placement cancelled before dispatch".into(),
        }));
    }
    if graph_candidates.is_empty() {
        return Err(plan_error(
            crate::placement::session::no_resource_candidate(insufficient_resource_evidence),
        ));
    }
    let compared_routes = graph_candidates
        .iter()
        .any(|candidate| !candidate.descriptor.kind.is_provider())
        && graph_candidates
            .iter()
            .any(|candidate| candidate.descriptor.kind.is_provider());
    let select_local = |candidates: &[PlacementGraphCandidate]| {
        select_candidate(
            candidates
                .iter()
                .map(|candidate| candidate.descriptor.clone())
                .collect::<Vec<_>>(),
            policy,
        )
        .ok_or_else(|| plan_error(crate::placement::session::no_resource_candidate(false)))
    };
    let mut feedback = None;
    let mut observation_admitted = false;
    let selected = if compare_profitability {
        if let Some(runtime) = runtime {
            let revision = observation_revision(
                runtime,
                observation_contract,
                prepared_authority,
                vm_right,
                native_origin.as_ref(),
                digest,
                policy,
            );
            if let (Some(service), Some(revision)) = (runtime.service_ports().placement(), revision)
            {
                let facts = if admitted.len() == 1
                    && admitted[0].priced().identity == candidate_name(provider, operation, None)
                {
                    signature::with_output_resources(
                        signature::runtime_facts(
                            &admitted[0].cost_query,
                            cpu.preparation,
                            &residency,
                            &resources,
                            host_epoch,
                        ),
                        admitted[0].output_size(),
                    )
                } else {
                    signature::candidate_set_facts(
                        &admitted
                            .iter()
                            .map(|entry| {
                                (
                                    &entry.priced().identity,
                                    &entry.cost_query,
                                    entry.output_bytes,
                                )
                            })
                            .collect::<Vec<_>>(),
                        &rejected,
                        source_digest,
                        cpu.preparation,
                        &residency,
                        &resources,
                        host_epoch,
                    )
                };
                let facts = native_set_identity.map_or(facts, |identity| {
                    signature::with_semantic_identity(facts, Some(identity))
                });
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
                    provider_provenance: super::super::session_provider_provenance(
                        &revision, digest,
                    ),
                    graph_output_authority: None,
                    revision,
                };
                let selection = service.plan_with_observation_permit(PlacementPlanRequest {
                    signature: signature.clone(),
                    graph: PlacementGraph {
                        nodes: vec![PlacementGraphNode {
                            identity: operation.into(),
                            candidates: graph_candidates.clone(),
                        }],
                        edges: Vec::new(),
                    },
                    limits: PlacementGraphLimits::default(),
                    resources,
                    deterministic: false,
                    require_transactional_results: true,
                })?;
                let decision = selection.decision;
                if decision.signature != signature || decision.selections.len() != 1 {
                    return Err(Box::new(
                        runmat_runtime::build_runtime_error(
                            "placement returned a different physical graph decision",
                        )
                        .with_identifier("RunMat:Placement:InvalidDecision")
                        .build(),
                    ));
                }
                observation_admitted = true;
                let selected_name = decision
                    .selections
                    .first()
                    .map(|selected| &selected.candidate);
                let selected = graph_candidates
                    .iter()
                    .find(|entry| {
                        Some(&entry.descriptor.identity) == selected_name
                            && decision.selections[0].kind == entry.descriptor.kind
                            && decision.selections[0].node == operation
                    })
                    .map(|entry| entry.descriptor.clone())
                    .ok_or_else(|| {
                        Box::new(
                            runmat_runtime::build_runtime_error(
                                "placement selected an unknown physical candidate",
                            )
                            .with_identifier("RunMat:Placement:InvalidDecision")
                            .build(),
                        )
                    })?;
                feedback = selection.observation_permit.map(|permit| {
                    Box::new(LocalPlacementFeedback {
                        runtime: runtime.clone(),
                        signature,
                        candidate: selected.identity.clone(),
                        kind: selected.kind,
                        authority: LocalPlacementFeedbackAuthority::Graph(permit),
                    })
                });
                selected
            } else {
                select_local(&graph_candidates)?
            }
        } else {
            select_local(&graph_candidates)?
        }
    } else {
        select_local(&graph_candidates)?
    };
    let chosen = admitted
        .iter()
        .find(|entry| entry.priced().identity == selected.identity);
    let receipt = chosen
        .or_else(|| admitted.first())
        .expect("admitted provider candidate");
    Ok(LocalPlacementOutcome::Selected {
        predicted_total_ns: selected.cost.checked_total_ns(),
        selected: Box::new(runmat_execution::SelectedExecutionCandidate {
            node: operation.into(),
            candidate: selected.identity,
            kind: selected.kind,
        }),
        decision_kind: if matches!(
            constraint,
            runmat_execution::WholeWorkConstraint::ProviderRequired { .. }
        ) {
            runmat_execution::WholeWorkDecisionKind::ProviderRequired
        } else if compared_routes {
            runmat_execution::WholeWorkDecisionKind::Compared
        } else {
            runmat_execution::WholeWorkDecisionKind::Uncompared
        },
        cpu_cost: if selected.kind.is_provider() {
            cpu_cost
        } else {
            runmat_execution::CandidateCostEvidence::Estimated(selected.cost)
        },
        provider_cost: Some(receipt.priced().cost),
        provider_rejection: None,
        feedback,
        provider_query: Box::new(receipt.cost_query.operation.clone()),
        provider_observation: observation_admitted
            && if admitted.len() == 1
                && admitted[0].priced().identity == candidate_name(provider, operation, None)
            {
                // Preserve the old one-record receipt even when CPU won.
                receipt.cost_query.context.observation.is_some()
            } else {
                chosen.is_some_and(|entry| entry.cost_query.context.observation.is_some())
            },
        provider_index: chosen.map(|entry| entry.original_index),
    })
}
