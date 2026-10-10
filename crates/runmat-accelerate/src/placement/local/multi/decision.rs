//! One service graph and one selected physical receipt for the whole candidate set.
use super::*;

pub(super) struct DecisionRequest<'a> {
    pub(super) provider: &'a dyn AccelProvider,
    pub(super) operation: &'a str,
    pub(super) semantic_identity: Option<Digest>,
    pub(super) observation_contract: TotalCandidateObservationContract,
    pub(super) prepared_authority:
        Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
    pub(super) vm_right: Option<&'a runmat_runtime::context::VmFusionCompletionRight>,
    pub(super) cpu: ExecutionCandidateDescriptor,
    pub(super) cpu_cost: ExecutionCostEstimate,
    pub(super) admitted: Vec<Admitted>,
    pub(super) rejected: Vec<(
        Option<Digest>,
        ProviderFeasibilityQuery,
        ProviderRejectionCode,
    )>,
    pub(super) source_digest: Digest,
    pub(super) residency: crate::placement::residency::ResidencySummary,
    pub(super) session: Option<(
        runmat_execution::PlacementResourceSnapshot,
        Digest,
        u64,
        bool,
    )>,
    pub(super) compare_profitability: bool,
    pub(super) runtime: Option<&'a runmat_runtime::context::RuntimeContext>,
    pub(super) policy: PlacementPolicy,
}

pub(super) fn select_decision(
    request: DecisionRequest<'_>,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    let DecisionRequest {
        provider,
        operation,
        semantic_identity,
        observation_contract,
        prepared_authority,
        vm_right,
        cpu,
        cpu_cost,
        mut admitted,
        rejected,
        source_digest,
        residency,
        session,
        compare_profitability,
        runtime,
        policy,
    } = request;
    // Stable order makes the set signature and the service graph independent
    // of provider candidate enumeration order. Original indices still map to
    // the caller's exact checked selection/recipe for execution.
    admitted.sort_by(|left, right| left.descriptor.identity.cmp(&right.descriptor.identity));
    if admitted
        .windows(2)
        .any(|pair| pair[0].descriptor.identity == pair[1].descriptor.identity)
    {
        return Err(Box::new(
            runmat_runtime::build_runtime_error("duplicate physical candidate identity")
                .with_identifier("RunMat:Placement:InvalidRequest")
                .build(),
        ));
    }
    let mut graph_candidates = Vec::with_capacity(admitted.len() + 1);
    graph_candidates.push(PlacementGraphCandidate {
        resources: cpu_resource(&cpu),
        descriptor: cpu.clone(),
        output_facts: None,
        transactional_results: true,
    });
    for entry in &admitted {
        graph_candidates.push(PlacementGraphCandidate {
            resources: candidate_resource(entry),
            descriptor: entry.descriptor.clone(),
            output_facts: None,
            transactional_results: entry.transactional,
        });
    }
    let select_local = || {
        select_candidate(
            if compare_profitability {
                std::iter::once(cpu.clone())
                    .chain(admitted.iter().map(|entry| entry.descriptor.clone()))
                    .collect::<Vec<_>>()
            } else {
                admitted
                    .iter()
                    .map(|entry| entry.descriptor.clone())
                    .collect()
            },
            policy,
        )
        .expect("at least one admitted placement candidate")
    };
    let mut feedback = None;
    let mut observation_admitted = false;
    let selected = if compare_profitability {
        if let (Some(runtime), Some((resources, digest, host_epoch, _))) = (runtime, session) {
            let revision = observation_revision(
                runtime,
                observation_contract,
                prepared_authority,
                vm_right,
                digest,
                policy,
            );
            if let (Some(service), Some(revision)) = (runtime.service_ports().placement(), revision)
            {
                let facts = if admitted.len() == 1
                    && admitted[0].descriptor.identity == candidate_name(provider, operation, None)
                {
                    signature::with_output_resources(
                        signature::runtime_facts(
                            &admitted[0].cost_query,
                            cpu.preparation,
                            &residency,
                            &resources,
                            host_epoch,
                        ),
                        Some(admitted[0].output_bytes),
                    )
                } else {
                    signature::candidate_set_facts(
                        &admitted
                            .iter()
                            .map(|entry| {
                                (
                                    &entry.descriptor.identity,
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
                            candidates: graph_candidates,
                        }],
                        edges: Vec::new(),
                    },
                    limits: PlacementGraphLimits::default(),
                    resources,
                    deterministic: false,
                    require_transactional_results: true,
                })?;
                let decision = selection.decision;
                observation_admitted = true;
                let selected_name = &decision.selections[0].candidate;
                let selected = if selected_name == &cpu.identity {
                    cpu.clone()
                } else {
                    admitted
                        .iter()
                        .find(|entry| &entry.descriptor.identity == selected_name)
                        .map(|entry| entry.descriptor.clone())
                        .ok_or_else(|| {
                            Box::new(
                                runmat_runtime::build_runtime_error(
                                    "placement selected an unknown physical candidate",
                                )
                                .with_identifier("RunMat:Placement:InvalidDecision")
                                .build(),
                            )
                        })?
                };
                feedback = Some(Box::new(LocalPlacementFeedback {
                    runtime: runtime.clone(),
                    signature,
                    candidate: selected.identity.clone(),
                    kind: selected.kind,
                    permit: selection.observation_permit,
                }));
                selected
            } else {
                select_local()
            }
        } else {
            select_local()
        }
    } else {
        select_local()
    };
    let chosen = admitted
        .iter()
        .find(|entry| entry.descriptor.identity == selected.identity);
    let receipt = chosen
        .or_else(|| admitted.first())
        .expect("admitted provider candidate");
    Ok(LocalPlacementOutcome::Selected {
        selected: Box::new(selected),
        cpu_cost,
        provider_cost: receipt.descriptor.cost,
        feedback,
        provider_query: Box::new(receipt.cost_query.operation.clone()),
        provider_observation: observation_admitted
            && if admitted.len() == 1
                && admitted[0].descriptor.identity == candidate_name(provider, operation, None)
            {
                // Preserve the old one-record receipt even when CPU won.
                receipt.cost_query.context.observation.is_some()
            } else {
                chosen.is_some_and(|entry| entry.cost_query.context.observation.is_some())
            },
        provider_index: chosen.map(|entry| entry.original_index),
    })
}
