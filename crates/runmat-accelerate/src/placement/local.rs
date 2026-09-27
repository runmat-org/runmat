use runmat_accelerate_api::{
    AccelProvider, ProviderCostQuery, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderRejectionCode,
    ProviderRepresentation, ProviderWorkload,
};
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateDescriptor, ExecutionCandidateKind,
    ExecutionCostEstimate, PlacementGraph, PlacementGraphCandidate, PlacementGraphLimits,
    PlacementGraphNode, PlacementPlanRequest, PlacementResourceSnapshot, PlacementRevision,
    PlacementSignature,
};
use runmat_value::Value;

use super::{select_candidate, summarize_values, tensor_representation, PlacementPolicy};
mod signature;
#[cfg(test)]
mod resource_tests;

pub(crate) struct LocalPlacementRequest<'a> {
    pub(crate) operation: &'a str,
    /// Optional deterministic identity of the selected typed program. This
    /// augments, rather than replaces, argument/resource/preparation facts.
    pub(crate) semantic_identity: Option<Digest>,
    pub(crate) family: ProviderOperationFamily,
    pub(crate) provider_kind: ExecutionCandidateKind,
    pub(crate) inputs: &'a [&'a Value],
    pub(crate) outputs: Vec<ProviderRepresentation>,
    pub(crate) workload: ProviderWorkload,
    pub(crate) preparation: CandidatePreparationState,
    pub(crate) preparation_program: Option<runmat_accelerate_api::ProviderPreparationProgram<'a>>,
    pub(crate) cpu: ExecutionCandidateDescriptor,
    /// Provider execution prior without transfer components. It is used only
    /// when the provider has no trustworthy estimate of its own.
    pub(crate) provider_fallback: ExecutionCostEstimate,
    /// Physical transfer bytes when an admitted preparation changes the input
    /// representation. None preserves ordinary residency-based accounting.
    pub(crate) required_upload_bytes: Option<u64>,
    pub(crate) required_download_bytes: u64,
    pub(crate) downstream_materialization: bool,
    /// Direct provider APIs preserve caller intent after feasibility succeeds;
    /// correlated runtime requests compare against the CPU candidate.
    pub(crate) compare_profitability: bool,
    /// Explicit session authority for guarded cached/adaptive planning. Direct
    /// provider APIs and legacy call sites may omit it and use deterministic
    /// local selection.
    pub(crate) runtime: Option<&'a runmat_runtime::context::RuntimeContext>,
}

#[derive(Clone, Debug)]
pub(crate) enum LocalPlacementOutcome {
    Selected {
        selected: Box<ExecutionCandidateDescriptor>,
        cpu_cost: ExecutionCostEstimate,
        provider_cost: ExecutionCostEstimate,
        feedback: Option<Box<LocalPlacementFeedback>>,
        /// Exact compact query already validated by this placement authority.
        provider_query: Box<ProviderFeasibilityQuery>,
        provider_observation: bool,
    },
    ProviderRejected {
        code: ProviderRejectionCode,
        cpu_cost: ExecutionCostEstimate,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct LocalPlacementFeedback {
    pub(crate) runtime: runmat_runtime::context::RuntimeContext,
    pub(crate) signature: PlacementSignature,
    pub(crate) candidate: String,
}

/// Normalize provider feasibility and complete transfer costs before comparing
/// profitability. This is the one local authority shared by legacy automatic
/// offload and correlated fusion execution.
pub(crate) fn plan_local(
    provider: &dyn AccelProvider,
    request: LocalPlacementRequest<'_>,
    policy: PlacementPolicy,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    debug_assert!(request.provider_kind.is_provider());
    let residency = summarize_values(request.inputs);
    let provider_resident_bytes = residency
        .provider_bytes
        .values()
        .copied()
        .fold(0_u64, u64::saturating_add);
    let mut cpu = request.cpu;
    if provider_resident_bytes > 0 {
        cpu.cost.components.synchronization_ns =
            cpu.cost.components.synchronization_ns.saturating_add(5_000);
        cpu.cost.components.download_ns = cpu
            .cost
            .components
            .download_ns
            .saturating_add(transfer_prior_ns(provider_resident_bytes));
    }
    let mut workload = request.workload;
    // Physical route evidence comes from the admitted program, never the
    // operation name or an unresolved legacy/template workload.
    workload.preparation_route = request
        .preparation_program
        .and_then(|program| program.route());
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(request.operation),
        family: request.family,
        inputs: request
            .inputs
            .iter()
            .filter_map(|value| tensor_representation(value, provider.precision()))
            .collect(),
        outputs: request.outputs,
        workload,
    };
    let estimate = match provider.query_feasibility(&query) {
        ProviderFeasibility::Supported { estimate } => estimate,
        ProviderFeasibility::Rejected { rejection } => {
            return Ok(LocalPlacementOutcome::ProviderRejected {
                code: rejection.code,
                cpu_cost: cpu.cost,
            });
        }
    };
    // The provider owns physical allocation widths and inferred outputs. Empty
    // query.outputs means unspecified output facts, not zero retained bytes.
    // Explicit provider execution remains supported when this estimate is absent.
    if request.compare_profitability && estimate.output_bytes.is_none() {
        return Ok(LocalPlacementOutcome::ProviderRejected {
            code: ProviderRejectionCode::UnknownResourceEstimate,
            cpu_cost: cpu.cost,
        });
    }

    let upload_bytes = request
        .required_upload_bytes
        .unwrap_or_else(|| residency.required_upload_bytes(provider.device_id()));
    let context = runmat_accelerate_api::ProviderCostContext {
            preparation: request.preparation_program.and_then(|program| provider.preparation_snapshot(&query, program)),
            observation: request.preparation_program.filter(|_| request.compare_profitability).and_then(|_| request.runtime)
                .and_then(|runtime| runtime.service_ports().placement())
                .and_then(|_| match provider.operation_observation_capability(&query) {
                    runmat_accelerate_api::observation::ProviderObservationCapability::Supported(overhead) => Some(overhead),
                    runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported => None,
                }),
        };
    let preparation = if request.preparation_program.is_some() {
        context
            .preparation
            .as_ref()
            .and_then(|snapshot| snapshot.state)
            .unwrap_or(CandidatePreparationState::Cold)
    } else {
        request.preparation
    };
    let cost_query = ProviderCostQuery {
        context,
        operation: query.clone(),
        preparation,
        required_upload_bytes: upload_bytes,
        required_download_bytes: request.required_download_bytes,
        downstream_materialization: request.downstream_materialization,
    };
    let provider_cost = provider
        .estimate_cost(&cost_query)
        .map(|estimate| estimate.cost)
        .unwrap_or_else(|| {
            let mut cost = request.provider_fallback;
            cost.components.upload_ns = transfer_prior_ns(upload_bytes);
            cost.components.download_ns = transfer_prior_ns(request.required_download_bytes);
            cost.components.synchronization_ns = if request.required_download_bytes == 0 {
                0
            } else {
                cost.components.synchronization_ns.max(5_000)
            };
            cost
        });
    let provider_candidate = ExecutionCandidateDescriptor {
        identity: format!("provider.{}.{}", provider.device_id(), request.operation),
        region: cpu.region,
        kind: request.provider_kind,
        execution_location: CandidateExecutionLocation::Provider {
            device_id: provider.device_id(),
        },
        preparation,
        cost: provider_cost,
        output_residency: CandidateOutputResidency::Provider {
            device_id: provider.device_id(),
        },
        guards: cpu.guards.clone(),
    };
    let cpu_cost = cpu.cost;
    let mut feedback = None;
    let selected = if request.compare_profitability {
        if let Some(runtime) = request.runtime {
            if let Some(service) = runtime.service_ports().placement() {
                let capability = provider.capability_snapshot();
                let provider_resources = provider.placement_resources();
                let provider_digest = Digest::sha256(
                    serde_json::to_vec(&capability)
                        .expect("provider capability snapshot must serialize"),
                );
                let host_resources = runtime
                    .service_ports()
                    .parallel()
                    .map(|service| service.placement_resources())
                    .unwrap_or_default();
                let resources = PlacementResourceSnapshot {
                    cpu_millicores_available: host_resources.cpu_millicores_available,
                    memory_available_bytes: host_resources.memory_available_bytes,
                    cancellation_requested: runtime.is_cancelled(),
                    providers: vec![provider_resources.clone()],
                    epoch: provider_resources.epoch ^ host_resources.epoch.rotate_left(32),
                };
                let signature = PlacementSignature {
                    region: cpu.region,
                    operation: request.operation.into(),
                    runtime_facts: signature::with_semantic_identity(
                        signature::with_output_resources(
                            signature::runtime_facts(
                                &cost_query,
                                cpu.preparation,
                                &residency,
                                &resources,
                                host_resources.epoch,
                            ),
                            estimate.output_bytes,
                        ),
                        request.semantic_identity,
                    ),
                    revision: PlacementRevision {
                        program: runtime.program_revision().cloned(),
                        catalog: Digest::sha256(b"runmat-placement-catalog-v1"),
                        compiler: Digest::sha256(b"runmat-placement-compiler-v1"),
                        provider: provider_digest,
                        policy: Digest::sha256(format!(
                            "local:{}:{}",
                            policy.absolute_margin_ns, policy.relative_margin_basis_points
                        )),
                    },
                };
                let transactional = capability.concurrency.transactional_results;
                let decision = service.plan(PlacementPlanRequest {
                    signature: signature.clone(),
                    graph: PlacementGraph {
                        nodes: vec![PlacementGraphNode {
                            identity: request.operation.into(),
                            candidates: vec![
                                PlacementGraphCandidate {
                                    resources: CandidateResourceDemand {
                                        cpu_millicores: 1_000,
                                        retained_bytes: 0,
                                        scratch_bytes: cpu_cost.scratch_bytes,
                                        queue_slots: 0,
                                    },
                                    descriptor: cpu.clone(),
                                    transactional_results: true,
                                },
                                PlacementGraphCandidate {
                                    resources: CandidateResourceDemand {
                                        cpu_millicores: 0,
                                        retained_bytes: estimate.output_bytes.expect(
                                            "automatic placement admitted known physical output bytes",
                                        ),
                                        scratch_bytes: provider_cost.scratch_bytes,
                                        queue_slots: 1,
                                    },
                                    descriptor: provider_candidate.clone(),
                                    transactional_results: transactional,
                                },
                            ],
                        }],
                        edges: Vec::new(),
                    },
                    limits: PlacementGraphLimits::default(),
                    resources,
                    deterministic: false,
                    require_transactional_results: true,
                })?;
                if decision.selections[0].kind.is_provider() {
                    feedback = Some(Box::new(LocalPlacementFeedback {
                        runtime: runtime.clone(),
                        signature: signature.clone(),
                        candidate: provider_candidate.identity.clone(),
                    }));
                    provider_candidate
                } else {
                    feedback = Some(Box::new(LocalPlacementFeedback {
                        runtime: runtime.clone(),
                        signature,
                        candidate: cpu.identity.clone(),
                    }));
                    cpu
                }
            } else {
                select_candidate([cpu, provider_candidate], policy).expect(
                    "CPU and provider placement candidates must contain one valid descriptor",
                )
            }
        } else {
            select_candidate([cpu, provider_candidate], policy)
                .expect("CPU and provider placement candidates must contain one valid descriptor")
        }
    } else {
        provider_candidate
    };
    Ok(LocalPlacementOutcome::Selected {
        selected: Box::new(selected),
        cpu_cost,
        provider_cost,
        feedback,
        provider_observation: cost_query.context.observation.is_some(),
        provider_query: Box::new(cost_query.operation),
    })
}

pub(crate) fn transfer_prior_ns(bytes: u64) -> u64 {
    if bytes == 0 {
        0
    } else {
        20_000_u64.saturating_add(bytes.saturating_add(7) / 8)
    }
}
