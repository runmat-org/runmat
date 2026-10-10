//! One canonical CPU/provider candidate set for a local placement decision.
//! Each physical record is normalized before a single service plan call.
use super::{
    LocalPlacementFeedback, LocalPlacementOutcome, PlacementPolicy, RejectedCpuFeedbackRequest,
    TotalCandidateObservationContract, observation_revision, observation_semantic_identity,
    rejected_cpu_feedback, session_facts, signature, transfer_prior_ns,
};
use crate::placement::{select_candidate, summarize_values, tensor_representation};
use runmat_accelerate_api::{
    AccelProvider, ProviderCostQuery, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderPreparationProgram,
    ProviderPreparationRoute, ProviderRejectionCode, ProviderRepresentation,
    ProviderShaderRequirements, ProviderWorkload,
};
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateDescriptor, ExecutionCandidateKind,
    ExecutionCostEstimate, PlacementGraph, PlacementGraphCandidate, PlacementGraphLimits,
    PlacementGraphNode, PlacementPlanRequest, PlacementSignature, ResourceBoundUnavailable,
    ResourceByteBound,
};
use runmat_value::Value;

pub(crate) struct LocalPhysicalCandidate<'a> {
    /// Selection.identity() for a typed strategy. None preserves legacy
    /// single-candidate naming and feedback identity.
    pub(crate) physical_identity: Option<Digest>,
    pub(crate) provider_kind: ExecutionCandidateKind,
    /// Actual ordered data leaves, including constants, excluding controls.
    pub(crate) data_inputs: &'a [&'a Value],
    /// Exact representations of the actual ordered values before effects.
    /// None retains the legacy one-record provider-default projection.
    pub(crate) source_inputs: Option<Vec<ProviderRepresentation>>,
    /// Representations handed to the physical provider hook after admitted
    /// preparation. They may differ from sources only with a checked route.
    pub(crate) prepared_inputs: Option<Vec<ProviderRepresentation>>,
    /// Domain-admitted transformation, never inferred from equal byte counts.
    /// The execution owner must recheck and perform this route before dispatch.
    pub(crate) input_adaptation: Option<ProviderPreparationRoute>,
    pub(crate) outputs: Vec<ProviderRepresentation>,
    pub(crate) workload: ProviderWorkload,
    pub(crate) preparation: CandidatePreparationState,
    pub(crate) preparation_program: Option<ProviderPreparationProgram<'a>>,
    pub(crate) shader_requirements: Option<ProviderShaderRequirements>,
    pub(crate) fallback: ExecutionCostEstimate,
    pub(crate) required_upload_bytes: Option<u64>,
    pub(crate) required_download_bytes: u64,
    pub(crate) downstream_materialization: bool,
}

pub(crate) enum LocalCpuAdmission {
    /// Preserve the established resource projection for ordinary host routes.
    Legacy,
    /// A domain owner proved an executable host route and its complete demand.
    Exact(Box<CandidateResourceDemand>),
    /// No executable host route was admitted; do not fabricate a CPU candidate.
    Unavailable,
}

pub(crate) struct LocalMultiPlacementRequest<'a> {
    pub(crate) operation: &'a str,
    pub(crate) semantic_identity: Option<Digest>,
    pub(crate) observation_contract: TotalCandidateObservationContract,
    pub(crate) prepared_authority:
        Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
    pub(crate) vm_right: Option<&'a runmat_runtime::context::VmFusionCompletionRight>,
    pub(crate) family: ProviderOperationFamily,
    pub(crate) cpu: ExecutionCandidateDescriptor,
    pub(crate) cpu_admission: LocalCpuAdmission,
    pub(crate) physical: Vec<LocalPhysicalCandidate<'a>>,
    pub(crate) compare_profitability: bool,
    pub(crate) runtime: Option<&'a runmat_runtime::context::RuntimeContext>,
}

mod decision;
mod graph;
mod inputs;
use graph::{Admitted, candidate_name, candidate_resource, cpu_resource};

pub(crate) fn plan_local_multi(
    provider: &dyn AccelProvider,
    request: LocalMultiPlacementRequest<'_>,
    policy: PlacementPolicy,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    let LocalMultiPlacementRequest {
        operation,
        semantic_identity,
        observation_contract,
        prepared_authority,
        vm_right,
        family,
        mut cpu,
        cpu_admission,
        physical,
        compare_profitability,
        runtime,
    } = request;
    let cpu_resources = match cpu_admission {
        LocalCpuAdmission::Legacy => Some(cpu_resource(&cpu)),
        LocalCpuAdmission::Exact(resources) => Some(*resources),
        LocalCpuAdmission::Unavailable => None,
    };
    let inputs::SourcePreflight {
        digest: source_digest,
        routes,
    } = inputs::validate_set(provider, operation, &physical)?;
    let residency = summarize_values(physical[0].data_inputs);
    let resident_bytes = residency
        .provider_bytes
        .values()
        .copied()
        .fold(0_u64, u64::saturating_add);
    if resident_bytes > 0 {
        cpu.cost.components.synchronization_ns =
            cpu.cost.components.synchronization_ns.saturating_add(5_000);
        cpu.cost.components.download_ns = cpu
            .cost
            .components
            .download_ns
            .saturating_add(transfer_prior_ns(resident_bytes));
    }
    let cpu_cost = cpu.cost;
    let mut admitted = Vec::new();
    let mut rejected = Vec::new();
    let mut session = None;
    if let Some(runtime) = runtime.filter(|_| compare_profitability) {
        if runtime.service_ports().placement().is_some() {
            let facts = session_facts(provider, runtime);
            session = Some(facts);
        }
    }
    for ((original_index, candidate), route) in physical.into_iter().enumerate().zip(routes) {
        debug_assert!(candidate.provider_kind.is_provider());
        let candidate_residency = summarize_values(candidate.data_inputs);
        let mut workload = candidate.workload;
        workload.preparation_route = route;
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(operation),
            family,
            inputs: candidate.prepared_inputs.unwrap_or_else(|| {
                candidate
                    .data_inputs
                    .iter()
                    .filter_map(|value| tensor_representation(value, provider.precision()))
                    .collect()
            }),
            outputs: candidate.outputs,
            workload,
        };
        if query
            .outputs
            .iter()
            .try_fold(0_u64, |total, output| {
                total.checked_add(output.checked_byte_len()?)
            })
            .is_none()
        {
            return Err(Box::new(
                runmat_runtime::build_runtime_error("automatic placement output size overflows")
                    .with_identifier("RunMat:Placement:ResourceOverflow")
                    .build(),
            ));
        }
        let feasibility = match candidate.shader_requirements {
            Some(requirements) if !provider.supports_shader_requirements(requirements) => {
                ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnsupportedElementType,
                    "selected_shader_requirements_unavailable",
                )
            }
            _ => provider.query_feasibility(&query),
        };
        let estimate = match feasibility {
            ProviderFeasibility::Supported { estimate } => estimate,
            ProviderFeasibility::Rejected { rejection } => {
                rejected.push((candidate.physical_identity, query, rejection.code));
                continue;
            }
        };
        if compare_profitability && estimate.output_bytes.is_none() {
            rejected.push((
                candidate.physical_identity,
                query,
                ProviderRejectionCode::UnknownResourceEstimate,
            ));
            continue;
        }
        let context = runmat_accelerate_api::ProviderCostContext {
            preparation: candidate.preparation_program
                .and_then(|program| provider.preparation_snapshot(&query, program)),
            observation: candidate.preparation_program.filter(|_| compare_profitability)
                .and(runtime)
                .and_then(|runtime| runtime.service_ports().placement())
                .and_then(|_| match provider.operation_observation_capability(&query) {
                    runmat_accelerate_api::observation::ProviderObservationCapability::Supported(overhead) =>
                        Some(overhead),
                    runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported => None,
                }),
        };
        let preparation = if candidate.preparation_program.is_some() {
            context
                .preparation
                .as_ref()
                .and_then(|snapshot| snapshot.state)
                .unwrap_or(CandidatePreparationState::Cold)
        } else {
            candidate.preparation
        };
        let upload_bytes = candidate
            .required_upload_bytes
            .unwrap_or_else(|| candidate_residency.required_upload_bytes(provider.device_id()));
        let cost_query = ProviderCostQuery {
            context,
            operation: query,
            preparation,
            required_upload_bytes: upload_bytes,
            required_download_bytes: candidate.required_download_bytes,
            downstream_materialization: candidate.downstream_materialization,
        };
        let cost = provider
            .estimate_cost(&cost_query)
            .map(|estimate| estimate.cost)
            .unwrap_or_else(|| {
                let mut cost = candidate.fallback;
                cost.components.upload_ns = transfer_prior_ns(upload_bytes);
                cost.components.download_ns = transfer_prior_ns(candidate.required_download_bytes);
                cost.components.synchronization_ns = if candidate.required_download_bytes == 0 {
                    0
                } else {
                    cost.components.synchronization_ns.max(5_000)
                };
                cost
            });
        let descriptor = ExecutionCandidateDescriptor {
            identity: candidate_name(provider, operation, candidate.physical_identity),
            region: cpu.region,
            kind: candidate.provider_kind,
            execution_location: CandidateExecutionLocation::Provider {
                device_id: provider.device_id(),
            },
            preparation,
            cost,
            output_residency: CandidateOutputResidency::Provider {
                device_id: provider.device_id(),
            },
            presentation: runmat_execution::CandidateOutputPresentation::Direct,
            guards: cpu.guards.clone(),
        };
        let typed_peak = cost_query
            .operation
            .workload
            .typed_reduction
            .as_ref()
            .and_then(|typed| {
                typed
                    .work
                    .output_requested_bytes
                    .checked_add(typed.work.partial_requested_bytes)
            });
        if cost_query.operation.workload.typed_reduction.is_some() && typed_peak.is_none() {
            return Err(Box::new(
                runmat_runtime::build_runtime_error("typed physical peak size overflows")
                    .with_identifier("RunMat:Placement:ResourceOverflow")
                    .build(),
            ));
        }
        let output_bytes = cost_query
            .operation
            .workload
            .typed_reduction
            .as_ref()
            .map_or(estimate.output_bytes.unwrap_or(0), |typed| {
                estimate
                    .output_bytes
                    .unwrap_or(0)
                    .max(typed.work.output_requested_bytes)
            });
        let transient_bytes = estimate
            .transient_bytes
            .map(|observed| observed.max(typed_peak.unwrap_or(0)));
        let input_resources = cost_query
            .operation
            .workload
            .typed_reduction
            .as_ref()
            .and_then(|typed| typed.input_preparation.as_ref())
            .map(|inputs| {
                crate::placement::typed_reduction_input_resources(provider.device_id(), inputs)
                    .ok_or_else(|| {
                        Box::new(
                            runmat_runtime::build_runtime_error(
                                "typed reduction input resources are unbounded",
                            )
                            .with_identifier("RunMat:Placement:ResourceOverflow")
                            .build(),
                        )
                    })
            })
            .transpose()?;
        admitted.push(Admitted {
            original_index,
            descriptor,
            cost_query,
            output_bytes,
            transient_bytes,
            transactional: session.as_ref().is_some_and(|facts| facts.3),
            input_resources,
        });
    }
    rejected.sort_by(|left, right| left.0.cmp(&right.0));
    if admitted.is_empty() {
        if cpu_resources.is_none() {
            return Err(Box::new(
                runmat_runtime::build_runtime_error(
                    "typed provider candidates rejected and no executable host route exists",
                )
                .with_identifier("RunMat:Placement:NoLegalCandidate")
                .build(),
            ));
        }
        let (identity, query, code) = rejected.first().expect("at least one physical record");
        let rejection_identity = if rejected.len() == 1 {
            semantic_identity
        } else {
            Some(
                Digest::sha256_serialized(&(
                    "local-multi-rejected-v1",
                    semantic_identity,
                    source_digest,
                    &rejected,
                ))
                .expect("rejected candidate set serializes"),
            )
        };
        let feedback = rejected_cpu_feedback(
            provider,
            RejectedCpuFeedbackRequest {
                runtime,
                compare_profitability,
                operation,
                semantic_identity: rejection_identity.or(*identity),
                observation_contract,
                prepared_authority,
                vm_right,
                query,
                code: *code,
                cpu: &cpu,
                residency: &residency,
                policy,
            },
        );
        return Ok(LocalPlacementOutcome::ProviderRejected {
            code: *code,
            cpu_cost,
            feedback,
        });
    }
    decision::select_decision(decision::DecisionRequest {
        provider,
        operation,
        semantic_identity,
        observation_contract,
        prepared_authority,
        vm_right,
        cpu,
        cpu_resources,
        cpu_cost,
        admitted,
        rejected,
        source_digest,
        residency,
        session,
        compare_profitability,
        runtime,
        policy,
    })
}
