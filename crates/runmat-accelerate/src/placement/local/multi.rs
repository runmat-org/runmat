//! One canonical CPU/provider candidate set for a local placement decision.
//! Each physical record is normalized before a single service plan call.
use super::{
    observation_revision, observation_semantic_identity, plan_error, rejected_cpu_feedback,
    signature, transfer_prior_ns, LocalCpuCandidate, LocalPlacementFeedback,
    LocalPlacementFeedbackAuthority, LocalPlacementOutcome, PlacementPolicy,
    RejectedCpuFeedbackRequest, TotalCandidateObservationContract,
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
    PlacementGraph, PlacementGraphCandidate, PlacementGraphLimits, PlacementGraphNode,
    PlacementPlanRequest, PlacementSignature, ResourceBoundUnavailable, ResourceByteBound,
};
use runmat_value::Value;

pub(crate) struct LocalPhysicalCandidate<'a> {
    pub(crate) compound: Option<&'a runmat_runtime::context::preparation::AdmittedBuiltinCompound>,
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
    pub(crate) fallback: runmat_execution::CandidateCostEvidence,
    pub(crate) required_upload_bytes: Option<u64>,
    pub(crate) required_download_bytes: u64,
    pub(crate) downstream_materialization: bool,
}

pub(crate) enum LocalCpuAdmission {
    /// Preserve the established resource projection for ordinary host routes.
    Legacy,
    /// A domain owner proved an executable host route and its complete demand.
    Exact(Box<CandidateResourceDemand>),
    /// Exact sequential logical-view downloads. These replace, rather than
    /// augment, the legacy residency-summary transfer prior.
    ExactReadbacks {
        resources: Box<CandidateResourceDemand>,
        download_bytes: u64,
        readback_count: u64,
    },
    /// No executable host route was admitted; do not fabricate a CPU candidate.
    Unavailable,
}

/// Source implementation admission is separate from provider feasibility.
/// A valid host candidate does not imply that a shader implementation exists.
#[derive(Clone, Copy, serde::Serialize)]
pub(crate) enum LocalProviderAdmission {
    Available,
    ImplementationUnavailable,
}

pub(crate) struct LocalMultiPlacementRequest<'a> {
    pub(crate) operation: &'a str,
    pub(crate) semantic_identity: Option<Digest>,
    pub(crate) observation_contract: TotalCandidateObservationContract,
    pub(crate) prepared_authority:
        Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
    pub(crate) vm_right: Option<&'a runmat_runtime::context::VmFusionCompletionRight>,
    pub(crate) family: ProviderOperationFamily,
    pub(crate) cpu: LocalCpuCandidate,
    pub(crate) cpu_admission: LocalCpuAdmission,
    pub(crate) physical: Vec<LocalPhysicalCandidate<'a>>,
    pub(crate) compare_profitability: bool,
    pub(crate) runtime: Option<&'a runmat_runtime::context::RuntimeContext>,
}

mod decision;
mod graph;
mod inputs;
mod unpriced;
use graph::{candidate_name, candidate_resource, Admitted};

pub(crate) fn plan_local_multi(
    provider: &dyn AccelProvider,
    request: LocalMultiPlacementRequest<'_>,
    policy: PlacementPolicy,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    plan_local_multi_with_provider(provider, request, policy, LocalProviderAdmission::Available)
}

pub(super) fn plan_local_multi_with_provider(
    provider: &dyn AccelProvider,
    request: LocalMultiPlacementRequest<'_>,
    policy: PlacementPolicy,
    implementation: LocalProviderAdmission,
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
    let semantic_identity = match implementation {
        LocalProviderAdmission::Available => semantic_identity,
        LocalProviderAdmission::ImplementationUnavailable => Some(
            Digest::sha256_serialized(&(
                "local-source-implementation-v1",
                semantic_identity,
                implementation,
            ))
            .expect("implementation admission serializes"),
        ),
    };
    let exact_readbacks = match &cpu_admission {
        LocalCpuAdmission::ExactReadbacks {
            download_bytes,
            readback_count,
            ..
        } => Some((*download_bytes, *readback_count)),
        _ => None,
    };
    let cpu_resources = match cpu_admission {
        LocalCpuAdmission::Legacy => Some(cpu.route.resources),
        LocalCpuAdmission::Exact(resources) => Some(*resources),
        LocalCpuAdmission::ExactReadbacks { resources, .. } => Some(*resources),
        LocalCpuAdmission::Unavailable => None,
    };
    let constraint = physical
        .first()
        .and_then(|candidate| {
            candidate.data_inputs.iter().find_map(|value| match value {
                Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle) => {
                    Some(runmat_execution::WholeWorkConstraint::ProviderRequired {
                        device_id: handle.device_id(),
                    })
                }
                _ => None,
            })
        })
        .unwrap_or(runmat_execution::WholeWorkConstraint::AnyLegal);
    let cpu_resources = if matches!(
        constraint,
        runmat_execution::WholeWorkConstraint::ProviderRequired { .. }
    ) {
        None
    } else {
        cpu_resources
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
    if let Some(mut cost) = cpu.route.cost.estimate() {
        if let Some((bytes, count)) = exact_readbacks {
            cost.components.synchronization_ns = cost
                .components
                .synchronization_ns
                .saturating_add(5_000_u64.saturating_mul(count));
            cost.components.download_ns = cost
                .components
                .download_ns
                .saturating_add(transfer_prior_ns(bytes));
        } else if resident_bytes > 0 {
            cost.components.synchronization_ns =
                cost.components.synchronization_ns.saturating_add(5_000);
            cost.components.download_ns = cost
                .components
                .download_ns
                .saturating_add(transfer_prior_ns(resident_bytes));
        }
        cpu.route.cost = runmat_execution::CandidateCostEvidence::Estimated(cost);
    }
    let cpu_cost = cpu.route.cost;
    let mut admitted = Vec::new();
    let mut rejected = Vec::new();
    let physical_facts = super::session_facts::physical_facts(provider, runtime);
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
        let feasibility = match (implementation, candidate.shader_requirements) {
            (LocalProviderAdmission::ImplementationUnavailable, _) => {
                ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnsupportedOperation,
                    "selected_implementation_unavailable",
                )
            }
            (_, Some(requirements)) if !provider.supports_shader_requirements(requirements) => {
                ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnsupportedElementType,
                    "selected_shader_requirements_unavailable",
                )
            }
            _ if candidate.compound.is_some() => {
                let resources = candidate.compound.unwrap().resources();
                ProviderFeasibility::Supported {
                    estimate: runmat_accelerate_api::ProviderResourceEstimate {
                        transient_bytes: resources.peak_live.map(|peak| peak.peak_bytes),
                        output_bytes: match resources.retained_bytes {
                            ResourceByteBound::Known(bytes) => Some(bytes),
                            ResourceByteBound::Unavailable(_) => None,
                        },
                        // Conditional physical stages are not an exact execution
                        // count until the actual domain probe chooses its branch.
                        dispatches: None,
                    },
                }
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
        if compare_profitability
            && estimate.output_bytes.is_none()
            && query.workload.typed_reduction.is_none()
        {
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
                .and_then(|_| match candidate.compound.map_or_else(
                    || provider.operation_observation_capability(&query),
                    |plan| plan.observation_capability(provider)) {
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
        // A quote for the original single real operation does not price this
        // wider probe/conditional-host/restore scope. Until a provider supplies
        // that complete scoped quote, whole-work policy receives true absence
        // and may use only its existing exact-route observations/learning.
        let cost = candidate
            .compound
            .is_none()
            .then(|| provider.estimate_cost(&cost_query))
            .flatten()
            .map(|estimate| runmat_execution::CandidateCostEvidence::Estimated(estimate.cost))
            .unwrap_or_else(|| {
                let Some(mut cost) = candidate.fallback.estimate() else {
                    return candidate.fallback;
                };
                cost.components.upload_ns = transfer_prior_ns(upload_bytes);
                cost.components.download_ns = transfer_prior_ns(candidate.required_download_bytes);
                cost.components.synchronization_ns = if candidate.required_download_bytes == 0 {
                    0
                } else {
                    cost.components.synchronization_ns.max(5_000)
                };
                runmat_execution::CandidateCostEvidence::Estimated(cost)
            });
        let identity = candidate_name(provider, operation, candidate.physical_identity);
        let descriptor = cost.estimate().map(|cost| ExecutionCandidateDescriptor {
            identity: identity.clone(),
            region: cpu.region,
            kind: candidate.provider_kind,
            execution_location: CandidateExecutionLocation::Provider {
                device_id: provider.device_id(),
            },
            preparation,
            cost,
            output_residency: candidate
                .compound
                .map(|plan| plan.output_residency())
                .unwrap_or(CandidateOutputResidency::Provider {
                    device_id: provider.device_id(),
                }),
            presentation: runmat_execution::CandidateOutputPresentation::Direct,
            guards: cpu.guards.clone(),
        });
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
            .map_or(estimate.output_bytes, |typed| {
                Some(
                    estimate
                        .output_bytes
                        .map_or(typed.work.output_requested_bytes, |bytes| {
                            bytes.max(typed.work.output_requested_bytes)
                        }),
                )
            })
            .map_or(
                ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
                ResourceByteBound::Known,
            );
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
            exact_resources: candidate
                .compound
                .map(|plan| plan.resources_with_observation(cost_query.context.observation))
                .transpose()
                .map_err(Box::new)?,
            exact_residency: candidate.compound.map(|plan| plan.output_residency()),
            original_index,
            descriptor,
            identity,
            kind: candidate.provider_kind,
            cost,
            cost_query,
            output_bytes,
            transient_bytes,
            transactional: physical_facts.3,
            input_resources,
        });
    }
    rejected.sort_by(|left, right| left.0.cmp(&right.0));
    if admitted.is_empty() && cpu.route.cost.estimate().is_some() {
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
        // Provider rejection does not certify the fallback host's resource
        // legality. Admit that exact single route without observation rights.
        let resources = &physical_facts.0;
        let mut host = cpu.route.clone();
        host.resources = cpu_resources.expect("checked host candidate");
        crate::placement::session::select_unobserved(
            operation,
            std::slice::from_ref(&host),
            runmat_execution::WholeWorkConstraint::AnyLegal,
            resources,
            true,
        )
        .map_err(plan_error)?;
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
        let cpu = cpu.into_priced().expect("priced rejection route");
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
                cpu_resources: cpu_resources.as_ref().expect("checked host candidate"),
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
        admitted,
        rejected,
        source_digest,
        residency,
        physical_facts,
        compare_profitability,
        runtime,
        policy,
    })
}
