//! CPU observation after a provider feasibility or resource-estimate rejection.
//! There is no provider cost candidate in this graph.
use runmat_accelerate_api::{AccelProvider, ProviderFeasibilityQuery, ProviderRejectionCode};
use runmat_execution::{
    CandidateResourceDemand, Digest, ExecutionCandidateDescriptor, PlacementGraph,
    PlacementGraphCandidate, PlacementGraphLimits, PlacementGraphNode, PlacementPlanRequest,
    PlacementSignature, ResourceBoundUnavailable, ResourceByteBound,
};
use runmat_runtime::context::RuntimeContext;

use super::{session_facts, signature, LocalPlacementFeedback};
use crate::placement::{residency::ResidencySummary, PlacementPolicy};

pub(super) struct RejectedCpuFeedbackRequest<'a> {
    pub(super) runtime: Option<&'a RuntimeContext>,
    pub(super) compare_profitability: bool,
    pub(super) operation: &'a str,
    pub(super) semantic_identity: Option<Digest>,
    pub(super) query: &'a ProviderFeasibilityQuery,
    pub(super) code: ProviderRejectionCode,
    pub(super) cpu: &'a ExecutionCandidateDescriptor,
    pub(super) residency: &'a ResidencySummary,
    pub(super) policy: PlacementPolicy,
}

pub(super) fn rejected_cpu_feedback(
    provider: &dyn AccelProvider,
    request: RejectedCpuFeedbackRequest<'_>,
) -> Option<Box<LocalPlacementFeedback>> {
    if !request.compare_profitability {
        return None;
    }
    let runtime = request.runtime?;
    let service = runtime.service_ports().placement()?;
    let (resources, provider_digest, host_epoch, _) = session_facts(provider, runtime);
    let signature = PlacementSignature {
        region: request.cpu.region,
        operation: request.operation.into(),
        runtime_facts: signature::with_semantic_identity(
            signature::rejected_runtime_facts(
                request.query,
                request.code,
                request.cpu.preparation,
                request.residency,
                &resources,
                host_epoch,
            ),
            request.semantic_identity,
        ),
        revision: super::super::revision::local(
            runtime.program_revision().cloned(),
            provider_digest,
            request.policy,
        ),
    };
    let graph = PlacementGraph {
        nodes: vec![PlacementGraphNode {
            identity: request.operation.into(),
            candidates: vec![PlacementGraphCandidate {
                resources: CandidateResourceDemand {
                    cpu_millicores: 1_000,
                    retained_bytes: ResourceByteBound::Unavailable(
                        ResourceBoundUnavailable::NoRetainedValueBound,
                    ),
                    scratch_bytes: request.cpu.cost.scratch_bytes.map_or(
                        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
                        Into::into,
                    ),
                    queue_slots: 0,
                    provider_host_workspace: None,
                    peak_live: None,
                },
                descriptor: request.cpu.clone(),
                transactional_results: true,
            }],
        }],
        edges: Vec::new(),
    };
    // Observation admission is advisory. A session that cannot plan this
    // CPU-only graph must not prevent the already-admitted host operation.
    let decision = service
        .plan(PlacementPlanRequest {
            signature: signature.clone(),
            graph,
            limits: PlacementGraphLimits::default(),
            resources,
            deterministic: false,
            require_transactional_results: true,
        })
        .ok()?;
    let selected = decision.selections.first()?;
    if selected.candidate != request.cpu.identity || selected.kind != request.cpu.kind {
        return None;
    }
    Some(Box::new(LocalPlacementFeedback {
        runtime: runtime.clone(),
        signature,
        candidate: request.cpu.identity.clone(),
        kind: request.cpu.kind,
    }))
}
