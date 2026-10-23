//! CPU observation after a provider feasibility or resource-estimate rejection.
//! There is no provider cost candidate in this graph.
use runmat_accelerate_api::{AccelProvider, ProviderFeasibilityQuery, ProviderRejectionCode};
use runmat_execution::{
    CandidateResourceDemand, Digest, ExecutionCandidateDescriptor, PlacementGraph,
    PlacementGraphCandidate, PlacementGraphLimits, PlacementGraphNode, PlacementPlanRequest,
    PlacementSignature,
};
use runmat_runtime::context::RuntimeContext;

use super::{
    observation_revision, observation_semantic_identity, session_facts,
    session_provider_provenance, signature, LocalPlacementFeedback,
    LocalPlacementFeedbackAuthority, TotalCandidateObservationContract,
};
use crate::placement::{residency::ResidencySummary, PlacementPolicy};

pub(super) struct RejectedCpuFeedbackRequest<'a> {
    pub(super) runtime: Option<&'a RuntimeContext>,
    pub(super) compare_profitability: bool,
    pub(super) operation: &'a str,
    pub(super) semantic_identity: Option<Digest>,
    pub(super) observation_contract: TotalCandidateObservationContract,
    pub(super) prepared_authority:
        Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
    pub(super) vm_right: Option<&'a runmat_runtime::context::VmFusionCompletionRight>,
    pub(super) native_right:
        Option<&'a runmat_runtime::numeric_region::service::NativeNumericRegionSelectionAuthority>,
    pub(super) query: &'a ProviderFeasibilityQuery,
    pub(super) code: ProviderRejectionCode,
    pub(super) cpu: &'a ExecutionCandidateDescriptor,
    pub(super) cpu_resources: &'a CandidateResourceDemand,
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
    let revision = observation_revision(
        runtime,
        request.observation_contract,
        request.prepared_authority,
        request.vm_right,
        request.native_right,
        provider_digest,
        request.policy,
    )?;
    let signature = PlacementSignature {
        region: request.cpu.region,
        operation: request.operation.into(),
        runtime_facts: signature::with_total_candidate_observation_contract(
            signature::with_semantic_identity(
                signature::with_cpu_candidate(
                    signature::rejected_runtime_facts(
                        request.query,
                        request.code,
                        request.cpu.preparation,
                        request.residency,
                        &resources,
                        host_epoch,
                    ),
                    request.cpu,
                    request.cpu_resources,
                ),
                observation_semantic_identity(
                    request.observation_contract,
                    request.semantic_identity,
                    provider_digest,
                ),
            ),
            request.observation_contract,
        ),
        provider_provenance: session_provider_provenance(&revision, provider_digest),
        graph_output_authority: None,
        revision,
    };
    let graph = PlacementGraph {
        nodes: vec![PlacementGraphNode {
            identity: request.operation.into(),
            candidates: vec![PlacementGraphCandidate {
                resources: *request.cpu_resources,
                descriptor: request.cpu.clone(),
                output_facts: None,
                transactional_results: true,
            }],
        }],
        edges: Vec::new(),
    };
    // Observation admission is advisory. A session that cannot plan this
    // CPU-only graph must not prevent the already-admitted host operation.
    let selection = service
        .plan_with_observation_permit(PlacementPlanRequest {
            signature: signature.clone(),
            graph,
            limits: PlacementGraphLimits::default(),
            resources,
            deterministic: false,
            require_transactional_results: true,
        })
        .ok()?;
    let selected = selection.decision.selections.first()?;
    if selected.candidate != request.cpu.identity || selected.kind != request.cpu.kind {
        return None;
    }
    Some(Box::new(LocalPlacementFeedback {
        runtime: runtime.clone(),
        signature,
        candidate: request.cpu.identity.clone(),
        kind: request.cpu.kind,
        authority: LocalPlacementFeedbackAuthority::Graph(selection.observation_permit?),
    }))
}
