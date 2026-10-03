//! One complete, exact provider operation admitted by the shared whole-work
//! service. Domain owners supply the typed query and physical host workspace;
//! this boundary owns policy selection, the provider fence, and feedback.

use super::{host_presentation::HostPresentationReceipt, RuntimeContext};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationSupport, ProviderOperationRecorder,
};
use runmat_accelerate_api::{AccelProvider, ProviderFeasibility, ProviderFeasibilityQuery};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePeakLiveDemand,
    CandidatePreparationState, CandidateResourceDemand, Digest, ExecutionCandidateKind,
    PlacementResourceSnapshot, PlacementSignature, ResourceBoundUnavailable, ResourceByteBound,
    WholeWorkCandidate, WholeWorkConstraint, WholeWorkPlacementRequest,
};

pub(crate) struct ExactProviderOperation<'a> {
    pub(crate) provider: &'static dyn AccelProvider,
    pub(crate) query: &'a ProviderFeasibilityQuery,
    pub(crate) variant: &'static str,
    /// Typed controls and original operand provenance. Never a textual
    /// substitute for the live handles that the domain owner authenticates.
    pub(crate) domain_facts: Digest,
    /// Peak operation-owned host workspace, including any work before the
    /// provider hook but excluding provider internal host workspace.
    pub(crate) host_scratch: ResourceByteBound,
    pub(crate) host_cpu_millicores: u32,
    /// Physical bytes of authenticated borrowed inputs already represented
    /// in the provider's live-allocation resource snapshot.
    pub(crate) already_live_bytes: u64,
    /// The public form permits a host result only after this provider hook
    /// returns Unsupported. This is one conditional route, not an equivalent
    /// host alternative offered to placement policy.
    pub(crate) conditional_host_fallback: Option<ResourceByteBound>,
}

pub(crate) struct SelectedProviderOperation {
    provider: &'static dyn AccelProvider,
    receipt: HostPresentationReceipt,
}

impl SelectedProviderOperation {
    pub(crate) fn recorder(&self) -> ProviderOperationRecorder {
        self.receipt.recorder()
    }

    pub(crate) fn provider(&self) -> &'static dyn AccelProvider {
        self.provider
    }

    pub(crate) fn finish(
        self,
        published: Result<bool, runmat_accelerate_api::observation::ProviderObservationFailure>,
    ) {
        self.receipt.finish(published);
    }

    pub(crate) fn finish_conditional_host_fallback(self) {
        self.receipt.finish_conditional_host_fallback();
    }
}

pub(crate) fn select_exact_provider_operation(
    runtime: Option<&RuntimeContext>,
    request: ExactProviderOperation<'_>,
) -> BuiltinResult<Option<SelectedProviderOperation>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(None);
    };
    let provider = request.provider;
    let query = request.query;
    let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(query) else {
        return Ok(None);
    };
    let ProviderObservationCapability::Supported(observation) =
        provider.operation_observation_capability(query)
    else {
        return Ok(None);
    };
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&provider.capability_snapshot())
            .expect("capability snapshot serializes"),
    );
    let Some(revision) =
        service.whole_work_revision(runtime.program_revision().cloned(), provider_digest)
    else {
        return Ok(None);
    };
    let signature = PlacementSignature {
        region: None,
        operation: query.operation.0.clone(),
        runtime_facts: Digest::sha256(
            serde_json::to_vec(&(
                request.variant,
                request.domain_facts,
                query,
                provider.device_id(),
            ))
            .expect("typed provider operation serializes"),
        ),
        revision,
    };
    // Completion fences coexist with the operation's output/temporary peak.
    // An overflow is unknown physical demand, never a wrapped small bound.
    let peak_live = estimate
        .transient_bytes
        .and_then(|bytes| bytes.checked_add(observation.requested_peak_buffer_bytes))
        .map(|peak_bytes| CandidatePeakLiveDemand {
            peak_bytes,
            already_live_bytes: request.already_live_bytes,
        });
    let candidate = WholeWorkCandidate {
        kind: ExecutionCandidateKind::ProviderOperation,
        execution_location: CandidateExecutionLocation::Provider {
            device_id: provider.device_id(),
        },
        preparation: CandidatePreparationState::Ready,
        output_residency: if let Some(host_bytes) = request.conditional_host_fallback {
            CandidateOutputResidency::ConditionalHostOrProvider {
                device_id: provider.device_id(),
                host_bytes,
            }
        } else {
            CandidateOutputResidency::Provider {
                device_id: provider.device_id(),
            }
        },
        presentation: CandidateOutputPresentation::Direct,
        route_facts: Digest::sha256(
            serde_json::to_vec(&(query, &estimate, request.conditional_host_fallback))
                .expect("provider route serializes"),
        ),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: request.host_cpu_millicores,
            retained_bytes: estimate
                .output_bytes
                .map(ResourceByteBound::Known)
                .unwrap_or(ResourceByteBound::Unavailable(
                    ResourceBoundUnavailable::NoRetainedValueBound,
                )),
            scratch_bytes: ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            provider_host_workspace: Some(request.host_scratch),
            provider_input_stages: None,
            queue_slots: estimate.dispatches.unwrap_or(1),
            peak_live,
        },
        transactional_results: provider
            .capability_snapshot()
            .concurrency
            .transactional_results,
    };
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let provider_resource = provider.placement_resources();
    let digest = Digest::sha256(
        serde_json::to_vec(&(parallel.epoch, &provider_resource))
            .expect("resource snapshot serializes"),
    );
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: vec![provider_resource],
        epoch: u64::from_le_bytes(digest.bytes()[..8].try_into().expect("digest length")),
    };
    let selected = service.select_whole_work(WholeWorkPlacementRequest {
        signature: signature.clone(),
        constraint: WholeWorkConstraint::ProviderRequired {
            device_id: provider.device_id(),
        },
        candidates: vec![candidate.clone()],
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if selected.decision.signature != signature
        || selected.decision.selected != candidate.selected(&query.operation.0)
    {
        return Err(
            build_runtime_error("placement selected an unadmitted provider operation")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(query)
    else {
        return Ok(None);
    };
    receipt.require_provider_execution(provider.device_id(), query.operation.clone());
    if request.conditional_host_fallback.is_some() {
        receipt.allow_conditional_host_fallback();
    }
    Ok(Some(SelectedProviderOperation {
        provider,
        receipt: HostPresentationReceipt::new(runtime.clone(), selected, receipt),
    }))
}
