//! Shared exact host-result upload admission. The caller supplies checked
//! operation-owned geometry and scratch; this boundary asks the owner about
//! its actual upload route and submits one whole-work candidate to policy.

use super::HostPresentationReceipt;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationSupport,
};
use runmat_accelerate_api::{
    provider_for_handle, GpuTensorHandle, ProviderFeasibilityQuery, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderWorkload,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementResourceSnapshot,
    PlacementSignature, ResourceBoundUnavailable, ResourceByteBound, WholeWorkCandidate,
    WholeWorkConstraint, WholeWorkPlacementRequest,
};

pub(crate) struct ExactHostUpload<'a> {
    pub(crate) operation: &'static str,
    pub(crate) variant: &'static str,
    pub(crate) owner: &'a GpuTensorHandle,
    /// Digest of typed input/control/output-mode facts, never a serialized
    /// substitute for the live owner handle used in provider admission.
    pub(crate) domain_facts: Digest,
    pub(crate) outputs: Vec<ProviderRepresentation>,
    pub(crate) host_compute_scratch: ResourceByteBound,
}

pub(crate) fn select_exact_host_upload(
    runtime: Option<&RuntimeContext>,
    request: ExactHostUpload<'_>,
) -> BuiltinResult<Option<HostPresentationReceipt>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(None);
    };
    let Some(provider) = provider_for_handle(request.owner) else {
        return Ok(None);
    };
    if request.outputs.is_empty()
        || request.outputs.iter().any(|output| {
            output.residency != ProviderResidency::Device || output.checked_byte_len().is_none()
        })
    {
        return Ok(None);
    }
    let Some(transfer_bytes) = request.outputs.iter().try_fold(0_u64, |total, output| {
        total.checked_add(output.checked_byte_len()?)
    }) else {
        return Ok(None);
    };
    let requested_outputs = u32::try_from(request.outputs.len()).ok();
    let upload = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.upload"),
        family: ProviderOperationFamily::Upload,
        inputs: request
            .outputs
            .iter()
            .cloned()
            .map(|mut input| {
                input.residency = ProviderResidency::Host;
                input
            })
            .collect(),
        outputs: request.outputs,
        workload: ProviderWorkload {
            requested_outputs,
            ..ProviderWorkload::default()
        },
    };
    if !provider.query_feasibility(&upload).is_supported()
        || !matches!(
            provider.operation_observation_capability(&upload),
            ProviderObservationCapability::Supported(_)
        )
    {
        return Ok(None);
    }
    let device_id = provider.device_id();
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
        operation: request.operation.into(),
        runtime_facts: Digest::sha256(
            serde_json::to_vec(&(
                request.variant,
                request.domain_facts,
                &upload,
                device_id,
                runmat_accelerate_api::handle_provenance(request.owner),
            ))
            .expect("typed upload route serializes"),
        ),
        revision,
    };
    let candidate = WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Provider { device_id },
        presentation: CandidateOutputPresentation::ProviderUpload {
            device_id,
            transfer_bytes: Some(transfer_bytes),
            host_result_bytes: Some(ResourceByteBound::Known(transfer_bytes)),
            host_scratch_bytes: ResourceByteBound::Known(transfer_bytes),
            provider_scratch_bytes: ResourceByteBound::Unavailable(
                ResourceBoundUnavailable::NoScratchBound,
            ),
            queue_slots: u32::try_from(upload.outputs.len()).unwrap_or(u32::MAX),
            preparation: CandidatePreparationState::Ready,
        },
        route_facts: Digest::sha256(serde_json::to_vec(&upload).expect("typed route serializes")),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            retained_bytes: ResourceByteBound::Known(transfer_bytes),
            scratch_bytes: request.host_compute_scratch,
            queue_slots: 0,
            peak_live: None,
        },
        transactional_results: true,
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
        constraint: WholeWorkConstraint::AnyLegal,
        candidates: vec![candidate.clone()],
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if selected.decision.signature != signature
        || selected.decision.selected != candidate.selected(request.operation)
    {
        return Err(
            build_runtime_error("placement selected an unadmitted host upload route")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(&upload)
    else {
        return Ok(None);
    };
    receipt.require_provider_execution(device_id, upload.operation);
    Ok(Some(HostPresentationReceipt::new(
        runtime.clone(),
        selected,
        receipt,
    )))
}
