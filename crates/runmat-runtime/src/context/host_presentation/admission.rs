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
    provider_for_handle, GpuTensorHandle, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation, ProviderResidency,
    ProviderWorkload,
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
    let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&upload) else {
        return Ok(None);
    };
    if !matches!(
        provider.operation_observation_capability(&upload),
        ProviderObservationCapability::Supported(_)
    ) {
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
    let candidate = upload_candidate(
        device_id,
        transfer_bytes,
        estimate.output_bytes,
        request.host_compute_scratch,
        u32::try_from(upload.outputs.len()).unwrap_or(u32::MAX),
        Digest::sha256(serde_json::to_vec(&upload).expect("typed route serializes")),
    );
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

fn upload_candidate(
    device_id: u32,
    transfer_bytes: u64,
    physical_retained_bytes: Option<u64>,
    host_compute_scratch: ResourceByteBound,
    queue_slots: u32,
    route_facts: Digest,
) -> WholeWorkCandidate {
    WholeWorkCandidate {
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
            queue_slots,
            preparation: CandidatePreparationState::Ready,
        },
        route_facts,
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            // Physical provider retention may include padding/alignment and
            // is not the host payload or transfer byte count. Unknown stays
            // unknown under a finite provider memory budget.
            retained_bytes: physical_retained_bound(physical_retained_bytes),
            scratch_bytes: host_compute_scratch,
            queue_slots: 0,
            peak_live: None,
        },
        transactional_results: true,
    }
}

fn physical_retained_bound(output_bytes: Option<u64>) -> ResourceByteBound {
    output_bytes
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoRetainedValueBound,
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{PlacementRevision, ProviderResourceSnapshot};

    fn budget_request(logical: u64, physical: u64, capacity: u64) -> WholeWorkPlacementRequest {
        let digest = Digest::sha256(b"physical-host-upload-budget");
        WholeWorkPlacementRequest {
            signature: PlacementSignature {
                region: None,
                operation: "imhist".into(),
                runtime_facts: digest,
                revision: PlacementRevision {
                    program: None,
                    catalog: digest,
                    compiler: digest,
                    provider: digest,
                    policy: digest,
                },
            },
            constraint: WholeWorkConstraint::AnyLegal,
            candidates: vec![upload_candidate(
                17,
                logical,
                Some(physical),
                ResourceByteBound::Known(0),
                1,
                digest,
            )],
            resources: PlacementResourceSnapshot {
                cpu_millicores_available: 1_000,
                memory_available_bytes: Some(1_024),
                cancellation_requested: false,
                providers: vec![ProviderResourceSnapshot {
                    device_id: 17,
                    capacity_bytes: Some(capacity),
                    scratch_available_bytes: None,
                    ..Default::default()
                }],
                epoch: 0,
            },
            deterministic: true,
            require_transactional_results: false,
        }
    }

    #[test]
    fn shared_service_budgets_physical_retention_not_logical_payload() {
        let session = runmat_accelerate::placement::PlacementSession::new(
            runmat_accelerate::placement::PlacementSessionConfig::default(),
        );
        assert!(session
            .select_whole_work(budget_request(32, 40, 39))
            .is_err());
        assert!(session
            .select_whole_work(budget_request(32, 40, 40))
            .is_ok());
        // Zero logical elements can still require a nonzero provider buffer.
        assert!(session.select_whole_work(budget_request(0, 8, 7)).is_err());
        assert!(session.select_whole_work(budget_request(0, 8, 8)).is_ok());
    }

    #[test]
    fn provider_retention_is_not_logical_transfer_or_unknown_as_zero() {
        let logical_transfer = 32;
        assert_eq!(
            physical_retained_bound(Some(40)),
            ResourceByteBound::Known(40)
        );
        assert_ne!(
            physical_retained_bound(Some(40)),
            ResourceByteBound::Known(logical_transfer)
        );
        // A zero-element logical payload can still retain a nonzero provider
        // allocation floor; never replace the reported physical result by 0.
        assert_eq!(
            physical_retained_bound(Some(8)),
            ResourceByteBound::Known(8)
        );
        assert_eq!(
            physical_retained_bound(None),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound)
        );
    }
}
