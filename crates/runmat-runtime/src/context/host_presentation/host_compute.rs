//! Select host publication or a prequoted upload for the *same* host
//! computation. No provider generation or input readback is implied.

use super::{
    selected_upload_metadata_claim, AdmittedHostUploadOutputs, HostPresentationReceipt,
    HostSinkReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationSupport,
};
use runmat_accelerate_api::{
    provider_for_handle, same_provider_instance, GpuTensorHandle, ProviderFeasibilityQuery,
    ProviderRepresentation,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementResourceSnapshot,
    PlacementSignature, ResourceByteBound, WholeWorkCandidate, WholeWorkConstraint,
    WholeWorkPlacementRequest,
};

pub(crate) struct ExactHostComputedUpload<'a> {
    pub(crate) candidate: &'a WholeWorkCandidate,
    pub(crate) query: &'a ProviderFeasibilityQuery,
    pub(crate) output: &'a ProviderRepresentation,
    pub(crate) requested_bytes: u64,
}

pub(crate) struct ExactHostComputedResult<'a> {
    pub(crate) operation: &'static str,
    pub(crate) variant: &'static str,
    pub(crate) source: &'a GpuTensorHandle,
    pub(crate) domain_facts: Digest,
    pub(crate) host_result_bytes: ResourceByteBound,
    pub(crate) host_compute_scratch: ResourceByteBound,
    pub(crate) upload: Option<ExactHostComputedUpload<'a>>,
}

pub(crate) enum HostComputedSelection {
    Host(Option<HostSinkReceipt>),
    Upload(HostPresentationReceipt),
}

/// Both routes consume one domain-owned host result. A resident prototype
/// contributes type/intent and exact owner identity, never a fake download.
pub(crate) fn select_host_computed_result(
    runtime: Option<&RuntimeContext>,
    request: ExactHostComputedResult<'_>,
) -> BuiltinResult<HostComputedSelection> {
    let Some(provider) = provider_for_handle(request.source).filter(|provider| {
        provider.device_id() == request.source.device_id()
            && request.source.allocation_lease().validate_live().is_ok()
    }) else {
        return Err(
            build_runtime_error("host-computed result has no live prototype owner")
                .with_identifier("RunMat:Placement:InvalidSourceOwner")
                .build(),
        );
    };
    let (Some(runtime), Some(service)) = (
        runtime,
        runtime.and_then(|runtime| runtime.service_ports().placement()),
    ) else {
        return Ok(HostComputedSelection::Host(None));
    };
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let host = WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Host,
        presentation: CandidateOutputPresentation::Direct,
        route_facts: Digest::sha256_serialized(&(
            request.variant,
            request.domain_facts,
            request.host_result_bytes,
            request.host_compute_scratch,
        ))
        .expect("typed host result serializes"),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            retained_bytes: request.host_result_bytes,
            scratch_bytes: request.host_compute_scratch,
            provider_host_workspace: None,
            provider_input_stages: None,
            queue_slots: 0,
            peak_live: None,
        },
        transactional_results: true,
    };
    let upload = request.upload.filter(|upload| {
        upload.candidate.output_residency
            == CandidateOutputResidency::Provider {
                device_id: provider.device_id(),
            }
            && upload.output.checked_byte_len().is_some()
            && provider.operation_observation_capability(upload.query)
                != ProviderObservationCapability::Unsupported
    });
    let mut candidates = vec![host.clone()];
    if let Some(upload) = upload.as_ref() {
        candidates.push((*upload.candidate).clone());
    }
    let provider_digest = Digest::sha256_serialized(&provider.capability_snapshot())
        .expect("provider capability serializes");
    let Some(revision) =
        service.whole_work_revision(runtime.program_revision().cloned(), provider_digest)
    else {
        return Ok(HostComputedSelection::Host(None));
    };
    let signature = PlacementSignature {
        region: None,
        operation: request.operation.into(),
        runtime_facts: Digest::sha256_serialized(&(
            request.variant,
            request.domain_facts,
            &host.route_facts,
            upload
                .as_ref()
                .map(|upload| (&upload.candidate.route_facts, upload.query, upload.output)),
        ))
        .expect("typed host-computed routes serialize"),
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    };
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: vec![provider.placement_resources()],
        epoch: parallel.epoch,
    };
    let selected = service.select_whole_work(WholeWorkPlacementRequest {
        signature: signature.clone(),
        constraint: WholeWorkConstraint::AnyLegal,
        candidates,
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if selected.decision.signature != signature {
        return Err(
            build_runtime_error("host-computed selection lost its typed request")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    if selected.decision.selected == host.selected(request.operation) {
        return Ok(HostComputedSelection::Host(Some(HostSinkReceipt::new(
            runtime.clone(),
            selected,
        ))));
    }
    let Some(upload) = upload.filter(|upload| {
        selected.decision.selected == upload.candidate.selected(request.operation)
    }) else {
        return Err(
            build_runtime_error("host-computed selection chose an unadmitted route")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    };
    if !provider_for_handle(request.source)
        .is_some_and(|current| same_provider_instance(current, provider))
        || request.source.allocation_lease().validate_live().is_err()
    {
        return Err(
            build_runtime_error("host-computed prototype owner changed during selection")
                .with_identifier("RunMat:Placement:InvalidSourceOwner")
                .build(),
        );
    }
    let metadata_claim = selected_upload_metadata_claim(
        runtime,
        std::slice::from_ref(upload.output),
        &upload.query.operation,
        request.operation,
    )?;
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(upload.query)
    else {
        return Err(
            build_runtime_error("selected host upload lost provider observation")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    };
    receipt.require_provider_execution(provider.device_id(), upload.query.operation.clone());
    Ok(HostComputedSelection::Upload(
        HostPresentationReceipt::new(runtime.clone(), selected, receipt)
            .with_publication_metadata_claim(metadata_claim)
            .with_admitted_upload_outputs(AdmittedHostUploadOutputs {
                provider,
                outputs: vec![upload.output.clone()],
                one_output_requested_bytes: Some(upload.requested_bytes),
            }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_presentation::tests::Witness;
    use crate::context::physical_test_support::OwnedPhysicalProvider;
    use runmat_accelerate_api::ThreadProviderGuard;
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::rc::Rc;

    #[test]
    fn host_computed_route_uses_no_prototype_readback_or_gpu_draw() {
        let provider: &'static OwnedPhysicalProvider =
            Box::leak(Box::new(OwnedPhysicalProvider::new()));
        let _owner = ThreadProviderGuard::set(Some(provider));
        let allocation = provider.allocate_f64(&[2, 2]);
        let witness = Rc::new(Witness::default());
        let selected = select_host_computed_result(
            Some(&witness.context()),
            ExactHostComputedResult {
                operation: "rand",
                variant: "automatic-like-host-random",
                source: &allocation.handle,
                domain_facts: Digest::sha256(b"single-route-host-result"),
                host_result_bytes: ResourceByteBound::Known(32),
                host_compute_scratch: ResourceByteBound::Known(0),
                upload: None,
            },
        )
        .expect("host result selected without readback capability");
        let HostComputedSelection::Host(Some(receipt)) = selected else {
            panic!("host route required");
        };
        receipt.finish(true);
        assert_eq!(witness.selections(), 1);
        assert_eq!(witness.feedback().len(), 1);

        let cancellation = CancellationSource::new();
        let cancelled_runtime = witness
            .context()
            .with_cancellation_token(cancellation.token());
        let selected = select_host_computed_result(
            Some(&cancelled_runtime),
            ExactHostComputedResult {
                operation: "rand",
                variant: "automatic-like-host-random",
                source: &allocation.handle,
                domain_facts: Digest::sha256(b"cancelled-host-result"),
                host_result_bytes: ResourceByteBound::Known(32),
                host_compute_scratch: ResourceByteBound::Known(0),
                upload: None,
            },
        )
        .expect("host result selected before cancellation");
        let HostComputedSelection::Host(Some(receipt)) = selected else {
            panic!("host route required");
        };
        cancellation.cancel(CancellationReason::User);
        receipt.finish(true);
        assert_eq!(witness.feedback().len(), 1);
    }
}
