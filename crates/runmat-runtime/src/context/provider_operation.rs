//! One complete, exact provider operation admitted by the shared whole-work
//! service. Domain owners supply typed query and physical host workspace;
//! children authenticate readbacks, build candidate demands, and own feedback.

use super::{host_presentation::HostPresentationReceipt, RuntimeContext};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationSupport,
};
use runmat_accelerate_api::ProviderFeasibility;
use runmat_execution::{Digest, WholeWorkConstraint, WholeWorkPlacementRequest};

mod candidate;
mod readbacks;
mod types;
use candidate::admitted_candidate;
use readbacks::prepare_readbacks;
use types::{ConditionalUploadPublication, DirectProviderPublication};
pub(crate) use types::{
    ExactConditionalProviderUpload, ExactProducedProviderReadback, ExactProviderOperation,
    ExactProviderReadback, ExactProviderReadbackPhase, SelectedProviderOperation,
};

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
    if request.conditional_host_fallback.is_some() && request.conditional_provider_upload.is_some()
    {
        return Err(build_runtime_error(
            "provider operation has conflicting conditional publications",
        )
        .with_identifier("RunMat:Placement:InvalidConditionalPresentation")
        .build());
    }
    let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(query) else {
        return Ok(None);
    };
    let Some(direct_output) =
        DirectProviderPublication::new(query.outputs.len(), estimate.output_bytes)
    else {
        // No selected route may claim a physical retained result from its
        // logical output geometry. The provider call remains available to the
        // caller without a qualified placement receipt only when no finite
        // memory resource contract would thereby be bypassed.
        let resources = provider.placement_resources();
        let finite_host = runtime
            .service_ports()
            .parallel()
            .is_some_and(|port| port.placement_resources().memory_available_bytes.is_some());
        if finite_host
            || resources.capacity_bytes.is_some()
            || resources.scratch_available_bytes.is_some()
        {
            return Err(build_runtime_error(
                "provider result has no physical publication bound under a finite resource budget",
            )
            .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
            .build());
        }
        return Ok(None);
    };
    if request
        .conditional_provider_upload
        .as_ref()
        .is_some_and(|upload| estimate.output_bytes != Some(upload.provider_output_requested_bytes))
    {
        return Err(build_runtime_error(
            "conditional provider upload has a different physical output bound",
        )
        .with_identifier("RunMat:Placement:InvalidConditionalPresentation")
        .build());
    }
    let ProviderObservationCapability::Supported(observation) =
        provider.operation_observation_capability(query)
    else {
        return Ok(None);
    };
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let Some(mut prepared) = prepare_readbacks(
        &request,
        estimate.output_bytes,
        parallel.memory_available_bytes,
    )?
    else {
        return Ok(None);
    };
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&(
            provider.capability_snapshot(),
            &prepared.source_capabilities,
        ))
        .expect("capability snapshot serializes"),
    );
    let Some(revision) =
        service.whole_work_revision(runtime.program_revision().cloned(), provider_digest)
    else {
        return Ok(None);
    };
    let (signature, candidate, resources) = admitted_candidate(
        runtime,
        &request,
        &estimate,
        observation,
        &mut prepared,
        revision,
        parallel,
    );
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
        || selected.decision.selected != candidate.selected(request.semantic_operation)
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
    if prepared.conditional_provider_upload.is_some() {
        receipt.allow_conditional_provider_upload();
    }
    if request.provider_host_mediated {
        receipt.allow_provider_host_mediated();
    }
    Ok(Some(SelectedProviderOperation {
        provider,
        receipt: HostPresentationReceipt::new(runtime.clone(), selected, receipt),
        direct_output,
        conditional_upload: ConditionalUploadPublication::new(
            prepared
                .conditional_provider_upload
                .map(|upload| upload.provider_output_requested_bytes),
        ),
        produced_readback: prepared.produced_readback,
    }))
}
