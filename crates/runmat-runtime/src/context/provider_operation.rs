//! One complete, exact provider operation admitted by the shared whole-work
//! service. Domain owners supply typed query and physical host workspace;
//! children authenticate readbacks, build candidate demands, and own feedback.

use super::{
    host_presentation::{AdmittedHostUploadOutputs, HostPresentationReceipt},
    RuntimeContext,
};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationSupport,
};
use runmat_execution::{Digest, WholeWorkConstraint, WholeWorkPlacementRequest};

pub(crate) fn provider_resource_estimate(
    runtime: Option<&RuntimeContext>,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    query: &runmat_accelerate_api::ProviderFeasibilityQuery,
) -> BuiltinResult<Option<runmat_accelerate_api::ProviderResourceEstimate>> {
    provider_resource_query(runtime, provider, || provider.query_feasibility(query))
}

pub(crate) fn provider_resource_query(
    runtime: Option<&RuntimeContext>,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    query: impl FnOnce() -> runmat_accelerate_api::ProviderFeasibility,
) -> BuiltinResult<Option<runmat_accelerate_api::ProviderResourceEstimate>> {
    check_cancelled(runtime)?;
    let budget = runtime
        .and_then(|runtime| runtime.service_ports().parallel())
        .and_then(|port| port.placement_resources().memory_available_bytes);
    let result = admission::estimate(query(), &provider.placement_resources(), budget)?;
    check_cancelled(runtime)?;
    Ok(result)
}

pub(crate) fn unavailable_provider_route<T>(
    runtime: Option<&RuntimeContext>,
    provider: &dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<Option<T>> {
    check_cancelled(runtime)?;
    let snapshot = provider.placement_resources();
    check_cancelled(runtime)?;
    admission::validate_live(&snapshot)?;
    let budget = runtime
        .and_then(|runtime| runtime.service_ports().parallel())
        .and_then(|port| port.placement_resources().memory_available_bytes);
    if admission::finite_budget(&snapshot, budget) {
        return Err(build_runtime_error(
            "provider route has no proven physical continuation under a finite budget",
        )
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    check_cancelled(runtime)?;
    declined_provider_route(runtime)
}

pub(crate) fn declined_provider_route<T>(
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Option<T>> {
    if runtime
        .and_then(RuntimeContext::acceleration_policy)
        .is_some_and(|policy| !policy.allow_host_fallback)
    {
        return Err(build_runtime_error(
            "provider implementation is unavailable and this invocation forbids host fallback",
        )
        .with_identifier("RunMat:Acceleration:FallbackDisabled")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    Ok(None)
}

mod candidate;
mod admission;
mod readbacks;
mod types;
use candidate::{operation_signature, resource_candidate};
use readbacks::prepare_readbacks;
use types::{
    ConditionalUploadPublication, DirectProviderOutputContract, DirectProviderPublication,
};
pub(crate) use types::{
    ExactConditionalProviderUpload, ExactHostUploadAlternative, ExactProducedProviderReadback,
    ExactProviderOperation, ExactProviderOrHostSelection, ExactProviderReadback,
    ExactProviderReadbackPhase, ProviderPhysicalCalibration, SelectedProviderOperation,
};

pub(crate) fn select_exact_provider_operation(
    runtime: Option<&RuntimeContext>,
    request: ExactProviderOperation<'_>,
) -> BuiltinResult<Option<SelectedProviderOperation>> {
    match select_exact_provider_or_host_operation_with_output_contract(
        runtime,
        request,
        None,
        DirectProviderOutputContract::Fresh,
    )? {
        Some(ExactProviderOrHostSelection::Provider(selected)) => Ok(Some(selected)),
        Some(ExactProviderOrHostSelection::HostUpload(_)) => {
            unreachable!("no host candidate supplied")
        }
        None => Ok(None),
    }
}

pub(crate) fn select_exact_provider_reused_output_operation(
    runtime: Option<&RuntimeContext>,
    request: ExactProviderOperation<'_>,
) -> BuiltinResult<Option<SelectedProviderOperation>> {
    match select_exact_provider_or_host_operation_with_output_contract(
        runtime,
        request,
        None,
        DirectProviderOutputContract::ReusedInputs,
    )? {
        Some(ExactProviderOrHostSelection::Provider(selected)) => Ok(Some(selected)),
        Some(ExactProviderOrHostSelection::HostUpload(_)) => {
            unreachable!("no host candidate supplied")
        }
        None => Ok(None),
    }
}

pub(crate) fn select_exact_provider_or_host_operation(
    runtime: Option<&RuntimeContext>,
    request: ExactProviderOperation<'_>,
    host: Option<ExactHostUploadAlternative<'_>>,
) -> BuiltinResult<Option<ExactProviderOrHostSelection>> {
    select_exact_provider_or_host_operation_with_output_contract(
        runtime,
        request,
        host,
        DirectProviderOutputContract::Fresh,
    )
}

fn select_exact_provider_or_host_operation_with_output_contract(
    runtime: Option<&RuntimeContext>,
    mut request: ExactProviderOperation<'_>,
    mut host: Option<ExactHostUploadAlternative<'_>>,
    output_contract: DirectProviderOutputContract,
) -> BuiltinResult<Option<ExactProviderOrHostSelection>> {
    check_cancelled(runtime)?;
    if runtime
        .and_then(RuntimeContext::acceleration_policy)
        .is_some_and(|policy| !policy.allow_host_fallback)
    {
        // These are provider-owned calls. Disable only their explicit typed
        // CPU fallback alternatives, not unrelated host work or explicit gather.
        request.conditional_host_fallback = None;
        host = None;
    }
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
    let parallel = runtime
        .and_then(|runtime| runtime.service_ports().parallel())
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let Some(estimate) = provider_resource_estimate(runtime, provider, query)? else {
        return declined_provider_route(runtime);
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
    check_cancelled(runtime)?;
    let prepared = prepare_readbacks(
        &request,
        estimate.output_bytes,
        parallel.memory_available_bytes,
    )?;
    check_cancelled(runtime)?;
    let Some(mut prepared) = prepared else {
        return declined_provider_route(runtime);
    };
    // Observation metadata does not grant physical resource admission.
    let provider_digest = Digest::sha256_serialized(&(
        provider.capability_snapshot(),
        &prepared.source_capabilities,
    ))
    .expect("capability snapshot serializes");
    let observation = runtime.and_then(|runtime| {
        let service = runtime.service_ports().placement()?;
        let revision =
            service.whole_work_revision(runtime.program_revision().cloned(), provider_digest)?;
        let ProviderObservationCapability::Supported(overhead) =
            provider.operation_observation_capability(query)
        else {
            return None;
        };
        Some((runtime, service, revision, overhead))
    });
    let (candidate, resources) = resource_candidate(
        runtime,
        &request,
        &estimate,
        observation.as_ref().map(|(_, _, _, overhead)| *overhead),
        &mut prepared,
        parallel,
    );
    if let Some(snapshot) = resources.provider(provider.device_id()) {
        let queue = admission::require_queue_bound(
            snapshot,
            request.output_queue_slots.or(estimate.dispatches),
        );
        if host.is_some() && queue.is_err() {
            // Only the native candidate lacks a finite queue certificate.
            // The caller can still select its independently admitted host
            // route without treating this pure preflight as an effect.
            return declined_provider_route(runtime);
        }
        queue?;
    }
    if resources.cancellation_requested {
        return Err(
            build_runtime_error("provider operation cancelled before admission")
                .with_identifier("RunMat:Cancelled")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        );
    }
    if host.is_none() {
        runmat_execution::single_candidate_resources_admitted(
            runmat_execution::CandidateResourceFacts {
                location: candidate.execution_location,
                output_residency: candidate.output_residency,
                presentation: &candidate.presentation,
                kind: candidate.kind,
                demand: candidate.resources,
                transactional_results: candidate.transactional_results,
            },
            &resources,
            false,
        )
        .map_err(resource_error)?;
    }
    let Some(direct_output) = DirectProviderPublication::from_query(
        query,
        estimate.output_bytes,
        request.already_live_bytes,
        output_contract,
    ) else {
        // A resource-only route is not a validated output-publication receipt.
        if host.is_some() {
            return declined_provider_route(runtime);
        }
        let provider_resources = provider.placement_resources();
        if parallel.memory_available_bytes.is_some()
            || provider_resources.capacity_bytes.is_some()
            || provider_resources.scratch_available_bytes.is_some()
        {
            return Err(build_runtime_error(
                "provider result has no physical publication bound under a finite resource budget",
            )
            .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
            .build());
        }
        return declined_provider_route(runtime);
    };
    let Some((runtime, service, revision, _)) = observation else {
        return declined_provider_route(runtime);
    };
    let mut signature = operation_signature(&request, &prepared, revision);
    if let Some(host) = host.as_ref() {
        signature.runtime_facts = Digest::sha256_serialized(&(
            signature.runtime_facts,
            host.candidate.route_facts,
            Some(host.query),
            Some(&host.output),
            Some(host.requested_bytes),
        ))
        .expect("typed paired operation serializes");
    }
    let mut candidates = vec![candidate.clone()];
    if let Some(host) = host.as_ref() {
        candidates.push(host.candidate.clone());
    }
    let selected = service.select_whole_work(WholeWorkPlacementRequest {
        signature: signature.clone(),
        constraint: if host.is_some() {
            WholeWorkConstraint::AnyLegal
        } else {
            WholeWorkConstraint::ProviderRequired {
                device_id: provider.device_id(),
            }
        },
        candidates,
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if selected.decision.signature != signature {
        return Err(
            build_runtime_error("placement selected an unadmitted operation")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    if let Some(host) = host.as_ref() {
        if selected.decision.selected == host.candidate.selected(request.semantic_operation) {
            let upload = host;
            let metadata_claim = super::host_presentation::selected_upload_metadata_claim(
                runtime,
                std::slice::from_ref(&upload.output),
                &upload.query.operation,
                request.semantic_operation,
            )?;
            let ProviderObservationSupport::Observed(mut receipt) =
                provider.begin_operation_observation(upload.query)
            else {
                return declined_provider_route(Some(runtime));
            };
            receipt
                .require_provider_execution(provider.device_id(), upload.query.operation.clone());
            let presentation = HostPresentationReceipt::new(runtime.clone(), selected, receipt)
                .with_publication_metadata_claim(metadata_claim)
                .with_admitted_upload_outputs(AdmittedHostUploadOutputs {
                    provider,
                    outputs: vec![upload.output.clone()],
                    one_output_requested_bytes: Some(upload.requested_bytes),
                });
            return Ok(Some(ExactProviderOrHostSelection::HostUpload(presentation)));
        }
    }
    if selected.decision.selected != candidate.selected(request.semantic_operation) {
        return Err(
            build_runtime_error("placement selected an unadmitted provider operation")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(query)
    else {
        return declined_provider_route(Some(runtime));
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
    Ok(Some(ExactProviderOrHostSelection::Provider(
        SelectedProviderOperation {
            provider,
            receipt: HostPresentationReceipt::new(runtime.clone(), selected, receipt),
            direct_output,
            conditional_upload: ConditionalUploadPublication::new(
                prepared
                    .conditional_provider_upload
                    .map(|upload| upload.provider_output_requested_bytes),
            ),
            produced_readback: prepared.produced_readback,
        },
    )))
}

fn check_cancelled(runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    if runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(
            build_runtime_error("provider operation cancelled before admission")
                .with_identifier("RunMat:Cancelled")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        );
    }
    Ok(())
}

pub(crate) fn resource_error(
    cause: runmat_execution::ResourceAdmissionError,
) -> crate::RuntimeError {
    let identifier = match cause {
        runmat_execution::ResourceAdmissionError::InsufficientEvidence { .. } => {
            "RunMat:Placement:InsufficientResourceEvidence"
        }
        runmat_execution::ResourceAdmissionError::InvalidSnapshot(_) => {
            "RunMat:Placement:InvalidRequest"
        }
        _ => "RunMat:Placement:NoLegalCandidate",
    };
    build_runtime_error(format!(
        "provider operation resource admission failed: {cause:?}"
    ))
    .with_identifier(identifier)
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .with_source(cause)
    .build()
}
