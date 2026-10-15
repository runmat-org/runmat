//! One complete, exact provider operation admitted by the shared whole-work
//! service. Domain owners supply typed query and physical host workspace;
//! children authenticate readbacks, build candidate demands, and own feedback.

use super::{host_presentation::HostPresentationReceipt, RuntimeContext};
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
    Ok(None)
}

mod candidate;
mod admission;
mod readbacks;
mod types;
use candidate::{operation_signature, resource_candidate};
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
    check_cancelled(runtime)?;
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
    check_cancelled(runtime)?;
    let prepared = prepare_readbacks(
        &request,
        estimate.output_bytes,
        parallel.memory_available_bytes,
    )?;
    check_cancelled(runtime)?;
    let Some(mut prepared) = prepared else {
        return Ok(None);
    };
    // Observation metadata does not grant physical resource admission.
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&(
            provider.capability_snapshot(),
            &prepared.source_capabilities,
        ))
        .expect("capability snapshot serializes"),
    );
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
        admission::require_queue_bound(
            snapshot,
            request.output_queue_slots.or(estimate.dispatches),
        )?;
    }
    if resources.cancellation_requested {
        return Err(
            build_runtime_error("provider operation cancelled before admission")
                .with_identifier("RunMat:Cancelled")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        );
    }
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
    let Some(direct_output) =
        DirectProviderPublication::new(query.outputs.len(), estimate.output_bytes)
    else {
        // A resource-only route is not a validated output-publication receipt.
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
        return Ok(None);
    };
    let Some((runtime, service, revision, _)) = observation else {
        return Ok(None);
    };
    let signature = operation_signature(&request, &prepared, revision);
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
