//! Shared exact host-result upload admission. The caller supplies checked
//! operation-owned geometry and scratch; this boundary asks the owner about
//! its readback host workspace and actual upload route, then submits one
//! whole-work candidate to policy.

use super::{AdmittedHostUploadOutputs, HostPresentationReceipt};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationSupport,
};
use runmat_accelerate_api::{
    class_preserving_upload_query, provider_for_handle, GpuTensorHandle, ProviderFeasibility,
    ProviderOperationIdentity, ProviderRepresentation, ProviderResidency,
};
use runmat_execution::resource::RetainedHostAllocationClaim;
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementResourceSnapshot,
    PlacementSignature, ProviderReadbackDemand, ProviderReadbackDemands, ResourceBoundUnavailable,
    ResourceByteBound, WholeWorkCandidate, WholeWorkConstraint, WholeWorkPlacementRequest,
};
use std::sync::Arc;

/// Pre-admit the dynamic payload cloned for a selected upload publication.
/// The operation name remains in the provider observation after the caller's
/// temporary query receipt and the presentation frame have retired. A single
/// retained claim follows that longest-lived owner; it is not a second grant
/// to execute or upload the operation.
pub(crate) fn selected_upload_metadata_bytes(
    outputs: &[ProviderRepresentation],
    operation: &ProviderOperationIdentity,
    semantic_operation: &str,
) -> BuiltinResult<u64> {
    HostAllocationPlan::retained_metadata_bytes(selected_upload_payload_bytes(
        outputs,
        operation,
        semantic_operation,
    )?)
    .map_err(|cause| {
        build_runtime_error("selected upload publication metadata bound overflows")
            .with_identifier("RunMat:Placement:ResourceOverflow")
            .with_source(cause)
            .build()
    })
}

fn selected_upload_payload_bytes(
    outputs: &[ProviderRepresentation],
    operation: &ProviderOperationIdentity,
    semantic_operation: &str,
) -> BuiltinResult<u64> {
    let shapes = outputs.iter().try_fold(0_u64, |total, output| {
        total.checked_add(
            u64::try_from(output.shape.len())
                .ok()?
                .checked_mul(std::mem::size_of::<u64>() as u64)?,
        )
    });
    let payload = shapes
        .and_then(|bytes| {
            bytes.checked_add(
                u64::try_from(outputs.len())
                    .ok()?
                    .checked_mul(std::mem::size_of::<ProviderRepresentation>() as u64)?,
            )
        })
        .and_then(|bytes| bytes.checked_add(u64::try_from(operation.0.len()).ok()?))
        .and_then(|bytes| bytes.checked_add(u64::try_from(semantic_operation.len()).ok()?))
        .ok_or_else(|| {
            build_runtime_error("selected upload publication metadata bound overflows")
                .with_identifier("RunMat:Placement:ResourceOverflow")
                .build()
        })?;
    Ok(payload)
}

pub(crate) fn selected_upload_metadata_claim(
    runtime: &RuntimeContext,
    outputs: &[ProviderRepresentation],
    operation: &ProviderOperationIdentity,
    semantic_operation: &str,
) -> BuiltinResult<Option<Arc<dyn RetainedHostAllocationClaim>>> {
    let retained = selected_upload_metadata_bytes(outputs, operation, semantic_operation)?;
    let payload = selected_upload_payload_bytes(outputs, operation, semantic_operation)?;
    let plan = HostAllocationPlan::new(0, retained)
        .map_err(|cause| selected_upload_metadata_error(runtime, cause))?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|cause| selected_upload_metadata_error(runtime, cause))?
    else {
        return Ok(None);
    };
    let claim = receipt
        .claim_retained_metadata(runtime, payload)
        .map_err(|cause| selected_upload_metadata_error(runtime, cause))?;
    Ok(Some(claim))
}

fn selected_upload_metadata_error(
    runtime: &RuntimeContext,
    cause: HostAllocationError,
) -> crate::RuntimeError {
    let suffix = cause.runtime_identifier_suffix();
    build_runtime_error("selected upload publication metadata admission failed")
        .with_identifier(format!("{}:{suffix}", runtime.error_namespace()))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(cause)
        .build()
}

pub(crate) struct ExactHostUpload<'a> {
    pub(crate) operation: &'static str,
    pub(crate) variant: &'static str,
    pub(crate) owner: &'a GpuTensorHandle,
    /// Digest of typed input/control/output-mode facts, never a serialized
    /// substitute for the live owner handle used in provider admission.
    pub(crate) domain_facts: Digest,
    pub(crate) outputs: Vec<ProviderRepresentation>,
    /// Host temporaries owned by the operation, excluding source readback
    /// payload and provider-internal readback workspace. Shared admission
    /// queries the owner for those before submitting a host-memory bound.
    pub(crate) host_compute_scratch: ResourceByteBound,
}

/// A host-compute/upload route already quoted by the shared upload candidate
/// owner. This is the one-candidate form when no native provider operation is
/// legal; the paired form is selected by `provider_operation` with the same
/// candidate and publication contract.
pub(crate) fn select_prepared_host_upload(
    runtime: Option<&RuntimeContext>,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    semantic_operation: &'static str,
    variant: &'static str,
    domain_facts: Digest,
    candidate: WholeWorkCandidate,
    query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    output: ProviderRepresentation,
    requested_bytes: u64,
) -> BuiltinResult<Option<HostPresentationReceipt>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(None);
    };
    let Some(revision) = service.whole_work_revision(
        runtime.program_revision().cloned(),
        Digest::sha256_serialized(&provider.capability_snapshot())
            .expect("typed provider snapshot serializes"),
    ) else {
        return Ok(None);
    };
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let signature = PlacementSignature {
        region: None,
        operation: semantic_operation.into(),
        runtime_facts: Digest::sha256_serialized(&(
            variant,
            domain_facts,
            query,
            candidate.route_facts,
            &output,
            requested_bytes,
            provider.device_id(),
        ))
        .expect("typed host candidate serializes"),
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
        candidates: vec![candidate.clone()],
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if selected.decision.signature != signature
        || selected.decision.selected != candidate.selected(semantic_operation)
    {
        return Err(
            build_runtime_error("host upload selection disagrees with admitted route")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let metadata_claim = selected_upload_metadata_claim(
        runtime,
        std::slice::from_ref(&output),
        &query.operation,
        semantic_operation,
    )?;
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(query)
    else {
        return Ok(None);
    };
    receipt.require_provider_execution(provider.device_id(), query.operation.clone());
    Ok(Some(
        HostPresentationReceipt::new(runtime.clone(), selected, receipt)
            .with_publication_metadata_claim(metadata_claim)
            .with_admitted_upload_outputs(AdmittedHostUploadOutputs {
                provider,
                outputs: vec![output],
                one_output_requested_bytes: Some(requested_bytes),
            }),
    ))
}

pub(crate) fn select_exact_host_upload(
    runtime: Option<&RuntimeContext>,
    request: ExactHostUpload<'_>,
) -> BuiltinResult<Option<HostPresentationReceipt>> {
    let owner = request.owner;
    select_exact_host_upload_with_readbacks(runtime, request, std::slice::from_ref(owner))
}

/// The readback set is exactly the resident operands that the host computation
/// will gather. Each independently returned native payload (including aliases)
/// contributes to peak host workspace. The owner remains the original handle
/// selected for result publication, not an automatically prepared replacement.
pub(crate) fn select_exact_host_upload_with_readbacks(
    runtime: Option<&RuntimeContext>,
    request: ExactHostUpload<'_>,
    readbacks: &[GpuTensorHandle],
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
    // A binary host conversion may gather from two independently owned
    // devices. Only the final publication is constrained to `request.owner`.
    let source_providers = readbacks
        .iter()
        .map(|source| {
            provider_for_handle(source).filter(|source_provider| {
                source.device_id() == source_provider.device_id()
                    && source.allocation_lease().validate_live().is_ok()
            })
        })
        .collect::<Option<Vec<_>>>();
    let Some(source_providers) = source_providers.filter(|providers| !providers.is_empty()) else {
        return Err(
            build_runtime_error("host readback source has no live allocation owner")
                .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                .build(),
        );
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
    let admitted_outputs = request.outputs.clone();
    let Some(upload) = class_preserving_upload_query(request.outputs) else {
        return Ok(None);
    };
    let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&upload) else {
        return Ok(None);
    };
    let upload_physical = provider.upload_physical_evidence(&upload);
    if upload_physical.is_some_and(|physical| {
        upload.outputs.len() != 1 || estimate.output_bytes != Some(physical.output_requested_bytes)
    }) {
        return Err(build_runtime_error(
            "host upload physical evidence contradicts its exact one-output estimate",
        )
        .with_identifier("RunMat:Placement:InvalidPhysicalEvidence")
        .build());
    }
    let owner_resources = provider.placement_resources();
    let finite_host_quota = runtime.service_ports().parallel().is_some_and(|parallel| {
        parallel
            .placement_resources()
            .memory_available_bytes
            .is_some()
    });
    if upload_physical.is_none()
        && (finite_host_quota
            || owner_resources.capacity_bytes.is_some()
            || owner_resources.scratch_available_bytes.is_some()
            || owner_resources.queue_limit.is_some())
    {
        return Err(build_runtime_error(
            "host upload has no provider-certified physical publication under a finite resource budget",
        )
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }
    if !matches!(
        provider.operation_observation_capability(&upload),
        ProviderObservationCapability::Supported(_)
    ) {
        return Ok(None);
    }
    let readback_host_workspace = readbacks.iter().zip(&source_providers).try_fold(
        ResourceByteBound::Known(0),
        |total, (source, source_provider)| {
            combine_host_scratch(
                total,
                source_provider.download_numeric_host_workspace_bound(source),
            )
        },
    )?;
    let total_host_scratch =
        combine_host_scratch(request.host_compute_scratch, readback_host_workspace)?;
    // The gathers complete sequentially. Per owner, their peak staging and
    // occupied queue slot are therefore maxima, not a sum of unrelated
    // simultaneously live submissions. Unknown provider staging stays unknown.
    let readback_demands =
        ProviderReadbackDemands::sequential(readbacks.iter().zip(&source_providers).map(
            |(source, source_provider)| {
                let provider_scratch_bytes =
                    source_provider.download_numeric_provider_scratch_bound(source);
                ProviderReadbackDemand {
                    device_id: source_provider.device_id(),
                    provider_scratch_bytes,
                    queue_slots: source_provider.download_numeric_queue_slots(source),
                }
            },
        ))
        .ok_or_else(|| {
            build_runtime_error("host readback exceeds the provider resource set")
                .with_identifier("RunMat:Placement:TooManyProviders")
                .build()
        })?;
    let device_id = provider.device_id();
    // The reusable decision must invalidate when any readback owner's
    // capability changes, not just when the upload owner's capability does.
    let source_capabilities = source_providers
        .iter()
        .map(|source_provider| {
            (
                source_provider.device_id(),
                source_provider.capability_snapshot(),
            )
        })
        .collect::<Vec<_>>();
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&(provider.capability_snapshot(), &source_capabilities))
            .expect("capability snapshots serialize"),
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
                upload_physical,
                readback_host_workspace,
                readback_demands,
                readbacks
                    .iter()
                    .zip(&source_providers)
                    .map(|(source, source_provider)| {
                        (
                            source_provider.device_id(),
                            source.shape(),
                            source.physical().element_type(),
                            source.physical().requested_bytes(),
                            source.view().layout(),
                            runmat_accelerate_api::handle_provenance(source),
                        )
                    })
                    .collect::<Vec<_>>(),
                device_id,
                runmat_accelerate_api::handle_provenance(request.owner),
            ))
            .expect("typed upload route serializes"),
        ),
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    };
    let candidate = upload_candidate(
        device_id,
        transfer_bytes,
        UploadStageResources {
            physical_retained_bytes: upload_physical
                .map(|physical| physical.output_requested_bytes)
                .or(estimate.output_bytes),
            host_workspace: upload_physical.map_or(
                ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
                |physical| physical.host_workspace_bytes,
            ),
            provider_scratch: upload_physical.map_or(
                ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
                |physical| physical.provider_scratch_bytes,
            ),
            queue_slots: upload_physical.map_or(
                u32::try_from(upload.outputs.len()).unwrap_or(u32::MAX),
                |physical| physical.completion_queue_slots,
            ),
        },
        total_host_scratch,
        Digest::sha256(serde_json::to_vec(&upload).expect("typed route serializes")),
        Some(readback_demands),
    );
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let mut provider_resources = vec![provider.placement_resources()];
    for source_provider in &source_providers {
        if !provider_resources
            .iter()
            .any(|resource| resource.device_id == source_provider.device_id())
        {
            provider_resources.push(source_provider.placement_resources());
        }
    }
    provider_resources.sort_by_key(|resource| resource.device_id);
    let digest = Digest::sha256(
        serde_json::to_vec(&(parallel.epoch, &provider_resources))
            .expect("resource snapshot serializes"),
    );
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: provider_resources,
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
    Ok(Some(
        HostPresentationReceipt::new(runtime.clone(), selected, receipt)
            .with_admitted_upload_outputs(AdmittedHostUploadOutputs {
                provider,
                outputs: admitted_outputs,
                one_output_requested_bytes: upload_physical
                    .map(|physical| physical.output_requested_bytes),
            }),
    ))
}

/// The caller's scratch excludes the provider's native readback payload and
/// its internal host temporaries. Unknown on either side remains unknown;
/// only a checked sum of proven bounds can pass a finite host-memory quota.
/// Overflow is a placement error, never an unselected route that could run
/// without admission.
pub(crate) fn combine_host_scratch(
    operation_owned: ResourceByteBound,
    readback_host_workspace: ResourceByteBound,
) -> BuiltinResult<ResourceByteBound> {
    match (operation_owned, readback_host_workspace) {
        (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => left
            .checked_add(right)
            .map(ResourceByteBound::Known)
            .ok_or_else(|| {
                build_runtime_error("host readback workspace byte bound overflows")
                    .with_identifier("RunMat:Placement:ResourceOverflow")
                    .build()
            }),
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => Ok(ResourceByteBound::Unavailable(reason)),
    }
}

#[derive(Clone, Copy)]
pub(crate) struct UploadStageResources {
    pub(crate) physical_retained_bytes: Option<u64>,
    pub(crate) host_workspace: ResourceByteBound,
    pub(crate) provider_scratch: ResourceByteBound,
    pub(crate) queue_slots: u32,
}

pub(crate) fn upload_candidate(
    device_id: u32,
    transfer_bytes: u64,
    upload: UploadStageResources,
    host_compute_scratch: ResourceByteBound,
    route_facts: Digest,
    readbacks: Option<ProviderReadbackDemands>,
) -> WholeWorkCandidate {
    let presentation = match readbacks {
        Some(readbacks) => CandidateOutputPresentation::ProviderReadbackUpload {
            device_id,
            transfer_bytes: Some(transfer_bytes),
            host_result_bytes: Some(ResourceByteBound::Known(transfer_bytes)),
            host_scratch_bytes: upload.host_workspace,
            provider_scratch_bytes: upload.provider_scratch,
            queue_slots: upload.queue_slots,
            preparation: CandidatePreparationState::Ready,
            readbacks: Box::new(readbacks),
        },
        None => CandidateOutputPresentation::ProviderUpload {
            device_id,
            transfer_bytes: Some(transfer_bytes),
            host_result_bytes: Some(ResourceByteBound::Known(transfer_bytes)),
            host_scratch_bytes: upload.host_workspace,
            provider_scratch_bytes: upload.provider_scratch,
            queue_slots: upload.queue_slots,
            preparation: CandidatePreparationState::Ready,
        },
    };
    WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Provider { device_id },
        presentation,
        route_facts,
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            // Physical provider retention may include padding/alignment and
            // is not the host payload or transfer byte count. Unknown stays
            // unknown under a finite provider memory budget.
            retained_bytes: physical_retained_bound(upload.physical_retained_bytes),
            scratch_bytes: host_compute_scratch,
            provider_host_workspace: None,
            provider_input_stages: None,
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
#[path = "admission/tests.rs"]
mod tests;
