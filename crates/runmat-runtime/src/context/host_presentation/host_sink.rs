//! Whole-call host publication after an awaited resident readback. The
//! receipt times preparation, download, host computation and publication;
//! no upload or provider dispatch is asserted for this route.

use super::admission::combine_host_scratch;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{
    provider_for_handle, GpuTensorHandle, ProviderFeasibility, ProviderFeasibilityQuery,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementResourceSnapshot,
    PlacementSignature, ProviderReadbackDemand, ProviderReadbackDemands, ResourceByteBound,
    WholeWorkCandidate, WholeWorkConstraint, WholeWorkPlacementRequest,
};
mod receipt;
pub(crate) use receipt::HostSinkReceipt;
#[cfg(test)]
#[path = "host_sink/tests.rs"]
mod tests;

pub(crate) struct ExactHostSink<'a> {
    pub(crate) operation: &'static str,
    pub(crate) variant: &'static str,
    pub(crate) source: &'a GpuTensorHandle,
    pub(crate) domain_facts: Digest,
    pub(crate) output_bytes: ResourceByteBound,
    /// Operation-owned host temporaries, excluding the provider's native
    /// returned payload and provider-owned mapped/staging allocations.
    pub(crate) host_compute_scratch: ResourceByteBound,
}

pub(crate) fn select_exact_host_sink(
    runtime: Option<&RuntimeContext>,
    request: ExactHostSink<'_>,
) -> BuiltinResult<Option<HostSinkReceipt>> {
    let source = request.source;
    select_exact_host_sink_with_readbacks(runtime, request, std::slice::from_ref(source))
}

/// All resident operands whose native payloads remain available during the
/// host computation. A source may belong to a different provider than the
/// output-intent source; each readback uses its actual live owner.
pub(crate) fn select_exact_host_sink_with_readbacks(
    runtime: Option<&RuntimeContext>,
    request: ExactHostSink<'_>,
    readbacks: &[GpuTensorHandle],
) -> BuiltinResult<Option<HostSinkReceipt>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(None);
    };
    if readbacks.is_empty()
        || !readbacks
            .iter()
            .any(|source| source.view_identity() == request.source.view_identity())
    {
        return Err(
            build_runtime_error("host sink readbacks omit the output-intent source")
                .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                .build(),
        );
    }
    let providers = readbacks
        .iter()
        .map(|source| {
            provider_for_handle(source).filter(|owner| {
                owner.device_id() == source.device_id()
                    && source.allocation_lease().validate_live().is_ok()
            })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            build_runtime_error("host sink source has no live allocation owner")
                .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                .build()
        })?;
    let downloads = readbacks
        .iter()
        .map(|source| {
            ProviderFeasibilityQuery::exact_numeric_download(source).ok_or_else(|| {
                build_runtime_error("host sink has no exact numeric readback representation")
                    .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                    .build()
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let provider_resources = providers
        .iter()
        .map(|provider| provider.placement_resources())
        .collect::<Vec<_>>();
    if readbacks
        .iter()
        .zip(&providers)
        .zip(&downloads)
        .any(|((source, provider), download)| {
            !matches!(
                provider.query_numeric_download_feasibility(source, download),
                ProviderFeasibility::Supported { .. }
            )
        })
    {
        if parallel.memory_available_bytes.is_some()
            || provider_resources.iter().any(|resource| {
                resource.capacity_bytes.is_some() || resource.scratch_available_bytes.is_some()
            })
        {
            return Err(build_runtime_error(
                "host sink has no proven download route under a finite resource budget",
            )
            .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
            .build());
        }
        return Ok(None);
    }
    let readback_host = readbacks.iter().zip(&providers).try_fold(
        ResourceByteBound::Known(0),
        |total, (source, provider)| {
            combine_host_scratch(
                total,
                provider.download_numeric_host_workspace_bound(source),
            )
        },
    )?;
    let host_scratch = combine_host_scratch(request.host_compute_scratch, readback_host)?;
    let readback_demands =
        ProviderReadbackDemands::sequential(readbacks.iter().zip(&providers).map(
            |(source, provider)| ProviderReadbackDemand {
                device_id: provider.device_id(),
                provider_scratch_bytes: provider.download_numeric_provider_scratch_bound(source),
                queue_slots: provider.download_numeric_queue_slots(source),
            },
        ))
        .ok_or_else(|| {
            build_runtime_error("host sink exceeds the provider resource set")
                .with_identifier("RunMat:Placement:TooManyProviders")
                .build()
        })?;
    let source_capabilities = providers
        .iter()
        .map(|provider| (provider.device_id(), provider.capability_snapshot()))
        .collect::<Vec<_>>();
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&source_capabilities).expect("provider capability serializes"),
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
                &downloads,
                request.output_bytes,
                readback_host,
                readback_demands,
                readbacks
                    .iter()
                    .zip(&providers)
                    .map(|(source, provider)| {
                        (
                            provider.device_id(),
                            source.shape(),
                            source.physical().element_type(),
                            source.physical().requested_bytes(),
                            source.view().layout(),
                            runmat_accelerate_api::handle_provenance(source),
                        )
                    })
                    .collect::<Vec<_>>(),
            ))
            .expect("typed host-sink facts serialize"),
        ),
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    };
    let candidate = WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Host,
        presentation: if readbacks.len() == 1 {
            let demand = readback_demands.iter().next().expect("one readback");
            CandidateOutputPresentation::ProviderDownload {
                device_id: demand.device_id,
                provider_scratch_bytes: demand.provider_scratch_bytes,
                queue_slots: demand.queue_slots,
            }
        } else {
            CandidateOutputPresentation::ProviderReadbacksHost {
                readbacks: Box::new(readback_demands),
            }
        },
        route_facts: Digest::sha256(
            serde_json::to_vec(&downloads).expect("download routes serialize"),
        ),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            retained_bytes: request.output_bytes,
            scratch_bytes: host_scratch,
            provider_host_workspace: None,
            provider_input_stages: None,
            queue_slots: 0,
            peak_live: None,
        },
        transactional_results: true,
    };
    let mut unique_resources = Vec::new();
    for resource in provider_resources {
        if !unique_resources
            .iter()
            .any(|existing: &runmat_execution::ProviderResourceSnapshot| {
                existing.device_id == resource.device_id
            })
        {
            unique_resources.push(resource);
        }
    }
    unique_resources.sort_by_key(|resource| resource.device_id);
    let epoch_digest = Digest::sha256(
        serde_json::to_vec(&(parallel.epoch, &unique_resources))
            .expect("resource snapshot serializes"),
    );
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: unique_resources,
        epoch: u64::from_le_bytes(epoch_digest.bytes()[..8].try_into().expect("digest length")),
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
            build_runtime_error("placement selected an unadmitted host sink route")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    Ok(Some(HostSinkReceipt::new(runtime.clone(), selected)))
}
