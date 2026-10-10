//! Identity, resource snapshot and one exact whole-work provider candidate.

use super::{readbacks::PreparedReadbacks, types::ExactProviderOperation};
use crate::context::{RuntimeContext, RuntimeParallelResources};
use runmat_accelerate_api::observation::ProviderObservationOverhead;
use runmat_accelerate_api::ProviderResourceEstimate;
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePeakLiveDemand,
    CandidatePreparationState, CandidateResourceDemand, Digest, ExecutionCandidateKind,
    PlacementResourceSnapshot, PlacementRevision, PlacementSignature, ResourceBoundUnavailable,
    ResourceByteBound, WholeWorkCandidate,
};

pub(super) fn admitted_candidate(
    runtime: &RuntimeContext,
    request: &ExactProviderOperation<'_>,
    estimate: &ProviderResourceEstimate,
    observation: ProviderObservationOverhead,
    prepared: &mut PreparedReadbacks,
    revision: PlacementRevision,
    parallel: RuntimeParallelResources,
) -> (
    PlacementSignature,
    WholeWorkCandidate,
    PlacementResourceSnapshot,
) {
    let provider = request.provider;
    let query = request.query;
    let conditional_provider_upload = prepared.conditional_provider_upload;
    let input_stages = prepared.input_stages;
    let signature = PlacementSignature {
        region: None,
        operation: request.semantic_operation.into(),
        runtime_facts: Digest::sha256(
            serde_json::to_vec(&(
                request.variant,
                request.semantic_operation,
                request.domain_facts,
                query,
                &prepared.readback_facts,
                input_stages,
                prepared.host_scratch,
                prepared.preparation_host_workspace,
                prepared.conditional_readback_host_workspace,
                request.output_queue_slots,
                prepared.produced_readback.as_ref().map(|readback| {
                    (
                        &readback.output,
                        readback.evidence,
                        readback.public_host_bytes,
                    )
                }),
                request.conditional_host_fallback,
                request.conditional_host_scratch,
                request.provider_host_mediated,
                conditional_provider_upload.as_ref().map(|upload| {
                    (
                        upload.transfer_bytes,
                        upload.provider_output_requested_bytes,
                        upload.host_result_bytes,
                        upload.host_scratch_bytes,
                        upload.provider_scratch_bytes,
                        upload.queue_slots,
                    )
                }),
                provider.device_id(),
            ))
            .expect("typed provider operation serializes"),
        ),
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    };
    // Completion fences coexist with the operation's output/temporary peak.
    // An overflow is unknown physical demand, never a wrapped small bound.
    let peak_live = estimate
        .transient_bytes
        .and_then(|bytes| bytes.checked_add(observation.requested_peak_buffer_bytes))
        .and_then(|bytes| match prepared.produced_readback.as_ref() {
            Some(readback) => match readback.evidence.provider_scratch_bytes {
                ResourceByteBound::Known(scratch) => bytes.checked_add(scratch),
                ResourceByteBound::Unavailable(_) => None,
            },
            None => Some(bytes),
        })
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
        output_residency: if let Some(readback) = prepared.produced_readback.as_ref() {
            CandidateOutputResidency::ProviderAndHost {
                device_id: provider.device_id(),
                host_bytes: readback.public_host_bytes,
            }
        } else if let Some(host_bytes) = request.conditional_host_fallback {
            CandidateOutputResidency::ConditionalHostOrProvider {
                device_id: provider.device_id(),
                host_bytes,
            }
        } else {
            CandidateOutputResidency::Provider {
                device_id: provider.device_id(),
            }
        },
        presentation: if let Some(readback) = prepared.produced_readback.as_ref() {
            CandidateOutputPresentation::ProviderProducedReadback {
                device_id: provider.device_id(),
                transfer_bytes: readback.evidence.transfer_bytes,
                provider_scratch_bytes: readback.evidence.provider_scratch_bytes,
                queue_slots: readback.evidence.queue_slots,
            }
        } else {
            conditional_provider_upload.as_ref().map_or(
                CandidateOutputPresentation::Direct,
                |upload| CandidateOutputPresentation::ConditionalProviderUpload {
                    device_id: provider.device_id(),
                    transfer_bytes: upload.transfer_bytes,
                    host_result_bytes: Some(upload.host_result_bytes),
                    host_scratch_bytes: upload.host_scratch_bytes,
                    provider_scratch_bytes: upload.provider_scratch_bytes,
                    queue_slots: upload.queue_slots,
                },
            )
        },
        route_facts: Digest::sha256(
            serde_json::to_vec(&(
                query,
                estimate,
                request.conditional_host_fallback,
                request.provider_host_mediated,
                conditional_provider_upload.as_ref().map(|upload| {
                    (
                        upload.transfer_bytes,
                        upload.provider_output_requested_bytes,
                        upload.host_result_bytes,
                        upload.host_scratch_bytes,
                        upload.provider_scratch_bytes,
                        upload.queue_slots,
                    )
                }),
                input_stages,
                request.output_queue_slots,
                prepared.produced_readback.as_ref().map(|readback| {
                    (
                        &readback.output,
                        readback.evidence,
                        readback.public_host_bytes,
                    )
                }),
            ))
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
            provider_host_workspace: Some(prepared.host_scratch),
            provider_input_stages: input_stages,
            queue_slots: request
                .output_queue_slots
                .unwrap_or_else(|| estimate.dispatches.unwrap_or(1))
                .max(
                    prepared
                        .produced_readback
                        .as_ref()
                        .map_or(0, |readback| readback.evidence.queue_slots),
                ),
            peak_live,
        },
        transactional_results: provider
            .capability_snapshot()
            .concurrency
            .transactional_results,
    };
    prepared
        .provider_resources
        .sort_by_key(|resource| resource.device_id);
    let digest = Digest::sha256(
        serde_json::to_vec(&(parallel.epoch, &prepared.provider_resources))
            .expect("resource snapshot serializes"),
    );
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: std::mem::take(&mut prepared.provider_resources),
        epoch: u64::from_le_bytes(digest.bytes()[..8].try_into().expect("digest length")),
    };
    (signature, candidate, resources)
}
