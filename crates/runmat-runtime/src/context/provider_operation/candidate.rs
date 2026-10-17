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

pub(super) fn resource_candidate(
    runtime: Option<&RuntimeContext>,
    request: &ExactProviderOperation<'_>,
    estimate: &ProviderResourceEstimate,
    observation: Option<ProviderObservationOverhead>,
    prepared: &mut PreparedReadbacks,
    parallel: RuntimeParallelResources,
) -> (WholeWorkCandidate, PlacementResourceSnapshot) {
    let provider = request.provider;
    let query = request.query;
    let conditional_provider_upload = prepared.conditional_provider_upload;
    let input_stages = prepared.input_stages;
    // Completion fences coexist with the operation's output/temporary peak.
    // An overflow is unknown physical demand, never a wrapped small bound.
    let peak_live = estimate
        .transient_bytes
        .and_then(|bytes| match observation {
            Some(observation) => bytes.checked_add(observation.requested_peak_buffer_bytes),
            None => Some(bytes),
        })
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
        route_facts: Digest::sha256_serialized(&(
            request.calibration_facts.map(|facts| facts.digest()),
            request.calibration_facts.is_none().then_some(query),
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
            // The legacy fallback is not physical evidence. The owning
            // preflight rejects absent queue bounds under a finite queue;
            // signature/query facts retain the original absence.
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
    let digest = Digest::sha256_serialized(&(parallel.epoch, &prepared.provider_resources))
        .expect("resource snapshot serializes");
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_some_and(RuntimeContext::is_cancelled),
        providers: std::mem::take(&mut prepared.provider_resources),
        epoch: u64::from_le_bytes(digest.bytes()[..8].try_into().expect("digest length")),
    };
    (candidate, resources)
}

/// Identity is optional observation metadata, never a prerequisite to resources.
pub(super) fn operation_signature(
    request: &ExactProviderOperation<'_>,
    prepared: &PreparedReadbacks,
    revision: PlacementRevision,
) -> PlacementSignature {
    let provider = request.provider;
    let query = request.query;
    let conditional_provider_upload = prepared.conditional_provider_upload;
    let input_stages = prepared.input_stages;
    PlacementSignature {
        region: None,
        operation: request.semantic_operation.into(),
        runtime_facts: Digest::sha256_serialized(&(
            request.variant,
            request.semantic_operation,
            request
                .calibration_facts
                .map_or(request.domain_facts, |facts| facts.digest()),
            request.calibration_facts.is_none().then_some(query),
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
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::physical_test_support::OwnedPhysicalProvider;
    use runmat_accelerate_api::random::{
        GpuNormalTransform, GpuRandomAlgorithm, GpuRandomDistribution, GpuRandomReservation,
        GpuRandomWorkload,
    };
    use runmat_accelerate_api::{
        ProviderElementType, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
        ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
        ProviderWorkload,
    };
    use runmat_execution::PlacementExecutionAuthorityRevision;

    #[test]
    fn physical_feedback_reuses_work_without_reusing_random_reservation() {
        let provider = Box::leak(Box::new(OwnedPhysicalProvider::new()));
        let reservation = GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 0,
            first: 0,
            count: 4,
            normal_transform: GpuNormalTransform::BoxMuller,
        };
        let query = |reservation, distribution, class, shape: Vec<u64>| ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("random.gpu_child"),
            family: ProviderOperationFamily::Library,
            inputs: vec![],
            outputs: vec![ProviderRepresentation {
                element_type: class,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape,
                residency: ProviderResidency::Device,
            }],
            workload: ProviderWorkload {
                gpu_random: Some(GpuRandomWorkload {
                    distribution,
                    reservation,
                }),
                ..ProviderWorkload::default()
            },
        };
        let first = query(
            reservation,
            GpuRandomDistribution::Uniform,
            ProviderElementType::F32,
            vec![4],
        );
        let second = query(
            GpuRandomReservation {
                seed: 93,
                first: 1024,
                ..reservation
            },
            GpuRandomDistribution::Uniform,
            ProviderElementType::F32,
            vec![4],
        );
        let normal = query(
            reservation,
            GpuRandomDistribution::StandardNormal,
            ProviderElementType::F32,
            vec![4],
        );
        let wide = query(
            reservation,
            GpuRandomDistribution::Uniform,
            ProviderElementType::F64,
            vec![4],
        );
        let larger = query(
            GpuRandomReservation {
                count: 8,
                ..reservation
            },
            GpuRandomDistribution::Uniform,
            ProviderElementType::F32,
            vec![8],
        );
        let identity = |query: &ProviderFeasibilityQuery| {
            let output_bytes = query.outputs[0].checked_byte_len().unwrap();
            let calibration = super::super::types::ProviderPhysicalCalibration::from_typed(&(
                &query.outputs,
                query.workload.gpu_random.unwrap().distribution,
                output_bytes,
                1_u32,
            ));
            let request = ExactProviderOperation {
                provider,
                query,
                semantic_operation: "rand",
                variant: "random.gpu_child.explicit.v1",
                domain_facts: Digest::sha256_serialized(query).unwrap(),
                calibration_facts: Some(calibration),
                host_scratch: ResourceByteBound::Known(0),
                provider_host_scratch: ResourceByteBound::Known(0),
                host_cpu_millicores: 1_000,
                already_live_bytes: 0,
                readbacks: &[],
                produced_readback: None,
                internal_input_stages: None,
                output_queue_slots: Some(1),
                conditional_host_fallback: None,
                conditional_host_scratch: ResourceByteBound::Known(0),
                conditional_provider_upload: None,
                provider_host_mediated: false,
            };
            let mut prepared =
                super::super::readbacks::prepare_readbacks(&request, Some(output_bytes), None)
                    .unwrap()
                    .unwrap();
            let signature = operation_signature(
                &request,
                &prepared,
                PlacementRevision {
                    program: None,
                    catalog: Digest::sha256(b"catalog"),
                    compiler: Digest::sha256(b"compiler"),
                    authority: PlacementExecutionAuthorityRevision::Provider(Digest::sha256(
                        b"provider",
                    )),
                    policy: Digest::sha256(b"policy"),
                },
            );
            let estimate = ProviderResourceEstimate {
                transient_bytes: Some(output_bytes),
                output_bytes: Some(output_bytes),
                dispatches: Some(1),
            };
            let (candidate, _) = resource_candidate(
                None,
                &request,
                &estimate,
                None,
                &mut prepared,
                RuntimeParallelResources::default(),
            );
            (
                request.domain_facts,
                signature.runtime_facts,
                candidate.route_facts,
            )
        };
        let a = identity(&first);
        let b = identity(&second);
        assert_ne!(a.0, b.0, "exact reservations remain distinct");
        assert_eq!((a.1, a.2), (b.1, b.2), "physical work reuses learning");
        for changed in [&normal, &wide, &larger] {
            let other = identity(changed);
            assert_ne!(a.1, other.1);
            assert_ne!(a.2, other.2);
        }
    }
}
