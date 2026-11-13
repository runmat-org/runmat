//! Inputs that distinguish execution feedback for a local placement decision.
use super::super::residency::ResidencySummary;
use runmat_accelerate_api::ProviderCostQuery;
use runmat_accelerate_api::{ProviderFeasibilityQuery, ProviderRejectionCode};
use runmat_execution::{CandidatePreparationState, Digest, PlacementResourceSnapshot};

// Feedback totals now cover admitted operand preparation, the body, and owned
// cleanup. This is a measurement contract, not a host resource epoch or a
// provider capability revision. Keep old exported series out of both admitted
// and rejected local-placement decisions without invalidating unrelated keys.
const PREPARED_BODY_CLEANUP_CONTRACT: &[u8] =
    b"runmat-local-total-candidate-preparation-body-cleanup-v2";
const FULL_CALL_CONTRACT: &[u8] = b"runmat-local-total-candidate-outer-fence-v3";

pub(super) fn with_cpu_candidate(
    facts: Digest,
    cpu: &runmat_execution::ExecutionCandidateDescriptor,
    resources: &runmat_execution::CandidateResourceDemand,
) -> Digest {
    Digest::sha256_serialized(&("local-rejected-host-demand-v1", facts, cpu, resources))
        .expect("CPU candidate demand serializes")
}

pub(super) fn with_total_candidate_observation_contract(
    facts: Digest,
    contract: super::TotalCandidateObservationContract,
) -> Digest {
    let label = match contract {
        super::TotalCandidateObservationContract::PreparationBodyCleanup => {
            PREPARED_BODY_CLEANUP_CONTRACT
        }
        super::TotalCandidateObservationContract::FullCall => FULL_CALL_CONTRACT,
        super::TotalCandidateObservationContract::NativeFullCall => {
            b"runmat-local-native-source-outer-fence-v1"
        }
    };
    let mut bytes = Vec::with_capacity(label.len() + 32);
    bytes.extend_from_slice(label);
    bytes.extend_from_slice(facts.bytes());
    Digest::sha256(bytes)
}

pub(super) fn with_output_resources(facts: Digest, output_bytes: Option<u64>) -> Digest {
    Digest::sha256_serialized(&("physical-output-resources-v1", facts, output_bytes))
        .expect("provider output resources must serialize")
}

pub(super) fn rejected_runtime_facts(
    query: &ProviderFeasibilityQuery,
    rejection: ProviderRejectionCode,
    cpu_preparation: CandidatePreparationState,
    residency: &ResidencySummary,
    resources: &PlacementResourceSnapshot,
    host_epoch: u64,
) -> Digest {
    Digest::sha256_serialized(&(
        query,
        rejection,
        cpu_preparation,
        residency.host_bytes,
        &residency.provider_bytes,
        residency.unknown_values,
        resources,
        host_epoch,
    ))
    .expect("rejected placement facts must serialize")
}

pub(super) fn with_semantic_identity(facts: Digest, semantic: Option<Digest>) -> Digest {
    let Some(semantic) = semantic else {
        return facts;
    };
    let mut bytes = Vec::with_capacity(96);
    bytes.extend_from_slice(b"runmat-local-typed-program-v1");
    bytes.extend_from_slice(facts.bytes());
    bytes.extend_from_slice(semantic.bytes());
    Digest::sha256(bytes)
}

pub(super) fn runtime_facts(
    cost: &ProviderCostQuery,
    cpu_preparation: CandidatePreparationState,
    residency: &ResidencySummary,
    resources: &PlacementResourceSnapshot,
    host_epoch: u64,
) -> Digest {
    // Reuse the cost-query contract so preparation and transfer requirements
    // cannot diverge between cost estimation and its feedback/cache identity.
    Digest::sha256_serialized(&(
        cost,
        cpu_preparation,
        residency.host_bytes,
        &residency.provider_bytes,
        residency.unknown_values,
        resources,
        host_epoch,
    ))
    .expect("placement cost facts must serialize")
}

/// The entire executable/rejected physical set shares one feedback signature.
/// Candidate names are sorted by the caller, so provider probe order cannot
/// split warm/cold histories or change the service's graph identity.
/// Known byte bounds retain their original JSON number encoding. Unavailable
/// bounds are distinct facts, never a zero-byte resource permission.
pub(super) fn candidate_set_facts(
    admitted: &[(
        &String,
        &ProviderCostQuery,
        runmat_execution::ResourceByteBound,
    )],
    rejected: &[(
        Option<Digest>,
        ProviderFeasibilityQuery,
        ProviderRejectionCode,
    )],
    source_digest: Digest,
    cpu_preparation: CandidatePreparationState,
    residency: &ResidencySummary,
    resources: &PlacementResourceSnapshot,
    host_epoch: u64,
) -> Digest {
    Digest::sha256_serialized(&(
        "local-physical-candidate-set-v1",
        admitted,
        rejected,
        source_digest,
        cpu_preparation,
        residency.host_bytes,
        &residency.provider_bytes,
        residency.unknown_values,
        resources,
        host_epoch,
    ))
    .expect("physical candidate set facts must serialize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_output_resource_identity_matches_owned_json_bytes() {
        let facts = Digest::sha256(b"selected image source");
        for output_bytes in [None, Some(0), Some(u64::MAX)] {
            assert_eq!(
                with_output_resources(facts, output_bytes),
                Digest::sha256(
                    serde_json::to_vec(&("physical-output-resources-v1", facts, output_bytes))
                        .unwrap(),
                ),
            );
        }
    }
    use runmat_accelerate_api::{
        ProviderFeasibilityQuery, ProviderOperationFamily, ProviderOperationIdentity,
        ProviderWorkload,
    };

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn total_candidate_contract_separates_old_supported_and_rejected_series() {
        use super::super::TotalCandidateObservationContract as Contract;
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("test.reduction"),
            family: ProviderOperationFamily::Reduction,
            inputs: Vec::new(),
            outputs: Vec::new(),
            workload: ProviderWorkload::default(),
        };
        let residency = ResidencySummary::default();
        let resources = PlacementResourceSnapshot::default();
        let cost = ProviderCostQuery {
            operation: query.clone(),
            preparation: CandidatePreparationState::Ready,
            context: Default::default(),
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        };
        let supported_old = with_semantic_identity(
            with_output_resources(
                runtime_facts(
                    &cost,
                    CandidatePreparationState::Ready,
                    &residency,
                    &resources,
                    7,
                ),
                Some(32),
            ),
            Some(Digest::sha256(b"typed-form")),
        );
        let rejected_old = with_semantic_identity(
            rejected_runtime_facts(
                &query,
                ProviderRejectionCode::UnsupportedElementType,
                CandidatePreparationState::Ready,
                &residency,
                &resources,
                7,
            ),
            Some(Digest::sha256(b"typed-form")),
        );
        assert_ne!(
            supported_old,
            with_total_candidate_observation_contract(
                supported_old,
                Contract::PreparationBodyCleanup
            )
        );
        assert_ne!(
            rejected_old,
            with_total_candidate_observation_contract(
                rejected_old,
                Contract::PreparationBodyCleanup
            )
        );
        assert_ne!(
            with_total_candidate_observation_contract(
                supported_old,
                Contract::PreparationBodyCleanup
            ),
            with_total_candidate_observation_contract(
                rejected_old,
                Contract::PreparationBodyCleanup
            )
        );
        assert_ne!(
            with_total_candidate_observation_contract(
                supported_old,
                Contract::PreparationBodyCleanup
            ),
            with_total_candidate_observation_contract(supported_old, Contract::FullCall),
            "old v2 series cannot train the outer-fence v3 route"
        );
        assert_ne!(
            with_total_candidate_observation_contract(
                rejected_old,
                Contract::PreparationBodyCleanup
            ),
            with_total_candidate_observation_contract(rejected_old, Contract::FullCall),
            "rejected CPU feedback obeys the same contract version"
        );
    }

    #[test]
    fn rejected_cpu_series_is_keyed_by_typed_provider_decision() {
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("test.reduction"),
            family: ProviderOperationFamily::Reduction,
            inputs: Vec::new(),
            outputs: Vec::new(),
            workload: ProviderWorkload::default(),
        };
        let residency = ResidencySummary::default();
        let resources = PlacementResourceSnapshot::default();
        let facts = |query: &ProviderFeasibilityQuery, code, preparation| {
            rejected_runtime_facts(query, code, preparation, &residency, &resources, 7)
        };
        let original = facts(
            &query,
            ProviderRejectionCode::UnsupportedElementType,
            CandidatePreparationState::Ready,
        );
        assert_ne!(
            original,
            facts(
                &query,
                ProviderRejectionCode::UnknownResourceEstimate,
                CandidatePreparationState::Ready,
            )
        );
        assert_ne!(
            original,
            facts(
                &query,
                ProviderRejectionCode::UnsupportedElementType,
                CandidatePreparationState::Cold,
            )
        );
        let mut changed = query.clone();
        changed.workload.elements = Some(9);
        assert_ne!(
            original,
            facts(
                &changed,
                ProviderRejectionCode::UnsupportedElementType,
                CandidatePreparationState::Ready,
            )
        );
    }

    #[test]
    fn semantic_identity_extends_without_erasing_existing_argument_facts() {
        let first = Digest::sha256(b"first argument facts");
        let second = Digest::sha256(b"second argument facts");
        let direct = Digest::sha256(b"double operation");
        let rounded = Digest::sha256(b"single intermediate then double operation");
        assert_eq!(with_semantic_identity(first, None), first);
        assert_ne!(
            with_semantic_identity(first, Some(direct)),
            with_semantic_identity(first, Some(rounded))
        );
        assert_ne!(
            with_semantic_identity(first, Some(direct)),
            with_semantic_identity(second, Some(direct))
        );
    }

    #[test]
    fn preparation_and_materialization_do_not_share_feedback_series() {
        let query = ProviderCostQuery {
            operation: ProviderFeasibilityQuery {
                operation: ProviderOperationIdentity::new("test.product"),
                family: ProviderOperationFamily::MatrixMultiply,
                inputs: vec![],
                outputs: vec![],
                workload: ProviderWorkload {
                    gpu_random: None,
                    stochastic_evolution: None,
                    gpu_integer: None,
                    gpu_integer_batch: None,
                    fused_elementwise: None,
                    real_power_mode: None,
                    real_power_scalar: None,
                    preparation_route: None,
                    matmul: None,
                    typed_reduction: None,
                    sum: None,
                    convolution: None,
                    iir: None,
                    fspecial: None,
                    imfilter: None,
                    extrema: None,
                    image_stages: None,
                    transfer_output_bounds: None,
                    elements: Some(4),
                    flops: Some(8),
                    scalar_work_units: None,
                    batch: Some(1),
                    numerical_domain: Default::default(),
                    requested_outputs: None,
                },
            },
            preparation: CandidatePreparationState::Warm,
            context: Default::default(),
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        };
        let residency = ResidencySummary {
            host_bytes: 32,
            ..Default::default()
        };
        let resources = PlacementResourceSnapshot::default();
        let digest = |query: &ProviderCostQuery| {
            runtime_facts(
                query,
                CandidatePreparationState::Warm,
                &residency,
                &resources,
                1,
            )
        };
        let original = digest(&query);
        assert_eq!(original, digest(&query.clone()));
        let mut prepared = query.clone();
        prepared.context.preparation = Some(runmat_accelerate_api::ProviderPreparationSnapshot {
            identity: Digest::sha256(b"selected pipeline recipes"),
            state: Some(CandidatePreparationState::Cold),
            cold_pipelines: 1,
            warm_pipelines: 0,
        });
        let cold = digest(&prepared);
        assert_ne!(original, cold);
        prepared.context.preparation.as_mut().unwrap().state =
            Some(CandidatePreparationState::Warm);
        assert_ne!(cold, digest(&prepared));
        let without_fences = digest(&prepared);
        prepared.context.observation = Some(
            runmat_accelerate_api::observation::ProviderObservationOverhead {
                submissions: 2,
                requested_peak_buffer_bytes: 16,
                clear_commands: 2,
                map_callbacks: 2,
                wgpu_compute_query: None,
            },
        );
        assert_ne!(without_fences, digest(&prepared));
        let mut changed = query.clone();
        changed.operation.workload.scalar_work_units = Some(32);
        assert_ne!(original, digest(&changed));
        let first_work = digest(&changed);
        changed.operation.workload.scalar_work_units = Some(64);
        assert_ne!(first_work, digest(&changed));
        changed = query.clone();
        changed.operation.workload.requested_outputs = Some(0);
        assert_ne!(original, digest(&changed));
        let discarded = digest(&changed);
        changed.operation.workload.requested_outputs = Some(3);
        assert_ne!(discarded, digest(&changed));
        changed = query.clone();
        changed.preparation = CandidatePreparationState::Cold;
        assert_ne!(original, digest(&changed));
        changed = query.clone();
        changed.required_upload_bytes = 32;
        assert_ne!(original, digest(&changed));
        changed = query.clone();
        changed.required_download_bytes = 32;
        assert_ne!(original, digest(&changed));
        changed = query.clone();
        changed.downstream_materialization = true;
        assert_ne!(original, digest(&changed));
        assert_ne!(
            original,
            runtime_facts(
                &query,
                CandidatePreparationState::Cold,
                &residency,
                &resources,
                1
            )
        );
        assert_ne!(
            original,
            runtime_facts(
                &query,
                CandidatePreparationState::Warm,
                &residency,
                &resources,
                2
            )
        );
        let mut owner_a = residency.clone();
        owner_a.provider_bytes.insert(1, 128);
        let mut owner_b = residency.clone();
        owner_b.provider_bytes.insert(2, 128);
        assert_ne!(
            runtime_facts(
                &query,
                CandidatePreparationState::Warm,
                &owner_a,
                &resources,
                1
            ),
            runtime_facts(
                &query,
                CandidatePreparationState::Warm,
                &owner_b,
                &resources,
                1
            ),
        );
        let mut changed_resources = resources.clone();
        changed_resources.cpu_millicores_available = 2000;
        assert_ne!(
            original,
            runtime_facts(
                &query,
                CandidatePreparationState::Warm,
                &residency,
                &changed_resources,
                1
            )
        );
        changed_resources = resources.clone();
        changed_resources.memory_available_bytes = Some(4096);
        assert_ne!(
            original,
            runtime_facts(
                &query,
                CandidatePreparationState::Warm,
                &residency,
                &changed_resources,
                1
            )
        );
    }
}
