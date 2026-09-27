//! Inputs that distinguish execution feedback for a local placement decision.
use super::super::residency::ResidencySummary;
use runmat_accelerate_api::ProviderCostQuery;
use runmat_execution::{CandidatePreparationState, Digest, PlacementResourceSnapshot};

pub(super) fn with_output_resources(facts: Digest, output_bytes: Option<u64>) -> Digest {
    Digest::sha256(
        serde_json::to_vec(&("physical-output-resources-v1", facts, output_bytes))
            .expect("provider output resources must serialize"),
    )
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
    Digest::sha256(
        serde_json::to_vec(&(
            cost,
            cpu_preparation,
            residency.host_bytes,
            &residency.provider_bytes,
            residency.unknown_values,
            resources,
            host_epoch,
        ))
        .expect("placement cost facts must serialize"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        ProviderFeasibilityQuery, ProviderOperationFamily, ProviderOperationIdentity,
        ProviderWorkload,
    };

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
                    matmul: None,
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
