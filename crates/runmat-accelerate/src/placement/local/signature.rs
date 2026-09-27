//! Inputs that distinguish execution feedback for a local placement decision.
use super::super::residency::ResidencySummary;
use runmat_accelerate_api::ProviderCostQuery;
use runmat_execution::{CandidatePreparationState, Digest, PlacementResourceSnapshot};

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
    fn preparation_and_materialization_do_not_share_feedback_series() {
        let query = ProviderCostQuery {
            operation: ProviderFeasibilityQuery {
                operation: ProviderOperationIdentity::new("test.product"),
                family: ProviderOperationFamily::MatrixMultiply,
                inputs: vec![],
                outputs: vec![],
                workload: ProviderWorkload {
                    elements: Some(4),
                    flops: Some(8),
                    batch: Some(1),
                    numerical_domain: Default::default(),
                },
            },
            preparation: CandidatePreparationState::Warm,
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
        let mut changed = query.clone();
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
