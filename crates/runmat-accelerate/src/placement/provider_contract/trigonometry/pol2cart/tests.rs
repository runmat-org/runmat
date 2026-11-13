use super::*;

fn query(outputs: u32) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: operation().identity,
        family: ProviderOperationFamily::Elementwise,
        inputs: [vec![2, 1], vec![1, 1], vec![1, 3]]
            .into_iter()
            .map(|shape| ProviderRepresentation {
                element_type: ProviderElementType::F64,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape,
                residency: ProviderResidency::Host,
            })
            .collect(),
        outputs: vec![],
        workload: runmat_accelerate_api::ProviderWorkload {
            requested_outputs: Some(outputs),
            ..Default::default()
        },
    }
}

#[test]
fn coordinate_work_counts_discarded_xy_elevation_and_multiplier_storage() {
    let provider = crate::simple_provider::InProcessProvider::new();
    for outputs in 0..=3 {
        let query = query(outputs);
        assert!(matches!(
            provider.query_feasibility(&query),
            ProviderFeasibility::Supported { .. }
        ));
        let work = work::derive(ProviderPrecision::F64, false, &query).unwrap();
        assert_eq!(work.outputs.len(), outputs as usize);
        assert_eq!(work.execution_elements, if outputs == 3 { 26 } else { 20 });
        assert_eq!(work.extra_upload_bytes, 48);
        assert_eq!(work.dispatches, if outputs == 3 { 7 } else { 6 });
        assert_eq!(work.scratch_bytes, [208, 160, 112, 112][outputs as usize]);
    }
    let invalid = query(4);
    assert!(matches!(
        provider.query_feasibility(&invalid),
        ProviderFeasibility::Rejected { .. }
    ));
    let mut mismatch = query(3);
    mismatch.inputs[2].element_type = ProviderElementType::F32;
    assert!(matches!(
        provider.query_feasibility(&mismatch),
        ProviderFeasibility::Rejected { .. }
    ));
}

#[cfg(feature = "wgpu")]
#[test]
fn coordinate_wgpu_cost_includes_extra_upload_without_completion_claim() {
    use runmat_accelerate_api::ProviderCostQuery;
    use runmat_execution::{CandidatePreparationState, EstimateSource};
    let provider = crate::simple_provider::InProcessProvider::new();
    let operation = query(3);
    let work = work::derive(ProviderPrecision::F64, true, &operation).unwrap();
    assert_eq!(work.queue_submissions, 10);
    assert!(work.scratch_bytes > 112);
    let cost = crate::placement::provider_cost::wgpu_cost_estimate(
        &provider,
        &ProviderCostQuery {
            operation,
            preparation: CandidatePreparationState::Warm,
            context: Default::default(),
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        },
    )
    .cost;
    assert_eq!(
        cost.components.upload_ns,
        crate::placement::local::transfer_prior_ns(48)
    );
    assert_eq!(cost.components.queue_ns, 100_000);
    assert_eq!(cost.components.download_ns, 0);
    assert_eq!(cost.scratch_bytes, Some(work.scratch_bytes));
    assert_eq!(cost.source, EstimateSource::StaticPrior);
}
