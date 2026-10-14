use super::*;
use runmat_accelerate_api::{
    ProviderElementType, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationIdentity,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};

fn query(source: ProviderElementType, elements: u64, host: bool) -> ProviderCostQuery {
    ProviderCostQuery {
        operation: ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(
                runmat_builtins::SINGLE_CATALOG_ENTRY.identity.name,
            ),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![ProviderRepresentation {
                element_type: source,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: vec![1, elements],
                residency: if host {
                    ProviderResidency::Host
                } else {
                    ProviderResidency::Device
                },
            }],
            outputs: vec![],
            workload: runmat_accelerate_api::ProviderWorkload {
                preparation_route:
                    runmat_accelerate_api::ProviderPreparationProgram::FloatingCast {
                        target: runmat_accelerate_api::ProviderPrecision::F32,
                    }
                    .route(),
                ..Default::default()
            },
        },
        preparation: CandidatePreparationState::Cold,
        context: Default::default(),
        required_upload_bytes: if host {
            source.byte_width() * elements
        } else {
            0
        },
        required_download_bytes: 0,
        downstream_materialization: false,
    }
}

#[test]
fn plain_cast_prior_prices_actual_staging_params_and_fences_once() {
    let provider =
        crate::placement::provider_contract::fusion_elementwise::fixture::GeometryProvider(64);
    let mut request = query(ProviderElementType::F64, 8, true);
    let base = wgpu_cost_estimate(&provider, &request).cost;
    assert_eq!(base.components.execution_ns, 16);
    assert_eq!(base.components.allocation_ns, 15_000); // upload, result, params
    assert_eq!(base.components.queue_ns, 10_000);
    assert_eq!(base.components.upload_ns, transfer_prior_ns(64 + 16));
    assert_eq!(base.scratch_bytes, Some(64 + 16));
    request.context.observation = Some(
        runmat_accelerate_api::observation::ProviderObservationOverhead {
            submissions: 1,
            requested_peak_buffer_bytes: 8,
            clear_commands: 1,
            map_callbacks: 1,
            wgpu_compute_query: None,
        },
    );
    let observed = wgpu_cost_estimate(&provider, &request).cost;
    assert_eq!(
        observed.components.execution_ns,
        base.components.execution_ns + 2
    );
    assert_eq!(
        observed.components.allocation_ns,
        base.components.allocation_ns + 5_000
    );
    assert_eq!(observed.components.queue_ns, base.components.queue_ns);
    assert_eq!(
        observed.scratch_bytes,
        base.scratch_bytes.map(|bytes| bytes + 8)
    );
    assert_eq!(observed.source, EstimateSource::StaticPrior);
}

#[test]
fn alias_and_empty_casts_do_not_pay_for_unused_pipeline_or_submission() {
    let provider =
        crate::placement::provider_contract::fusion_elementwise::fixture::GeometryProvider(256);
    for (source, elements, allocations) in [
        (ProviderElementType::F32, 8, 0),
        (ProviderElementType::F32, 0, 0),
        (ProviderElementType::F64, 0, 1),
    ] {
        let cost = wgpu_cost_estimate(&provider, &query(source, elements, false)).cost;
        assert_eq!(cost.components.execution_ns, 0);
        assert_eq!(cost.components.queue_ns, 0);
        assert_eq!(cost.components.compile_or_prepare_ns, 0);
        assert_eq!(cost.components.allocation_ns, allocations * 5_000);
        assert_eq!(cost.components.upload_ns, 0);
        assert_eq!(cost.scratch_bytes, Some(0));
    }
}
