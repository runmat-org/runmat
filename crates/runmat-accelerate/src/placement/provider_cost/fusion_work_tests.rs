use super::*;
use runmat_accelerate_api::{
    ProviderElementType, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationIdentity,
    ProviderRepresentation, ProviderResidency, ProviderStorage, ProviderWorkload,
    RESIDENT_SELECT_OPERATION_IDENTITY,
};

fn query(element: ProviderElementType, elements: u64, units: Option<u64>) -> ProviderCostQuery {
    let value = ProviderRepresentation {
        element_type: element,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![1, elements],
        residency: ProviderResidency::Device,
    };
    ProviderCostQuery {
        operation: ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("fusion.elementwise"),
            family: ProviderOperationFamily::Fusion,
            inputs: vec![value.clone()],
            outputs: vec![value],
            workload: ProviderWorkload {
                elements: Some(elements),
                scalar_work_units: units,
                ..Default::default()
            },
        },
        preparation: CandidatePreparationState::Warm,
        context: Default::default(),
        required_upload_bytes: 0,
        required_download_bytes: 0,
        downstream_materialization: false,
    }
}

#[test]
fn exact_cache_readiness_is_distinct_from_uncertain_compile_prior_and_fence_service() {
    use runmat_accelerate_api::{
        observation::ProviderObservationOverhead, ProviderPreparationSnapshot,
    };
    let provider =
        crate::placement::provider_contract::fusion_elementwise::fixture::GeometryProvider::new();
    let mut query = query(ProviderElementType::F64, 8, Some(8));
    let baseline = wgpu_cost_estimate(&provider, &query).cost;
    query.context.preparation = Some(ProviderPreparationSnapshot {
        identity: runmat_execution::Digest::sha256(b"exact pipeline keys"),
        state: Some(CandidatePreparationState::Cold),
        cold_pipelines: 2,
        warm_pipelines: 1,
    });
    query.context.observation = Some(ProviderObservationOverhead {
        submissions: 2,
        requested_peak_buffer_bytes: 16,
        clear_commands: 2,
        map_callbacks: 2,
        wgpu_compute_query: None,
    });
    let measured_intent = wgpu_cost_estimate(&provider, &query).cost;
    assert_eq!(measured_intent.components.compile_or_prepare_ns, 200_000);
    assert_eq!(
        measured_intent.components.queue_ns,
        baseline.components.queue_ns
    );
    assert_eq!(
        measured_intent.components.allocation_ns,
        baseline.components.allocation_ns + 10_000
    );
    assert_eq!(
        measured_intent.components.synchronization_ns,
        baseline.components.synchronization_ns + 10_000
    );
    assert_eq!(
        measured_intent.components.execution_ns,
        baseline.components.execution_ns + 4
    );
    assert_eq!(
        measured_intent.scratch_bytes,
        baseline.scratch_bytes.map(|bytes| bytes + 16)
    );
    assert_eq!(measured_intent.source, EstimateSource::StaticPrior);
    query
        .context
        .observation
        .as_mut()
        .unwrap()
        .wgpu_compute_query = Some(
        runmat_accelerate_api::observation::WgpuEncoderQueryProspective {
            max_compute_encoders: 2,
            max_total_encoders: 2,
            max_query_sets: 2,
            max_timestamp_slots: 32,
            requested_peak_buffer_bytes: 512,
            max_map_callbacks: 2,
        },
    );
    let with_optional_queries = wgpu_cost_estimate(&provider, &query).cost;
    assert_eq!(
        with_optional_queries.components.allocation_ns,
        measured_intent.components.allocation_ns + 30_000
    );
    assert_eq!(
        with_optional_queries.components.synchronization_ns,
        measured_intent.components.synchronization_ns + 10_000
    );
    assert_eq!(
        with_optional_queries.scratch_bytes,
        measured_intent.scratch_bytes.map(|bytes| bytes + 512)
    );
    query
        .context
        .observation
        .as_mut()
        .unwrap()
        .wgpu_compute_query = None;
    query.context.preparation.as_mut().unwrap().state = Some(CandidatePreparationState::Warm);
    assert_eq!(
        wgpu_cost_estimate(&provider, &query)
            .cost
            .components
            .compile_or_prepare_ns,
        0
    );
    query.context.preparation.as_mut().unwrap().state = None;
    query.preparation = CandidatePreparationState::Cold;
    assert_eq!(
        wgpu_cost_estimate(&provider, &query)
            .cost
            .components
            .compile_or_prepare_ns,
        100_000
    );
}

#[test]
fn weighted_primary_program_is_not_repeated_for_output_adapters() {
    let provider =
        crate::placement::provider_contract::fusion_elementwise::fixture::GeometryProvider::new();
    let native = wgpu_cost_estimate(
        &provider,
        &query(ProviderElementType::F64, 8, Some(100_000)),
    )
    .cost;
    let adapted = wgpu_cost_estimate(
        &provider,
        &query(ProviderElementType::F32, 8, Some(100_000)),
    )
    .cost;
    assert_eq!(native.components.execution_ns, 100_000);
    assert_eq!(adapted.components.execution_ns, 100_000 + 2 * (8 + 8));
    assert!(adapted.components.queue_ns > native.components.queue_ns);
    assert!(adapted.scratch_bytes > native.scratch_bytes);
    assert_eq!(adapted.source, EstimateSource::StaticPrior);

    // Existing undeclared-work callers retain their old composite prior.
    let legacy = wgpu_cost_estimate(&provider, &query(ProviderElementType::F32, 8, None)).cost;
    assert_eq!(legacy.components.execution_ns, 3 * 5_000);
}

#[test]
fn weighted_work_is_total_across_chunks_and_empty_work_stays_zero() {
    let provider =
        crate::placement::provider_contract::fusion_elementwise::fixture::GeometryProvider::new();
    let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(
        provider.elementwise_geometry().unwrap().workgroup_size,
    )
    .unwrap();
    let elements = capacity + 1;
    let estimate = wgpu_cost_estimate(
        &provider,
        &query(ProviderElementType::F64, elements, Some(elements * 100)),
    )
    .cost;
    assert_eq!(estimate.components.execution_ns, elements * 100);
    assert_eq!(estimate.components.queue_ns, 30_000); // two chunks plus warmup

    let empty = wgpu_cost_estimate(&provider, &query(ProviderElementType::F64, 0, Some(0))).cost;
    assert_eq!(empty.components.execution_ns, 0);
    assert_eq!(empty.components.queue_ns, 0);
    assert_eq!(empty.scratch_bytes, Some(0));

    let mut cold_empty = query(ProviderElementType::F32, 0, Some(0));
    cold_empty.preparation = CandidatePreparationState::Cold;
    cold_empty.operation.inputs[0].shape = vec![1, 1];
    cold_empty.required_upload_bytes = 8;
    cold_empty.required_download_bytes = 8;
    cold_empty.downstream_materialization = true;
    let empty = wgpu_cost_estimate(&provider, &cold_empty).cost;
    assert_eq!(empty.components.execution_ns, 0);
    assert_eq!(empty.components.compile_or_prepare_ns, 0);
    assert_eq!(empty.components.queue_ns, 0);
    assert_eq!(empty.components.allocation_ns, 0);
    assert_eq!(empty.components.upload_ns, 0);
    assert_eq!(empty.components.download_ns, 0);
    assert_eq!(empty.components.synchronization_ns, 0);
    assert_eq!(empty.components.downstream_ns, 0);
    assert_eq!(empty.scratch_bytes, Some(0));
}

#[test]
fn resident_select_cost_uses_its_three_input_physical_route() {
    let provider =
        crate::placement::provider_contract::fusion_elementwise::fixture::GeometryProvider::new();
    let representation = |element_type, shape: Vec<u64>| ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: ProviderResidency::Device,
    };
    let query = ProviderCostQuery {
        operation: ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(RESIDENT_SELECT_OPERATION_IDENTITY),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![
                representation(ProviderElementType::Logical, vec![2, 1]),
                representation(ProviderElementType::F64, vec![1, 3]),
                representation(ProviderElementType::F64, vec![1, 3]),
            ],
            outputs: vec![representation(ProviderElementType::F64, vec![2, 3])],
            workload: ProviderWorkload {
                elements: Some(6),
                ..Default::default()
            },
        },
        preparation: CandidatePreparationState::Warm,
        context: Default::default(),
        required_upload_bytes: 0,
        required_download_bytes: 0,
        downstream_materialization: false,
    };
    let cost = wgpu_cost_estimate(&provider, &query).cost;
    let uniform_bytes = crate::backend::wgpu_geometry::fusion_broadcast_uniform_bytes(
        3,
        crate::backend::wgpu::params::BCAST_MAX_RANK,
    )
    .unwrap();
    assert_eq!(cost.components.execution_ns, 12);
    assert_eq!(cost.components.queue_ns, 20_000);
    assert_eq!(cost.components.allocation_ns, 10_000);
    assert_eq!(cost.components.upload_ns, transfer_prior_ns(uniform_bytes));
    assert_eq!(cost.scratch_bytes, Some(uniform_bytes));
    assert_eq!(cost.source, EstimateSource::StaticPrior);
}
