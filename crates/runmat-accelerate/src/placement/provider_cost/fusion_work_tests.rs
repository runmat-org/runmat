use super::*;
use runmat_accelerate_api::{
    ProviderElementType, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationIdentity,
    ProviderRepresentation, ProviderResidency, ProviderStorage, ProviderWorkload,
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
        required_upload_bytes: 0,
        required_download_bytes: 0,
        downstream_materialization: false,
    }
}

#[test]
fn weighted_primary_program_is_not_repeated_for_output_adapters() {
    let provider = crate::simple_provider::InProcessProvider::new();
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
    assert_eq!(adapted.components.execution_ns, 100_000);
    assert!(adapted.components.queue_ns > native.components.queue_ns);
    assert!(adapted.scratch_bytes > native.scratch_bytes);
    assert_eq!(adapted.source, EstimateSource::StaticPrior);

    // Existing undeclared-work callers retain their old composite prior.
    let legacy = wgpu_cost_estimate(&provider, &query(ProviderElementType::F32, 8, None)).cost;
    assert_eq!(legacy.components.execution_ns, 3 * 5_000);
}

#[test]
fn weighted_work_is_total_across_chunks_and_empty_work_stays_zero() {
    let provider = crate::simple_provider::InProcessProvider::new();
    let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(
        crate::backend::wgpu::config::effective_workgroup_size(),
    )
    .unwrap();
    let elements = capacity + 1;
    let estimate = wgpu_cost_estimate(
        &provider,
        &query(ProviderElementType::F64, elements, Some(elements * 100)),
    )
    .cost;
    assert_eq!(estimate.components.execution_ns, elements * 100);
    assert_eq!(estimate.components.queue_ns, 20_000);

    let empty = wgpu_cost_estimate(&provider, &query(ProviderElementType::F64, 0, Some(0))).cost;
    assert_eq!(empty.components.execution_ns, 0);
    assert_eq!(empty.components.queue_ns, 0);
    assert_eq!(empty.scratch_bytes, 0);
}
