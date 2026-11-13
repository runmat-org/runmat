//! Estimator composition checks; no hardware execution or timing calibration.
use super::*;
use runmat_accelerate_api::{
    NumericalDomainEvidence, ProviderElementType, ProviderFeasibilityQuery, ProviderLayout,
    ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
    ProviderWorkload, RoundDigitsMode,
};

fn query() -> ProviderCostQuery {
    // 512 lanes * 65,535 groups fills the first dispatch; one final lane
    // requires a second. Only representation metadata is allocated here.
    let input = ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![1, 33_553_921],
        residency: ProviderResidency::Host,
    };
    let mut output = input.clone();
    output.residency = ProviderResidency::Device;
    ProviderCostQuery {
        operation: ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(
                runmat_builtins::ROUND_CATALOG_ENTRY.identity.name,
            ),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![input],
            outputs: vec![output],
            workload: ProviderWorkload {
                elements: Some(33_553_921),
                numerical_domain: NumericalDomainEvidence::RoundDigits {
                    digits: 2,
                    mode: RoundDigitsMode::Decimals,
                },
                ..Default::default()
            },
        },
        preparation: CandidatePreparationState::Cold,
        context: Default::default(),
        required_upload_bytes: 268_431_368,
        required_download_bytes: 0,
        downstream_materialization: false,
    }
}

#[test]
fn digit_estimator_preserves_chunk_allocations_uniform_uploads_and_static_prior() {
    let provider = crate::simple_provider::InProcessProvider::new();
    let mut query = query();
    let cost = wgpu_cost_estimate(&provider, &query).cost;
    // One output and one live 16-byte uniform, reused across two dispatches.
    assert_eq!(cost.components.allocation_ns, 10_000);
    assert_eq!(cost.components.queue_ns, 20_000);
    assert_eq!(cost.scratch_bytes, Some(16));
    // Full F64 input plus two 16-byte uniform writes; no control tensor.
    assert_eq!(cost.components.upload_ns, transfer_prior_ns(268_431_400));
    assert_eq!(cost.components.compile_or_prepare_ns, 100_000);
    assert_eq!(cost.components.download_ns, 0);
    assert_eq!(cost.components.synchronization_ns, 0);
    assert_eq!(cost.components.downstream_ns, 0);
    // The declared static work bound for decimal N=2 is 98 units per lane,
    // using the existing two-nanosecond prior, not measured device completion.
    assert_eq!(cost.components.execution_ns, 6_576_568_516);
    assert_eq!(cost.confidence, EstimateConfidence::Prior);
    assert_eq!(cost.source, EstimateSource::StaticPrior);

    query.preparation = CandidatePreparationState::Warm;
    query.required_download_bytes = 268_431_368;
    query.downstream_materialization = true;
    let materialized = wgpu_cost_estimate(&provider, &query).cost;
    assert_eq!(materialized.components.compile_or_prepare_ns, 0);
    assert_eq!(
        materialized.components.download_ns,
        transfer_prior_ns(268_431_368)
    );
    assert_eq!(materialized.components.synchronization_ns, 5_000);
    assert_eq!(materialized.components.downstream_ns, 5_000);
    assert_eq!(
        materialized.components.execution_ns,
        cost.components.execution_ns
    );
    assert_eq!(materialized.components.queue_ns, cost.components.queue_ns);
    assert_eq!(materialized.components.upload_ns, cost.components.upload_ns);
    assert_eq!(materialized.scratch_bytes, cost.scratch_bytes);
    assert_eq!(materialized.source, EstimateSource::StaticPrior);
}

#[test]
fn digit_estimator_empty_cold_path_allocates_only_output_without_kernel_or_uniform() {
    let provider = crate::simple_provider::InProcessProvider::new();
    let mut query = query();
    query.operation.inputs[0].shape = vec![0, u64::MAX];
    query.operation.outputs[0].shape = vec![0, u64::MAX];
    query.operation.workload.elements = Some(0);
    query.required_upload_bytes = 0;
    for mode in [RoundDigitsMode::Decimals, RoundDigitsMode::Significant] {
        query.operation.workload.numerical_domain = NumericalDomainEvidence::RoundDigits {
            digits: i32::MAX,
            mode,
        };
        let cost = wgpu_cost_estimate(&provider, &query).cost;
        assert_eq!(
            cost.components,
            ExecutionCostComponents {
                allocation_ns: 5_000,
                ..Default::default()
            }
        );
        assert_eq!(cost.scratch_bytes, Some(0));
        assert_eq!(cost.confidence, EstimateConfidence::Prior);
        assert_eq!(cost.source, EstimateSource::StaticPrior);
    }
}

#[test]
fn digit_estimator_uses_provider_frozen_launch_for_queue_and_uniform_work() {
    let provider = crate::simple_provider::InProcessProvider::new();
    let query = query();
    let default = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(512, 65_535).unwrap();
    let selected = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(256, 65_535).unwrap();
    let baseline = wgpu_cost_estimate_with_geometry(&provider, &query, default).cost;
    let narrow = wgpu_cost_estimate_with_geometry(&provider, &query, selected).cost;
    assert_eq!(baseline.components.queue_ns, 20_000);
    assert_eq!(narrow.components.queue_ns, 30_000);
    assert_eq!(
        baseline.components.upload_ns,
        transfer_prior_ns(268_431_400)
    );
    assert_eq!(narrow.components.upload_ns, transfer_prior_ns(268_431_416));
    assert_eq!(
        baseline.components.execution_ns,
        narrow.components.execution_ns
    );
    assert_eq!(baseline.source, narrow.source);
    assert_eq!(baseline.confidence, narrow.confidence);
}
