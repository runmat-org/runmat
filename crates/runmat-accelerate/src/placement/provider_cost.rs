use runmat_accelerate_api::{
    AccelProvider, ProviderCostEstimate, ProviderCostQuery, ProviderDispatchStats,
    ProviderOperationFamily,
};
use runmat_execution::{
    CandidatePreparationState, EstimateConfidence, EstimateSource, ExecutionCostComponents,
    ExecutionCostEstimate,
};

use super::local::transfer_prior_ns;

pub(crate) fn wgpu_cost_estimate(
    provider: &dyn AccelProvider,
    query: &ProviderCostQuery,
) -> ProviderCostEstimate {
    let power_work = {
        #[cfg(feature = "wgpu")]
        {
            super::provider_contract::power::wgpu_work(provider.precision(), &query.operation)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            None::<super::provider_contract::power::PowerWork>
        }
    };
    let (probe_elements, probe_bytes, readback_bytes, dispatches) =
        power_work.as_ref().map_or((0, 0, 0, 1), |work| {
            (
                work.elements,
                work.temporary_bytes,
                work.readback_bytes,
                work.dispatches,
            )
        });
    let download_bytes = query.required_download_bytes.saturating_add(readback_bytes);
    let telemetry = provider.telemetry_snapshot();
    let stats = match query.operation.family {
        ProviderOperationFamily::MatrixMultiply => telemetry.matmul,
        ProviderOperationFamily::Reduction => telemetry.fused_reduction,
        ProviderOperationFamily::Fusion | ProviderOperationFamily::Graph => {
            telemetry.fused_elementwise
        }
        ProviderOperationFamily::Elementwise => telemetry.fused_elementwise,
        ProviderOperationFamily::Library
        | ProviderOperationFamily::Upload
        | ProviderOperationFamily::Download => ProviderDispatchStats::default(),
    };
    // A single elementwise enqueue sample does not measure the composite
    // classification/reduction/readback/kernel route. Keep its prior explicit
    // until the existing invocation feedback supplies a matching observation.
    let observed_execution_ns = if power_work.is_some() {
        None
    } else {
        average_ns(stats)
    };
    let execution_ns = observed_execution_ns
        .unwrap_or_else(|| operation_prior_ns(query))
        .saturating_add(probe_elements.saturating_mul(3));
    let compile_or_prepare_ns = match query.preparation {
        CandidatePreparationState::Ready | CandidatePreparationState::Warm => 0,
        CandidatePreparationState::Cold => 100_000,
        CandidatePreparationState::Preparing => 50_000,
    };
    ProviderCostEstimate {
        cost: ExecutionCostEstimate {
            components: ExecutionCostComponents {
                compile_or_prepare_ns,
                upload_ns: transfer_prior_ns(query.required_upload_bytes),
                allocation_ns: 5_000u64.saturating_mul(u64::from(dispatches)),
                queue_ns: 10_000u64.saturating_mul(u64::from(dispatches)),
                execution_ns,
                synchronization_ns: if download_bytes == 0 { 0 } else { 5_000 },
                download_ns: transfer_prior_ns(download_bytes),
                downstream_ns: if query.downstream_materialization {
                    5_000
                } else {
                    0
                },
            },
            scratch_bytes: probe_bytes,
            confidence: if observed_execution_ns.is_some() {
                EstimateConfidence::Medium
            } else {
                EstimateConfidence::Prior
            },
            source: if observed_execution_ns.is_some() {
                EstimateSource::Provider
            } else {
                EstimateSource::StaticPrior
            },
        },
    }
}

fn average_ns(stats: ProviderDispatchStats) -> Option<u64> {
    (stats.count > 0).then(|| stats.total_wall_time_ns / stats.count)
}

fn operation_prior_ns(query: &ProviderCostQuery) -> u64 {
    let elements = query.operation.workload.elements.unwrap_or(1);
    let flops = query.operation.workload.flops.unwrap_or(elements);
    match query.operation.family {
        ProviderOperationFamily::Upload | ProviderOperationFamily::Download => 0,
        ProviderOperationFamily::Fusion | ProviderOperationFamily::Graph => elements.max(5_000),
        ProviderOperationFamily::Elementwise => elements.saturating_mul(2).max(5_000),
        ProviderOperationFamily::Reduction => elements.saturating_mul(3).max(8_000),
        ProviderOperationFamily::MatrixMultiply | ProviderOperationFamily::Library => {
            flops.saturating_add(999) / 1_000 + 10_000
        }
    }
}

#[cfg(test)]
mod tests {
    use runmat_accelerate_api::{
        ProviderFeasibilityQuery, ProviderOperationIdentity, ProviderWorkload,
    };

    use super::*;

    #[cfg(feature = "wgpu")]
    #[test]
    fn power_domain_probe_requires_readback_even_when_result_stays_resident() {
        use runmat_accelerate_api::{
            NumericalDomainEvidence, PowerResultDomain, ProviderElementType, ProviderLayout,
            ProviderRepresentation, ProviderResidency, ProviderStorage,
        };
        let provider = crate::simple_provider::InProcessProvider::new();
        let input = ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 3],
            residency: ProviderResidency::Host,
        };
        let query = ProviderCostQuery {
            operation: ProviderFeasibilityQuery {
                operation: ProviderOperationIdentity::new(
                    runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
                ),
                family: ProviderOperationFamily::Elementwise,
                inputs: vec![input.clone(), input],
                outputs: vec![],
                workload: ProviderWorkload {
                    elements: Some(3),
                    numerical_domain: NumericalDomainEvidence::RealPower(
                        PowerResultDomain::Complex,
                    ),
                    ..Default::default()
                },
            },
            preparation: CandidatePreparationState::Warm,
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        };
        let estimate = wgpu_cost_estimate(&provider, &query).cost;
        assert!(estimate.components.synchronization_ns > 0);
        assert_eq!(estimate.components.download_ns, transfer_prior_ns(8));
        assert_eq!(estimate.scratch_bytes, 40);
        assert_eq!(estimate.components.queue_ns, 40_000);
        assert_eq!(estimate.source, EstimateSource::StaticPrior);
    }

    #[test]
    fn cold_and_transfer_components_are_explicit() {
        let provider = crate::simple_provider::InProcessProvider::new();
        let estimate = wgpu_cost_estimate(
            &provider,
            &ProviderCostQuery {
                operation: ProviderFeasibilityQuery {
                    operation: ProviderOperationIdentity::new("test.elementwise"),
                    family: ProviderOperationFamily::Elementwise,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    workload: ProviderWorkload {
                        elements: Some(10_000),
                        ..ProviderWorkload::default()
                    },
                },
                preparation: CandidatePreparationState::Cold,
                required_upload_bytes: 8_000,
                required_download_bytes: 4_000,
                downstream_materialization: true,
            },
        );
        assert_eq!(estimate.cost.components.compile_or_prepare_ns, 100_000);
        assert!(estimate.cost.components.upload_ns > 0);
        assert!(estimate.cost.components.download_ns > 0);
        assert_eq!(estimate.cost.components.downstream_ns, 5_000);
    }
}
