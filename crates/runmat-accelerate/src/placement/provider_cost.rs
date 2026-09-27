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
    let fusion_work =
        super::provider_contract::fusion_elementwise::physical_work(provider, &query.operation);
    let digit_work = {
        #[cfg(feature = "wgpu")]
        {
            super::provider_contract::rounding::wgpu_work(provider.precision(), &query.operation)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            None::<super::provider_contract::rounding::RoundDigitsWork>
        }
    };
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
    let unary_work = {
        #[cfg(feature = "wgpu")]
        {
            super::provider_contract::trigonometry::inverse::wgpu_work(
                provider.precision(),
                &query.operation,
            )
            .or_else(|| {
                super::provider_contract::logarithm_roots::wgpu_work(
                    provider.precision(),
                    &query.operation,
                )
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            None::<super::provider_contract::real_unary::UnaryWork>
        }
    };
    let (mut probe_elements, mut probe_bytes, mut readback_bytes, mut dispatches) =
        power_work.as_ref().map_or((0, 0, 0, 1), |work| {
            (
                work.elements,
                work.temporary_bytes,
                work.readback_bytes,
                work.dispatches,
            )
        });
    dispatches = fusion_work
        .as_ref()
        .and_then(|work| work.estimate.dispatches)
        .unwrap_or(dispatches);
    let coordinate_work = {
        #[cfg(feature = "wgpu")]
        {
            super::provider_contract::trigonometry::pol2cart::wgpu_work(
                provider.precision(),
                &query.operation,
            )
        }
        #[cfg(not(feature = "wgpu"))]
        {
            None::<super::provider_contract::trigonometry::pol2cart::CoordinateWork>
        }
    };
    let mut readbacks = u32::from(readback_bytes > 0);
    let mut queue_submissions = fusion_work
        .as_ref()
        .map_or(dispatches, |work| work.queue_submissions);
    if let Some(work) = &unary_work {
        probe_elements = work.probe_elements;
        probe_bytes = work.temporary_bytes;
        readback_bytes = work.readback_bytes;
        dispatches = work.dispatches;
        readbacks = work.readbacks;
        queue_submissions = work.queue_submissions;
    }
    let mut allocations = dispatches;
    let mut upload_bytes = query.required_upload_bytes;
    if let Some(work) = &coordinate_work {
        probe_bytes = work.scratch_bytes;
        queue_submissions = work.queue_submissions;
        allocations = work.allocations;
        upload_bytes = upload_bytes.saturating_add(work.extra_upload_bytes);
    }
    let download_bytes = query.required_download_bytes.saturating_add(readback_bytes);
    let synchronization_events = if unary_work.is_some() {
        u64::from(readbacks) + u64::from(query.required_download_bytes > 0)
    } else {
        u64::from(download_bytes > 0)
    };
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
    let observed_execution_ns = if power_work.is_some()
        || unary_work.is_some()
        || fusion_work.is_some()
        || coordinate_work.is_some()
        || digit_work.is_some()
    {
        None
    } else {
        average_ns(stats)
    };
    let execution_ns = coordinate_work
        .as_ref()
        .map(|work| work.execution_elements.saturating_mul(2).max(5_000))
        .unwrap_or_else(|| {
            observed_execution_ns
                .unwrap_or_else(|| {
                    operation_prior_ns(query).saturating_mul(if fusion_work.is_some() {
                        u64::from(dispatches)
                    } else {
                        1
                    })
                })
                .saturating_add(probe_elements.saturating_mul(3))
        });
    let execution_ns = digit_work.as_ref().map_or(execution_ns, |work| {
        if work.dispatches == 0 {
            0
        } else {
            work.execution_units.saturating_mul(2).max(5_000)
        }
    });
    let allocations = digit_work
        .as_ref()
        .map_or(allocations, |work| work.allocations);
    let queue_submissions = digit_work
        .as_ref()
        .map_or(queue_submissions, |work| work.queue_submissions);
    let scratch_bytes = fusion_work
        .as_ref()
        .map_or(probe_bytes, |work| work.scratch_bytes);
    let scratch_bytes = digit_work
        .as_ref()
        .map_or(scratch_bytes, |work| work.scratch_bytes);
    let upload_bytes = upload_bytes.saturating_add(
        digit_work
            .as_ref()
            .map_or(0, |work| work.extra_upload_bytes),
    );
    let compile_or_prepare_ns = match query.preparation {
        CandidatePreparationState::Ready | CandidatePreparationState::Warm => 0,
        CandidatePreparationState::Cold => 100_000,
        CandidatePreparationState::Preparing => 50_000,
    };
    ProviderCostEstimate {
        cost: ExecutionCostEstimate {
            components: ExecutionCostComponents {
                compile_or_prepare_ns,
                upload_ns: transfer_prior_ns(upload_bytes),
                allocation_ns: 5_000u64.saturating_mul(u64::from(allocations)),
                queue_ns: 10_000u64.saturating_mul(u64::from(queue_submissions)),
                execution_ns,
                synchronization_ns: 5_000u64.saturating_mul(synchronization_events),
                download_ns: transfer_prior_ns(download_bytes),
                downstream_ns: if query.downstream_materialization {
                    5_000
                } else {
                    0
                },
            },
            scratch_bytes,
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
    fn inverse_domain_cost_includes_both_readbacks_and_all_probe_passes() {
        use runmat_accelerate_api::{
            NumericalDomainEvidence, ProviderElementType, ProviderLayout, ProviderRepresentation,
            ProviderResidency, ProviderStorage,
        };
        use runmat_types::{RealUnaryClassification, RealUnaryPredicate};
        let provider = crate::simple_provider::InProcessProvider::new();
        let query = ProviderCostQuery {
            operation: ProviderFeasibilityQuery {
                operation: ProviderOperationIdentity::new(
                    runmat_builtins::ASIN_CATALOG_ENTRY.identity.name,
                ),
                family: ProviderOperationFamily::Elementwise,
                inputs: vec![ProviderRepresentation {
                    element_type: ProviderElementType::F64,
                    storage: ProviderStorage::DenseReal,
                    layout: ProviderLayout::ColumnMajorContiguous,
                    shape: vec![1, 1025],
                    residency: ProviderResidency::Device,
                }],
                outputs: vec![],
                workload: ProviderWorkload {
                    elements: Some(1025),
                    numerical_domain: NumericalDomainEvidence::RealUnary {
                        predicate: RealUnaryPredicate::ClosedUnitInterval,
                        classification: RealUnaryClassification::Inside,
                    },
                    ..Default::default()
                },
            },
            preparation: CandidatePreparationState::Warm,
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        };
        let estimate = wgpu_cost_estimate(&provider, &query).cost;
        assert_eq!(estimate.components.synchronization_ns, 10_000);
        assert_eq!(estimate.components.download_ns, transfer_prior_ns(16));
        assert_eq!(estimate.components.queue_ns, 80_000);
        assert_eq!(estimate.scratch_bytes, 144);
        assert_eq!(estimate.source, EstimateSource::StaticPrior);
    }

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

    #[test]
    fn fused_output_adapter_dispatches_and_physical_upload_bytes_reach_costing() {
        use runmat_accelerate_api::{
            ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency,
            ProviderStorage,
        };
        let provider = crate::simple_provider::InProcessProvider::new();
        let input = ProviderRepresentation {
            element_type: ProviderElementType::F32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 8],
            residency: ProviderResidency::Host,
        };
        let mut output = input.clone();
        output.residency = ProviderResidency::Device;
        let mut query = ProviderCostQuery {
            operation: ProviderFeasibilityQuery {
                operation: ProviderOperationIdentity::new("fusion.elementwise"),
                family: ProviderOperationFamily::Fusion,
                inputs: vec![input],
                outputs: vec![output],
                workload: ProviderWorkload {
                    elements: Some(8),
                    ..Default::default()
                },
            },
            preparation: CandidatePreparationState::Warm,
            required_upload_bytes: 64,
            required_download_bytes: 0,
            downstream_materialization: false,
        };
        let adapted = wgpu_cost_estimate(&provider, &query).cost;
        assert_eq!(adapted.components.queue_ns, 20_000);
        assert_eq!(adapted.components.allocation_ns, 10_000);
        assert_eq!(adapted.components.upload_ns, transfer_prior_ns(64));
        assert_eq!(adapted.scratch_bytes, 128);
        assert_eq!(adapted.source, EstimateSource::StaticPrior);
        query.operation.outputs[0].element_type = ProviderElementType::F64;
        let native = wgpu_cost_estimate(&provider, &query).cost;
        assert_eq!(native.components.queue_ns, 10_000);
        assert!(adapted.components.execution_ns > native.components.execution_ns);
    }
}
