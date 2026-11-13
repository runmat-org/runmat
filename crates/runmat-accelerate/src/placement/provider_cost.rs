use runmat_accelerate_api::{
    AccelProvider, ProviderCostEstimate, ProviderCostQuery, ProviderDispatchStats,
    ProviderOperationFamily,
};
use runmat_execution::{
    CandidatePreparationState, EstimateConfidence, EstimateSource, ExecutionCostComponents,
    ExecutionCostEstimate,
};

use super::local::transfer_prior_ns;

fn typed_input_stage_allocations(
    stage: &runmat_accelerate_api::reduction::ProviderReductionInputStage,
) -> u32 {
    use runmat_accelerate_api::reduction::ProviderReductionInputStage;
    match stage {
        ProviderReductionInputStage::BorrowResident { .. } => 0,
        ProviderReductionInputStage::ResidentCast { physical, .. } => {
            1_u32.saturating_add(physical.uniform_allocations)
        }
        ProviderReductionInputStage::UploadExact { .. }
        | ProviderReductionInputStage::HostCastThenUpload { .. } => 1,
    }
}

#[cfg(test)]
#[path = "provider_cost/floating_cast_tests.rs"]
mod floating_cast_tests;

#[cfg(all(test, feature = "wgpu"))]
mod digit_tests;

#[cfg(test)]
mod fusion_work_tests;

pub(crate) fn wgpu_cost_estimate_with_geometry(
    provider: &dyn AccelProvider,
    query: &ProviderCostQuery,
    linear_launch: crate::backend::wgpu::pipelines::FrozenLinearLaunch,
) -> ProviderCostEstimate {
    let matrix_work = super::provider_contract::matmul::physical_work(provider, &query.operation);
    let cast_work =
        super::provider_contract::floating_conversion::physical_work(provider, &query.operation);
    let select_work =
        super::provider_contract::resident_select::physical_work(provider, &query.operation);
    let sum_work = super::provider_contract::sum::physical_work(provider, &query.operation);
    let fusion_work =
        super::provider_contract::fusion_elementwise::physical_work(provider, &query.operation);
    let typed_reduction = {
        #[cfg(feature = "wgpu")]
        {
            super::typed_reduction_physical_work(&query.operation)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            None::<runmat_accelerate_api::ProviderResourceEstimate>
        }
    };
    let fusion_empty = fusion_work
        .as_ref()
        .is_some_and(|work| work.estimate.dispatches == Some(0));
    let remainder_work = {
        #[cfg(feature = "wgpu")]
        {
            super::provider_contract::remainder::wgpu_work(provider.precision(), &query.operation)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            None::<super::provider_contract::remainder::RemainderWork>
        }
    };
    let digit_work = {
        #[cfg(feature = "wgpu")]
        {
            super::provider_contract::rounding::wgpu_work(
                provider.precision(),
                &query.operation,
                linear_launch,
            )
        }
        #[cfg(not(feature = "wgpu"))]
        {
            None::<super::provider_contract::rounding::RoundDigitsWork>
        }
    };
    let power_work = {
        #[cfg(feature = "wgpu")]
        {
            super::provider_contract::power::wgpu_work(
                provider.precision(),
                provider,
                &query.operation,
            )
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
    dispatches = typed_reduction
        .as_ref()
        .and_then(|work| work.dispatches)
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
    if let Some(work) = &typed_reduction {
        queue_submissions = work.dispatches.unwrap_or(queue_submissions);
    }
    if let Some(work) = &unary_work {
        probe_elements = work.probe_elements;
        probe_bytes = work.temporary_bytes;
        readback_bytes = work.readback_bytes;
        dispatches = work.dispatches;
        readbacks = work.readbacks;
        queue_submissions = work.queue_submissions;
    }
    let mut allocations = dispatches;
    if let Some(work) = &typed_reduction {
        allocations = u32::from(work.output_bytes.unwrap_or(0) > 0)
            + u32::from(work.transient_bytes != work.output_bytes);
    }
    let typed_input_preparation = query
        .operation
        .workload
        .typed_reduction
        .as_ref()
        .and_then(|typed| typed.input_preparation.as_ref());
    let input_allocations = typed_input_preparation.map_or(0, |inventory| {
        inventory.entries().iter().fold(0_u32, |count, entry| {
            count.saturating_add(typed_input_stage_allocations(&entry.stage))
        })
    });
    allocations = allocations.saturating_add(input_allocations);
    let input_queue_submissions = typed_input_preparation.map_or(0, |inventory| {
        inventory.entries().iter().fold(0_u32, |count, entry| {
            count.saturating_add(match &entry.stage {
                runmat_accelerate_api::reduction::ProviderReductionInputStage::BorrowResident {
                    ..
                } => 0,
                runmat_accelerate_api::reduction::ProviderReductionInputStage::UploadExact {
                    upload,
                }
                | runmat_accelerate_api::reduction::ProviderReductionInputStage::HostCastThenUpload {
                    upload,
                    ..
                } => upload.completion_queue_slots,
                runmat_accelerate_api::reduction::ProviderReductionInputStage::ResidentCast {
                    physical,
                    ..
                } => physical.queue_submissions,
            })
        })
    });
    queue_submissions = queue_submissions.saturating_add(input_queue_submissions);
    let mut upload_bytes = query.required_upload_bytes;
    if let Some(work) = &coordinate_work {
        probe_bytes = work.scratch_bytes;
        queue_submissions = work.queue_submissions;
        allocations = work.allocations;
        upload_bytes = upload_bytes.saturating_add(work.extra_upload_bytes);
    }
    let download_bytes = if fusion_empty {
        0
    } else {
        query.required_download_bytes.saturating_add(readback_bytes)
    };
    if fusion_empty {
        upload_bytes = 0;
    }
    let synchronization_events = if fusion_empty {
        0
    } else if unary_work.is_some() {
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
        || remainder_work.is_some()
        || matrix_work.is_some()
        || cast_work.is_some()
        || select_work.is_some()
        || sum_work.is_some()
        || query.operation.workload.scalar_work_units.is_some()
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
                    // Declared scalar work already covers every operation and
                    // logical output element, across all kernel chunks. Adapter
                    // dispatches must not repeat that primary-program work.
                    operation_prior_ns(query).saturating_mul(
                        if fusion_work.is_some()
                            && query.operation.workload.scalar_work_units.is_none()
                        {
                            u64::from(dispatches)
                        } else {
                            1
                        },
                    )
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
    // Two ns per representation-conversion/copy visit is the same explicit
    // scalar prior used for ordinary elementwise work, not enqueue timing.
    // Undeclared-work callers retain their historical composite prior.
    let execution_ns = execution_ns.saturating_add(
        fusion_work
            .as_ref()
            .filter(|_| query.operation.workload.scalar_work_units.is_some())
            .map_or(0, |work| work.adapter_element_visits.saturating_mul(2)),
    );
    let input_conversion_units = typed_input_preparation.map_or(0, |inventory| {
        inventory.entries().iter().fold(0_u64, |units, entry| {
            units.saturating_add(match &entry.stage {
                runmat_accelerate_api::reduction::ProviderReductionInputStage::HostCastThenUpload {
                    conversion_elements,
                    ..
                } => *conversion_elements,
                runmat_accelerate_api::reduction::ProviderReductionInputStage::ResidentCast {
                    ..
                } => entry.prepared.checked_element_count().unwrap_or(u64::MAX),
                _ => 0,
            })
        })
    });
    let execution_ns = execution_ns.saturating_add(input_conversion_units.saturating_mul(2));
    let allocations = digit_work
        .as_ref()
        .map_or(allocations, |work| work.allocations);
    let queue_submissions = digit_work
        .as_ref()
        .map_or(queue_submissions, |work| work.queue_submissions);
    let scratch_bytes = fusion_work
        .as_ref()
        .map_or(probe_bytes, |work| work.scratch_bytes);
    let scratch_bytes = typed_reduction.as_ref().map_or(scratch_bytes, |work| {
        work.transient_bytes
            .unwrap_or(0)
            .saturating_sub(work.output_bytes.unwrap_or(0))
    });
    let scratch_bytes = digit_work
        .as_ref()
        .map_or(scratch_bytes, |work| work.scratch_bytes);
    let upload_bytes = upload_bytes.saturating_add(
        digit_work
            .as_ref()
            .map_or(0, |work| work.extra_upload_bytes),
    );
    let execution_ns = remainder_work.as_ref().map_or(execution_ns, |work| {
        if work.dispatches == 0 {
            0
        } else {
            work.execution_units.saturating_mul(2).max(5_000)
        }
    });
    let allocations = remainder_work
        .as_ref()
        .map_or(allocations, |work| work.allocations);
    let queue_submissions = remainder_work
        .as_ref()
        .map_or(queue_submissions, |work| work.queue_submissions);
    let scratch_bytes = remainder_work
        .as_ref()
        .map_or(scratch_bytes, |work| work.scratch_bytes);
    let upload_bytes = upload_bytes.saturating_add(
        remainder_work
            .as_ref()
            .map_or(0, |work| work.extra_upload_bytes),
    );
    let compile_or_prepare_ns = match query.preparation {
        CandidatePreparationState::Ready | CandidatePreparationState::Warm => 0,
        CandidatePreparationState::Cold => 100_000,
        CandidatePreparationState::Preparing => 50_000,
        CandidatePreparationState::Unknown => {
            unreachable!("unknown preparation is rejected at the provider cost boundary")
        }
    };
    let execution_ns = matrix_work
        .as_ref()
        .map_or(execution_ns, |work| work.scalar_units.saturating_mul(2));
    let allocations = matrix_work
        .as_ref()
        .map_or(allocations, |work| work.allocations);
    let queue_submissions = matrix_work
        .as_ref()
        .map_or(queue_submissions, |work| work.queue_submissions);
    let scratch_bytes = matrix_work
        .as_ref()
        .map_or(scratch_bytes, |work| work.scratch_bytes);
    let upload_bytes =
        upload_bytes.saturating_add(matrix_work.as_ref().map_or(0, |work| work.uniform_bytes));
    let compile_or_prepare_ns = if matrix_work
        .as_ref()
        .is_some_and(|work| work.estimate.dispatches == Some(0))
    {
        0
    } else {
        compile_or_prepare_ns
    };
    // Readiness is exact cache state; cold timing remains the existing explicit
    // StaticPrior, not a claimed measurement. Count distinct cold recipes, not
    // every invocation of a shared cast pipeline. Contention is unknown/one
    // conservative prior, never a fabricated warm hit.
    let compile_or_prepare_ns = match &query.context.preparation {
        Some(snapshot) => match snapshot.state {
            Some(CandidatePreparationState::Ready | CandidatePreparationState::Warm) => 0,
            Some(CandidatePreparationState::Cold) => {
                100_000u64.saturating_mul(u64::from(snapshot.cold_pipelines))
            }
            Some(CandidatePreparationState::Preparing | CandidatePreparationState::Unknown)
            | None => compile_or_prepare_ns,
        },
        None => compile_or_prepare_ns,
    };
    let compile_or_prepare_ns = if fusion_empty {
        0
    } else {
        compile_or_prepare_ns
    };
    let compile_or_prepare_ns = if digit_work.as_ref().is_some_and(|work| work.dispatches == 0) {
        0
    } else {
        compile_or_prepare_ns
    };
    let compile_or_prepare_ns = if remainder_work
        .as_ref()
        .is_some_and(|work| work.dispatches == 0)
    {
        0
    } else {
        compile_or_prepare_ns
    };
    let execution_ns = cast_work
        .as_ref()
        .map_or(execution_ns, |work| work.scalar_visits.saturating_mul(2));
    let allocations = cast_work
        .as_ref()
        .map_or(allocations, |work| work.allocations);
    let queue_submissions = cast_work.as_ref().map_or(queue_submissions, |work| {
        work.estimate.dispatches.unwrap_or(0)
    });
    let scratch_bytes = cast_work
        .as_ref()
        .map_or(scratch_bytes, |work| work.scratch_bytes);
    let upload_bytes =
        upload_bytes.saturating_add(cast_work.as_ref().map_or(0, |work| work.uniform_bytes));
    let compile_or_prepare_ns = if cast_work
        .as_ref()
        .is_some_and(|work| work.estimate.dispatches == Some(0))
    {
        0
    } else {
        compile_or_prepare_ns
    };
    // Selection is one fused broadcast route with one output and one uniform
    // allocation. Generic elementwise timing cannot stand in for its
    // three-input physical route.
    let execution_ns = select_work
        .as_ref()
        .map_or(execution_ns, |work| work.scalar_visits.saturating_mul(2));
    let allocations = select_work.as_ref().map_or(allocations, |_| 2);
    let queue_submissions = select_work
        .as_ref()
        .map_or(queue_submissions, |work| work.queue_submissions);
    let scratch_bytes = select_work
        .as_ref()
        .map_or(scratch_bytes, |work| work.uniform_bytes);
    let upload_bytes =
        upload_bytes.saturating_add(select_work.as_ref().map_or(0, |work| work.uniform_bytes));
    let execution_ns = sum_work.as_ref().map_or(execution_ns, |work| {
        work.scalar_visits.saturating_mul(3).max(8_000)
    });
    let allocations = sum_work
        .as_ref()
        .map_or(allocations, |work| work.allocations);
    let queue_submissions = sum_work
        .as_ref()
        .map_or(queue_submissions, |work| work.queue_submissions);
    let scratch_bytes = sum_work
        .as_ref()
        .map_or(scratch_bytes, |work| work.uniform_bytes);
    let upload_bytes =
        upload_bytes.saturating_add(sum_work.as_ref().map_or(0, |work| work.uniform_bytes));
    // Fence clear is in the existing command buffer, not another submission.
    // Allocation and map/progress service reuse the existing component priors;
    // they are not GPU timestamps or separately measured service latency.
    let observation = query.context.observation;
    let fence_allocations = observation.map_or(0, |work| u64::from(work.submissions));
    let completion_services = observation.map_or(0, |work| u64::from(work.map_callbacks));
    let clear_units = observation.map_or(0, |work| u64::from(work.clear_commands));
    let optional_query = observation.and_then(|work| work.wgpu_compute_query);
    // One QuerySet plus resolve and staging buffers per eligible encoder.
    // QuerySet's native bytes are opaque; only buffer requests enter scratch.
    let query_allocations = optional_query.map_or(0, |work| u64::from(work.max_query_sets) * 3);
    let query_map_services = optional_query.map_or(0, |work| u64::from(work.max_map_callbacks));
    let scratch_bytes = scratch_bytes
        .saturating_add(observation.map_or(0, |work| work.requested_peak_buffer_bytes))
        .saturating_add(optional_query.map_or(0, |work| work.requested_peak_buffer_bytes));
    ProviderCostEstimate {
        cost: ExecutionCostEstimate {
            components: ExecutionCostComponents {
                compile_or_prepare_ns,
                upload_ns: transfer_prior_ns(upload_bytes),
                allocation_ns: 5_000u64.saturating_mul(
                    u64::from(allocations)
                        .saturating_add(fence_allocations)
                        .saturating_add(query_allocations),
                ),
                queue_ns: 10_000u64.saturating_mul(u64::from(queue_submissions)),
                execution_ns: execution_ns.saturating_add(clear_units.saturating_mul(2)),
                synchronization_ns: 5_000u64.saturating_mul(
                    synchronization_events
                        .saturating_add(completion_services)
                        .saturating_add(query_map_services),
                ),
                download_ns: transfer_prior_ns(download_bytes),
                downstream_ns: if query.downstream_materialization && !fusion_empty {
                    5_000
                } else {
                    0
                },
            },
            scratch_bytes: Some(scratch_bytes),
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

#[cfg(test)]
pub(crate) fn wgpu_cost_estimate(
    provider: &dyn AccelProvider,
    query: &ProviderCostQuery,
) -> ProviderCostEstimate {
    let launch = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(
        crate::backend::wgpu::config::WORKGROUP_SIZE,
        crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS,
    )
    .expect("default linear launch");
    wgpu_cost_estimate_with_geometry(provider, query, launch)
}

fn average_ns(stats: ProviderDispatchStats) -> Option<u64> {
    (stats.count > 0).then(|| stats.total_wall_time_ns / stats.count)
}

fn operation_prior_ns(query: &ProviderCostQuery) -> u64 {
    let elements = query.operation.workload.elements.unwrap_or(1);
    let flops = query.operation.workload.flops.unwrap_or(elements);
    match query.operation.family {
        ProviderOperationFamily::Upload | ProviderOperationFamily::Download => 0,
        ProviderOperationFamily::Fusion => match query.operation.workload.scalar_work_units {
            Some(0) => 0,
            Some(units) => units.max(5_000),
            None => elements.max(5_000),
        },
        ProviderOperationFamily::Graph => elements.max(5_000),
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

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn typed_cast_allocation_prior_counts_selected_uniform_allocations() {
        use runmat_accelerate_api::reduction::ProviderReductionInputStage;
        use runmat_accelerate_api::{
            ProviderRealElementwiseOutputRoute, ProviderRealElementwisePhysicalProposal,
        };
        let mut physical = ProviderRealElementwisePhysicalProposal {
            output_route: ProviderRealElementwiseOutputRoute::FreshAllocation,
            output_buffer_bytes: 8,
            workgroup_size: 64,
            chunk_capacity: 2,
            uniform_allocations: 2,
            uniform_buffer_bytes: 16,
            dispatches: 2,
            queue_submissions: 2,
            pipeline_key: Some(1),
        };
        let recipe_identity = runmat_execution::Digest::sha256(b"cast");
        assert_eq!(
            typed_input_stage_allocations(&ProviderReductionInputStage::ResidentCast {
                device_id: 7,
                recipe_identity,
                physical: physical.clone(),
            }),
            3
        );
        physical.uniform_allocations = 4;
        assert_eq!(
            typed_input_stage_allocations(&ProviderReductionInputStage::ResidentCast {
                device_id: 7,
                recipe_identity,
                physical,
            }),
            5
        );
        assert_eq!(
            typed_input_stage_allocations(&ProviderReductionInputStage::BorrowResident {
                device_id: 7,
            }),
            0
        );
    }

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
            context: Default::default(),
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        };
        let estimate = wgpu_cost_estimate(&provider, &query).cost;
        assert_eq!(estimate.components.synchronization_ns, 10_000);
        assert_eq!(estimate.components.download_ns, transfer_prior_ns(16));
        assert_eq!(estimate.components.queue_ns, 80_000);
        assert_eq!(estimate.scratch_bytes, Some(144));
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
            context: Default::default(),
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        };
        let estimate = wgpu_cost_estimate(&provider, &query).cost;
        assert!(estimate.components.synchronization_ns > 0);
        assert_eq!(estimate.components.download_ns, transfer_prior_ns(8));
        assert_eq!(estimate.scratch_bytes, Some(40));
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
                context: Default::default(),
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
        use crate::placement::provider_contract::fusion_elementwise::fixture::GeometryProvider;
        use runmat_accelerate_api::{
            ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency,
            ProviderStorage,
        };
        let provider = GeometryProvider::new();
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
            context: Default::default(),
            required_upload_bytes: 64,
            required_download_bytes: 0,
            downstream_materialization: false,
        };
        let adapted = wgpu_cost_estimate(&provider, &query).cost;
        // One fused dispatch, one output adapter, and the pipeline warmup.
        assert_eq!(adapted.components.queue_ns, 30_000);
        assert_eq!(adapted.components.allocation_ns, 10_000);
        assert_eq!(adapted.components.upload_ns, transfer_prior_ns(64));
        assert_eq!(adapted.scratch_bytes, Some(128));
        assert_eq!(adapted.source, EstimateSource::StaticPrior);
        query.operation.outputs[0].element_type = ProviderElementType::F64;
        let native = wgpu_cost_estimate(&provider, &query).cost;
        // The native output omits the adapter but retains the warmup.
        assert_eq!(native.components.queue_ns, 20_000);
        assert!(adapted.components.execution_ns > native.components.execution_ns);
    }
}
