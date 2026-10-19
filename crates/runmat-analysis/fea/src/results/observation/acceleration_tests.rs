use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Mutex,
};

use crate::planning::resource::FeaResourcePlanIdentity;
use runmat_accelerate_api::{
    checked_f64_bytes, AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider,
    AccelProviderFuture, ApiDeviceInfo, DenseReductionExecutionEvidence, DenseReductionOperation,
    DenseReductionRequest, DenseReductionResult, GpuTensorHandle, GpuTensorStorage,
    HostNumericDataOwned, HostNumericTensorOwned, HostNumericTensorView, HostTensorOwned,
    HostTensorView, NumericElementType, ProviderCapabilityOperation, ProviderCapabilitySnapshot,
    ProviderConcurrencyCapabilities, ProviderCostEstimate, ProviderCostQuery, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderRejectionCode, ProviderResourceEstimate, SpawnHandleConcurrency,
    DENSE_REDUCTION_EVIDENCE_SCHEMA_VERSION, GOVERNED_DENSE_REDUCTION_OPERATION,
    PROVIDER_CAPABILITY_SCHEMA_VERSION,
};
use runmat_analysis_core::{
    ExtremaTiePolicy, ObservationId, ObservationRequest, ObservationTarget, ReductionPolicy,
    SeriesReduction, SpatialReduction,
};
use runmat_execution::{resource::AcceleratorRequest, Digest, ExecutionCostComponents};

use super::{
    acceleration::{
        classify_observation_partition_acceleration, evaluate_observation_partition_with_provider,
        governed_extrema_device_resident_bytes, ObservationAccelerationEligibility,
        ObservationProviderAccelerationInput,
    },
    distributed::combine_observation_partitions,
    evaluate_solid_observation_partition,
    partition_tests::{prepared, spec},
    prepare_solid_observation_field_partition, ObservationAccelerationSummary,
    ObservationExecutionPolicy, ObservationPartitionContribution, ObservationPartitionPlan,
    OBSERVATION_ACCELERATION_EVIDENCE_MAX_ENCODED_BYTES,
};

fn allocation() -> AcceleratorRequest {
    AcceleratorRequest {
        class: "test-gpu".into(),
        count: 1,
        memory_bytes_each: 1024 * 1024,
    }
}

fn resource_plan_identity() -> FeaResourcePlanIdentity {
    FeaResourcePlanIdentity(Digest::sha256(b"observation-acceleration-resource-plan"))
}

#[test]
fn governed_extrema_preserve_observation_identity_and_publish_execution_evidence() {
    let mut maximum = spec("accelerated_maximum", ObservationTarget::WholeModel);
    maximum.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Maximum,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let (projection, plan, field) = prepared(ObservationRequest::Reduction(maximum));
    let partition_plan = ObservationPartitionPlan::derive(
        &plan,
        &ObservationId("accelerated_maximum".into()),
        ObservationExecutionPolicy::serial(),
    )
    .unwrap();
    let partition = &partition_plan.partitions[0];
    let source = prepare_solid_observation_field_partition(
        &plan,
        &partition_plan,
        partition,
        &projection,
        &field,
    )
    .unwrap();
    assert_eq!(
        classify_observation_partition_acceleration(
            &plan,
            &partition_plan,
            partition,
            &source,
            None,
        )
        .unwrap(),
        ObservationAccelerationEligibility::GovernedExactF64Extrema
    );
    let required_device_bytes =
        governed_extrema_device_resident_bytes(partition.target.entity_count()).unwrap();
    let mut exact_allocation = allocation();
    exact_allocation.memory_bytes_each = required_device_bytes;
    let exact_provider = GovernedReductionTestProvider::default();
    evaluate_observation_partition_with_provider(
        &plan,
        &partition_plan,
        partition,
        ObservationProviderAccelerationInput {
            field: &source,
            event_attribution: None,
            allocation: &exact_allocation,
            resource_plan: resource_plan_identity(),
            provider: &exact_provider,
        },
    )
    .unwrap();
    let mut insufficient_allocation = exact_allocation;
    insufficient_allocation.memory_bytes_each -= 1;
    let insufficient_provider = GovernedReductionTestProvider::default();
    assert!(evaluate_observation_partition_with_provider(
        &plan,
        &partition_plan,
        partition,
        ObservationProviderAccelerationInput {
            field: &source,
            event_attribution: None,
            allocation: &insufficient_allocation,
            resource_plan: resource_plan_identity(),
            provider: &insufficient_provider,
        },
    )
    .is_err());
    assert!(insufficient_provider.values.lock().unwrap().is_empty());
    let host = evaluate_solid_observation_partition(
        &plan,
        &partition_plan,
        partition,
        &projection,
        &source,
        None,
    )
    .unwrap();
    let provider = GovernedReductionTestProvider::default();
    let accelerated = evaluate_observation_partition_with_provider(
        &plan,
        &partition_plan,
        partition,
        ObservationProviderAccelerationInput {
            field: &source,
            event_attribution: None,
            allocation: &allocation(),
            resource_plan: resource_plan_identity(),
            provider: &provider,
        },
    )
    .unwrap();
    assert_eq!(accelerated.identity, host.identity);
    assert_eq!(accelerated.payload, host.payload);
    assert!(accelerated.acceleration.is_some());
    assert_eq!(
        accelerated.acceleration.as_ref().unwrap().allocation,
        allocation()
    );
    let evidence = accelerated.acceleration.as_ref().unwrap();
    evidence.validate_for_allocation(&allocation()).unwrap();
    assert_eq!(evidence.resource_plan, resource_plan_identity());
    let evidence_growth = accelerated
        .canonical_bytes()
        .unwrap()
        .len()
        .checked_sub(host.canonical_bytes().unwrap().len())
        .unwrap();
    assert!(
        u64::try_from(evidence_growth).unwrap()
            <= OBSERVATION_ACCELERATION_EVIDENCE_MAX_ENCODED_BYTES
    );
    let summary = ObservationAccelerationSummary::from_evidence(std::slice::from_ref(evidence))
        .unwrap()
        .unwrap();
    let encoded_summary = serde_json::to_vec(&summary).unwrap();
    assert_eq!(
        serde_json::from_slice::<ObservationAccelerationSummary>(&encoded_summary).unwrap(),
        summary
    );
    let mut tampered_summary = serde_json::to_value(&summary).unwrap();
    tampered_summary["operations"][0]["execution_ns"] = serde_json::json!(0);
    assert!(serde_json::from_value::<ObservationAccelerationSummary>(tampered_summary).is_err());
    let mut substituted = allocation();
    substituted.class = "substituted-gpu".into();
    assert!(evidence.validate_for_allocation(&substituted).is_err());
    let mut undersized = allocation();
    undersized.memory_bytes_each = 1;
    assert!(evidence.validate_for_allocation(&undersized).is_err());
    assert_eq!(
        ObservationPartitionContribution::from_canonical_bytes(
            &accelerated.canonical_bytes().unwrap()
        )
        .unwrap(),
        accelerated
    );

    let host_result = combine_observation_partitions(&plan, &partition_plan, vec![host]).unwrap();
    let accelerated_result =
        combine_observation_partitions(&plan, &partition_plan, vec![accelerated.clone()]).unwrap();
    assert_eq!(
        accelerated_result.logical_identity,
        host_result.logical_identity
    );
    assert_eq!(accelerated_result.field, host_result.field);
    assert_ne!(
        accelerated_result.partition_evidence.identity,
        host_result.partition_evidence.identity
    );
    assert_eq!(accelerated_result.partition_evidence.accelerations.len(), 1);
    assert_eq!(
        &accelerated_result.partition_evidence.accelerations[0],
        evidence
    );
    assert_eq!(provider.live_handle_count(), 0);

    let mut tampered = accelerated;
    tampered.acceleration.as_mut().unwrap().input_upload_bytes += 8;
    assert!(tampered.validate().is_err());
}

#[test]
fn governed_extrema_reject_unsupported_work_before_provider_allocation() {
    for (ties, reject_provider) in [
        (ExtremaTiePolicy::AllLocations, false),
        (ExtremaTiePolicy::FirstCanonical, true),
    ] {
        let mut maximum = spec("rejected_acceleration", ObservationTarget::WholeModel);
        maximum.reduction = Some(ReductionPolicy {
            spatial: SpatialReduction::Maximum,
            series: SeriesReduction::None,
            extrema_ties: ties,
        });
        let (projection, plan, field) = prepared(ObservationRequest::Reduction(maximum));
        let partition_plan = ObservationPartitionPlan::derive(
            &plan,
            &ObservationId("rejected_acceleration".into()),
            ObservationExecutionPolicy::serial(),
        )
        .unwrap();
        let partition = &partition_plan.partitions[0];
        let source = prepare_solid_observation_field_partition(
            &plan,
            &partition_plan,
            partition,
            &projection,
            &field,
        )
        .unwrap();
        let provider = GovernedReductionTestProvider::default();
        provider.reject.store(reject_provider, Ordering::Relaxed);
        if !reject_provider {
            assert_eq!(
                classify_observation_partition_acceleration(
                    &plan,
                    &partition_plan,
                    partition,
                    &source,
                    None,
                )
                .unwrap(),
                ObservationAccelerationEligibility::Host
            );
        }
        assert!(evaluate_observation_partition_with_provider(
            &plan,
            &partition_plan,
            partition,
            ObservationProviderAccelerationInput {
                field: &source,
                event_attribution: None,
                allocation: &allocation(),
                resource_plan: resource_plan_identity(),
                provider: &provider,
            },
        )
        .is_err());
        assert_eq!(provider.next.load(Ordering::Relaxed), 0);
        assert_eq!(provider.live_handle_count(), 0);
    }
}

#[derive(Default)]
struct GovernedReductionTestProvider {
    next: AtomicU64,
    reject: AtomicBool,
    values: Mutex<BTreeMap<u64, HostNumericTensorOwned>>,
}

impl GovernedReductionTestProvider {
    const DEVICE_ID: u32 = 91;

    fn insert(&self, values: Vec<f64>, shape: Vec<usize>) -> GpuTensorHandle {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.values.lock().unwrap().insert(
            id,
            HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(values),
                shape: shape.clone(),
                storage: GpuTensorStorage::Real,
            },
        );
        GpuTensorHandle::new(shape, Self::DEVICE_ID, id)
            .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real)
    }

    fn numeric(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostNumericTensorOwned> {
        self.values
            .lock()
            .unwrap()
            .get(&handle.buffer_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("test provider handle is absent"))
    }

    fn live_handle_count(&self) -> usize {
        self.values.lock().unwrap().len()
    }
}

impl AccelProvider for GovernedReductionTestProvider {
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.insert(host.data.to_vec(), host.shape.to_vec()))
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let numeric = self.numeric(handle)?;
            let HostNumericDataOwned::F64(data) = numeric.data else {
                unreachable!("test provider stores only f64")
            };
            Ok(HostTensorOwned {
                data,
                shape: numeric.shape,
                storage: numeric.storage,
            })
        })
    }

    fn upload_numeric(&self, host: &HostNumericTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        host.validate()?;
        let runmat_accelerate_api::HostNumericDataView::F64(values) = host.data else {
            anyhow::bail!("test provider accepts only f64")
        };
        anyhow::ensure!(
            host.storage == GpuTensorStorage::Real,
            "real storage required"
        );
        Ok(self.insert(values.to_vec(), host.shape.to_vec()))
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move { self.numeric(handle) })
    }

    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.values
                .lock()
                .unwrap()
                .remove(&handle.buffer_id)
                .is_some(),
            "test provider handle was already released"
        );
        Ok(())
    }

    fn device_info(&self) -> String {
        "governed-reduction-test".into()
    }

    fn device_id(&self) -> u32 {
        Self::DEVICE_ID
    }

    fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
        SpawnHandleConcurrency::ImmutableShare
    }

    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        ProviderCapabilitySnapshot {
            schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
            revision: 1,
            device: ApiDeviceInfo {
                device_id: Self::DEVICE_ID,
                name: self.device_info(),
                vendor: "test".into(),
                memory_bytes: Some(1 << 20),
                backend: None,
            },
            operations: vec![ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(GOVERNED_DENSE_REDUCTION_OPERATION),
                family: ProviderOperationFamily::Reduction,
            }],
            element_types: vec![runmat_accelerate_api::ProviderElementType::F64],
            max_rank: Some(2),
            max_allocation_bytes: Some(1 << 20),
            sparse: None,
            concurrency: ProviderConcurrencyCapabilities {
                spawn_handles: SpawnHandleConcurrency::ImmutableShare,
                concurrent_dispatch: true,
                cancellation: true,
                transactional_results: true,
            },
        }
    }

    fn query_feasibility(&self, _query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        if self.reject.load(Ordering::Relaxed) {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedOperation,
                "test.provider.rejected",
            );
        }
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: Some(1 << 10),
            output_bytes: Some(16),
            dispatches: Some(1),
        })
    }

    fn estimate_cost(&self, _query: &ProviderCostQuery) -> Option<ProviderCostEstimate> {
        Some(ProviderCostEstimate::static_prior(
            ExecutionCostComponents {
                execution_ns: 1,
                ..ExecutionCostComponents::default()
            },
            0,
        ))
    }

    fn execute_dense_reduction<'a>(
        &'a self,
        request: &'a DenseReductionRequest<'a>,
    ) -> AccelProviderFuture<'a, DenseReductionResult> {
        Box::pin(async move {
            request.validate(Self::DEVICE_ID)?;
            let input = self.numeric(request.input)?;
            let HostNumericDataOwned::F64(values) = input.data else {
                unreachable!("test provider stores only f64")
            };
            let is_minimum = matches!(request.operation, DenseReductionOperation::Minimum { .. });
            let (index, value) = values
                .iter()
                .copied()
                .enumerate()
                .reduce(|left, right| {
                    if (is_minimum && right.1 < left.1) || (!is_minimum && right.1 > left.1) {
                        right
                    } else {
                        left
                    }
                })
                .ok_or_else(|| anyhow::anyhow!("test reduction input is empty"))?;
            let output_shape = request.expected_output_shape();
            let result = DenseReductionResult {
                values: self.insert(vec![value], output_shape.clone()),
                indices: Some(self.insert(vec![(index + 1) as f64], output_shape.clone())),
                evidence: DenseReductionExecutionEvidence {
                    schema_version: DENSE_REDUCTION_EVIDENCE_SCHEMA_VERSION,
                    attempt_id: request.attempt_id,
                    input_identity: request.input_identity,
                    provider_device_id: Self::DEVICE_ID,
                    provider_capability_schema_version: request.provider_capability_schema_version,
                    provider_capability_revision: request.provider_capability_revision,
                    operation: request.operation,
                    input_bytes: checked_f64_bytes(&request.input.shape)?,
                    output_bytes: checked_f64_bytes(&output_shape)? * 2,
                    dispatches: 1,
                    host_upload_bytes: 0,
                    host_download_bytes: 0,
                    host_synchronizations: 0,
                    execution_ns: 1,
                    used_host_fallback: false,
                    cancellation_checks: 1,
                    cancellation_observed: false,
                },
            };
            Ok(result)
        })
    }
}
