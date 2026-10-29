use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use crate::planning::resource::FeaResourcePlanIdentity;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, ApiDeviceInfo, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostNumericTensorView, HostTensorView, ProviderCapabilitySnapshot,
    ProviderConcurrencyCapabilities, ProviderExecutionFuture, ProviderExecutionResult,
    ReduceDimResult, SpawnHandleConcurrency, PROVIDER_CAPABILITY_SCHEMA_VERSION,
};
use runmat_analysis_core::{
    ExtremaTiePolicy, ObservationId, ObservationRequest, ObservationTarget, ReductionPolicy,
    SeriesReduction, SpatialReduction,
};
use runmat_execution::{
    resource::{AcceleratorClass, AcceleratorRequest},
    Digest,
};

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
        class: AcceleratorClass::new("test-gpu").unwrap(),
        count: 1,
        minimum_allocation_bytes: 1024 * 1024,
        provider: None,
        required_features: Default::default(),
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
    exact_allocation.minimum_allocation_bytes = required_device_bytes;
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
    insufficient_allocation.minimum_allocation_bytes -= 1;
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
    assert_eq!(insufficient_provider.allocation_count(), 0);
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
    substituted.class = AcceleratorClass::new("substituted-gpu").unwrap();
    assert!(evidence.validate_for_allocation(&substituted).is_err());
    let mut undersized = allocation();
    undersized.minimum_allocation_bytes = 1;
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

// witness: FEA-SHARED-OBSERVATIONS-001/governed_observation_corpus @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Governed extrema reject unsupported work before provider allocation.
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
        assert_eq!(provider.allocation_count(), 0);
        assert_eq!(provider.live_handle_count(), 0);
    }
}

/// Exact-f64 reductions over real fixture allocations. `reject` withholds
/// transactional results, so the governed path must decline before uploading.
#[derive(Default)]
struct GovernedReductionTestProvider {
    owner: TestAllocationOwner,
    reject: AtomicBool,
    probes: Mutex<Vec<RetirementProbe>>,
}

impl GovernedReductionTestProvider {
    fn allocate(&self, values: Vec<f64>, shape: Vec<usize>) -> anyhow::Result<GpuTensorHandle> {
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(values),
            shape,
            storage: GpuTensorStorage::Real,
        })?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }

    fn values(&self, handle: &GpuTensorHandle) -> anyhow::Result<Vec<f64>> {
        for probe in self.probes.lock().unwrap().iter() {
            if let Ok(values) = probe.with_data(handle, |data| match data {
                HostNumericDataView::F64(values) => Some(values.to_vec()),
                _ => None,
            }) {
                return values.ok_or_else(|| anyhow::anyhow!("test provider stores only f64"));
            }
        }
        anyhow::bail!("test provider does not own this handle")
    }

    fn allocation_count(&self) -> usize {
        self.probes.lock().unwrap().len()
    }

    fn live_handle_count(&self) -> usize {
        self.probes
            .lock()
            .unwrap()
            .iter()
            .filter(|probe| probe.retirements() == 0)
            .count()
    }

    fn reduce<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        minimum: bool,
    ) -> ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            let result: anyhow::Result<ReduceDimResult> = (|| {
                anyhow::ensure!(dim == 1 && input.shape()[0] == 1, "test reduces one row");
                let (index, value) = self
                    .values(input)?
                    .into_iter()
                    .enumerate()
                    .reduce(|left, right| {
                        if (minimum && right.1 < left.1) || (!minimum && right.1 > left.1) {
                            right
                        } else {
                            left
                        }
                    })
                    .ok_or_else(|| anyhow::anyhow!("test reduction input is empty"))?;
                Ok(ReduceDimResult {
                    values: self.allocate(vec![value], vec![1, 1])?,
                    indices: self.allocate(vec![(index + 1) as f64], vec![1, 1])?,
                })
            })();
            result.map_err(Into::into)
        })
    }
}

impl AccelProvider for GovernedReductionTestProvider {
    fn upload(&self, _host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("governed extrema upload through upload_numeric")
    }

    fn download<'a>(&'a self, _handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("governed extrema download through download_numeric") })
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        host.validate()?;
        let HostNumericDataView::F64(values) = host.data else {
            return Err(anyhow::anyhow!("test provider accepts only f64").into());
        };
        if host.storage != GpuTensorStorage::Real {
            return Err(anyhow::anyhow!("real storage required").into());
        }
        Ok(self.allocate(values.to_vec(), host.shape.to_vec())?)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(self.values(handle)?),
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }

    fn reduce_min_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
    ) -> ProviderExecutionFuture<'a, ReduceDimResult> {
        self.reduce(input, dim, true)
    }

    fn reduce_max_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
    ) -> ProviderExecutionFuture<'a, ReduceDimResult> {
        self.reduce(input, dim, false)
    }

    fn device_info(&self) -> String {
        "governed-reduction-test".into()
    }

    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
        SpawnHandleConcurrency::ImmutableShare
    }

    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        ProviderCapabilitySnapshot {
            schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
            revision: 1,
            device: ApiDeviceInfo {
                device_id: self.device_id(),
                name: self.device_info(),
                vendor: "test".into(),
                memory_bytes: Some(1 << 20),
                backend: None,
            },
            operations: Vec::new(),
            element_types: vec![runmat_accelerate_api::ProviderElementType::F64],
            max_rank: Some(2),
            max_allocation_bytes: Some(1 << 20),
            sparse: None,
            concurrency: ProviderConcurrencyCapabilities {
                spawn_handles: SpawnHandleConcurrency::ImmutableShare,
                concurrent_dispatch: true,
                cancellation: true,
                transactional_results: !self.reject.load(Ordering::Relaxed),
            },
        }
    }
}
