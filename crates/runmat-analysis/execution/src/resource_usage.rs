use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::planning::resource::{
    record_resource_graph_usage, record_resource_usage, FeaResourcePhase, FeaResourcePlan,
    FeaResourcePlanIdentity, ResourceUsageAttempt, ResourceUsageEvidence,
};
use runmat_analysis_fea::results::observation::ObservationArtifactManifest;
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_execution_runner::{
    AttemptResourceAccounting, AttemptResourceAggregate, AttemptSuccess,
};

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    ImportedStructuralLinearResult, PreparedAnalysisObjects, PreparedStructuralLinearProgram,
};

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedStructuralResourceUsage {
    pub evidence: ResourceUsageEvidence,
    pub objects: PreparedAnalysisObjects,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedResourceUsageAttempt {
    /// The analysis-domain plan admitted for this exact scheduler attempt.
    pub plan: FeaResourcePlan,
    pub accounting: AttemptResourceAccounting,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedResourceUsageGraph {
    pub attempts: Vec<PlannedResourceUsageAttempt>,
    /// Elapsed wall time of this subgraph, independent of summed reserved task time.
    pub wall_millis: u64,
}

impl PlannedResourceUsageGraph {
    pub fn new(
        attempts: Vec<PlannedResourceUsageAttempt>,
        wall_millis: u64,
    ) -> AnalysisBridgeResult<Self> {
        for attempt in &attempts {
            attempt
                .plan
                .validate_identity()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        }
        AttemptResourceAggregate::new(
            attempts
                .iter()
                .map(|attempt| attempt.accounting.clone())
                .collect(),
            wall_millis,
        )
        .map_err(|error| {
            AnalysisBridgeError::Invalid(format!("invalid execution-graph accounting: {error}"))
        })?;
        Ok(Self {
            attempts,
            wall_millis,
        })
    }
}

/// Adapts scheduler-owned accounting into source-bound FEA evidence without giving FEA a second
/// worker-metering implementation.
pub fn prepare_structural_resource_usage(
    prepared: &PreparedStructuralLinearProgram,
    result: &ImportedStructuralLinearResult,
    success: &AttemptSuccess,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedStructuralResourceUsage> {
    success.validate_accounting().map_err(|error| {
        AnalysisBridgeError::Invalid(format!("invalid scheduler resource accounting: {error}"))
    })?;
    let accounting = success.resources.as_ref().ok_or_else(|| {
        AnalysisBridgeError::Invalid(
            "structural result has no scheduler-bound resource accounting".into(),
        )
    })?;
    if accounting.resources.artifact_bytes != result.result_object_bytes {
        return Err(AnalysisBridgeError::Identity(
            "resource accounting artifact bytes differ from the imported structural result",
        ));
    }
    let evidence = record_resource_usage(
        &prepared.resource_plan,
        prepared.host.workload.identity.digest(),
        result.identity.digest(),
        vec![usage_attempt(prepared.resource_plan.identity, accounting)],
        accounting.resources.usage.wall_millis,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    prepare_usage_objects(prepared, evidence, chunk_bytes, limits)
}

/// Extends solve accounting with a measured downstream execution graph while retaining one
/// canonical FEA evidence object. Generic execution owns interval measurement and aggregation;
/// this bridge only binds those records to the scientific workload and result identities.
pub fn extend_structural_resource_usage(
    prepared: &PreparedStructuralLinearProgram,
    result: &ImportedStructuralLinearResult,
    baseline: &PreparedStructuralResourceUsage,
    additional: &PlannedResourceUsageGraph,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedStructuralResourceUsage> {
    validate_structural_resource_usage(prepared, result, baseline, &prepared.host.artifact_access)?;
    let mut attempts = baseline.evidence.attempts.clone();
    attempts.extend(
        additional
            .attempts
            .iter()
            .map(|attempt| usage_attempt(attempt.plan.identity, &attempt.accounting)),
    );
    let mut plans = BTreeMap::new();
    for attempt in &additional.attempts {
        if let Some(existing) = plans.insert(attempt.plan.identity, attempt.plan.clone()) {
            if existing != attempt.plan {
                return Err(AnalysisBridgeError::Identity(
                    "one resource-plan identity names distinct phase plans",
                ));
            }
        }
    }
    let graph_wall_millis = baseline
        .evidence
        .actual
        .scheduler
        .wall_millis
        .checked_add(additional.wall_millis)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("resource graph wall time overflowed".into())
        })?;
    let evidence = record_resource_graph_usage(
        &prepared.resource_plan,
        plans.into_values().collect(),
        prepared.host.workload.identity.digest(),
        result.identity.digest(),
        attempts,
        graph_wall_millis,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    prepare_usage_objects(prepared, evidence, chunk_bytes, limits)
}

fn usage_attempt(
    plan: runmat_analysis_fea::planning::resource::FeaResourcePlanIdentity,
    accounting: &AttemptResourceAccounting,
) -> ResourceUsageAttempt {
    ResourceUsageAttempt {
        attempt_id: accounting.attempt_id,
        worker_id: accounting.worker_id,
        plan,
        resources: accounting.resources.clone(),
    }
}

fn prepare_usage_objects(
    prepared: &PreparedStructuralLinearProgram,
    evidence: ResourceUsageEvidence,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedStructuralResourceUsage> {
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourceUsage,
        evidence.identity().digest(),
        &evidence
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        chunk_bytes,
        ObjectNamespace::ResultValue,
        limits,
    )?;
    // Keep the value role explicit at the boundary used by publication preparation.
    objects.references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    Ok(PreparedStructuralResourceUsage { evidence, objects })
}

pub fn validate_structural_resource_usage(
    prepared: &PreparedStructuralLinearProgram,
    result: &ImportedStructuralLinearResult,
    usage: &PreparedStructuralResourceUsage,
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<()> {
    let decoded = ResourceUsageEvidence::from_canonical_bytes(&usage.objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    usage
        .objects
        .references(access, ValueRefKind::ResultObject)?;
    if decoded != usage.evidence
        || decoded.identity().digest() != usage.objects.manifest.logical_identity
        || decoded.primary_plan != prepared.resource_plan.identity
        || decoded.workload_digest != prepared.host.workload.identity.digest()
        || decoded.result_digest != result.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "resource usage evidence differs from its structural sources",
        ));
    }
    Ok(())
}

/// Reconciles governed observation acceleration only against attempts admitted for accelerated
/// observation-partition plans. Solve or unrelated phase reservations cannot satisfy this proof.
pub fn validate_observation_resource_usage(
    observations: &ObservationArtifactManifest,
    usage: &ResourceUsageEvidence,
) -> AnalysisBridgeResult<()> {
    observations
        .validate()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    usage
        .validate_identity()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let accelerated_plans = usage
        .plans
        .iter()
        .filter(|plan| {
            matches!(
                plan.input.phase,
                FeaResourcePhase::StructuralObservationPartition(_)
            ) && !plan.request.accelerators.is_empty()
        })
        .map(|plan| (plan.identity, plan))
        .collect::<BTreeMap<_, _>>();

    let mut claims = BTreeMap::new();
    for summary in observations
        .results
        .iter()
        .filter_map(|result| result.acceleration.as_ref())
    {
        for operation in &summary.operations {
            if claims
                .insert(
                    operation.resource_plan,
                    ObservationAccelerationClaim {
                        allocation: summary.allocation.clone(),
                        execution_ns: operation.execution_ns,
                    },
                )
                .is_some()
            {
                return Err(AnalysisBridgeError::Identity(
                    "observation acceleration reuses a phase resource plan",
                ));
            }
        }
    }
    let mut reservations = BTreeMap::new();
    for (identity, plan) in accelerated_plans {
        let [planned_allocation] = plan.request.accelerators.as_slice() else {
            return Err(AnalysisBridgeError::Identity(
                "accelerated observation phase lacks one exact allocation",
            ));
        };
        let attempts = usage
            .attempts
            .iter()
            .filter(|attempt| attempt.plan == identity)
            .collect::<Vec<_>>();
        if attempts.is_empty() {
            return Err(AnalysisBridgeError::Identity(
                "accelerated observation phase has no scheduler attempt",
            ));
        }
        let device_millis = attempts.iter().try_fold(0_u64, |total, attempt| {
            validate_exact_accelerator_allocation(
                std::slice::from_ref(planned_allocation),
                &attempt.resources,
            )?;
            let matching = attempt
                .resources
                .usage
                .accelerator_millis
                .iter()
                .filter(|measured| measured.class == planned_allocation.class)
                .collect::<Vec<_>>();
            let [matching] = matching.as_slice() else {
                return Err(AnalysisBridgeError::Identity(
                    "accelerated observation attempt lacks one exact usage record",
                ));
            };
            total.checked_add(matching.device_millis).ok_or_else(|| {
                AnalysisBridgeError::Invalid("observation accelerator accounting overflowed".into())
            })
        })?;
        reservations.insert(
            identity,
            ObservationAccelerationReservation {
                allocation: planned_allocation.clone(),
                device_millis,
            },
        );
    }
    reconcile_observation_acceleration_claims(&claims, &reservations)
}

fn validate_exact_accelerator_allocation(
    planned: &[AcceleratorRequest],
    actual: &runmat_execution::resource::ResourceAccounting,
) -> AnalysisBridgeResult<()> {
    if actual.allocation.accelerators != planned {
        return Err(AnalysisBridgeError::Identity(
            "accelerated observation attempt allocation differs from its exact phase plan",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservationAccelerationClaim {
    allocation: AcceleratorRequest,
    execution_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservationAccelerationReservation {
    allocation: AcceleratorRequest,
    device_millis: u64,
}

fn reconcile_observation_acceleration_claims(
    claims: &BTreeMap<FeaResourcePlanIdentity, ObservationAccelerationClaim>,
    reservations: &BTreeMap<FeaResourcePlanIdentity, ObservationAccelerationReservation>,
) -> AnalysisBridgeResult<()> {
    if claims.keys().copied().collect::<BTreeSet<_>>()
        != reservations.keys().copied().collect::<BTreeSet<_>>()
    {
        return Err(AnalysisBridgeError::Identity(
            "observation acceleration does not exactly cover accelerated partition plans",
        ));
    }
    for (identity, claim) in claims {
        let reservation = &reservations[identity];
        if claim.allocation != reservation.allocation {
            return Err(AnalysisBridgeError::Identity(
                "observation acceleration allocation differs from its exact phase plan",
            ));
        }
        let reserved_ns = reservation
            .device_millis
            .checked_mul(1_000_000)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid("observation accelerator time overflowed".into())
            })?;
        if claim.execution_ns > reserved_ns {
            return Err(AnalysisBridgeError::Invalid(
                "observation accelerator execution exceeds exact scheduler accounting".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod acceleration_resource_tests {
    use super::*;
    use runmat_execution::{
        resource::{ResourceAccounting, ResourceRequest},
        Digest,
    };

    fn identity(label: &str) -> FeaResourcePlanIdentity {
        serde_json::from_value(serde_json::to_value(Digest::sha256(label)).unwrap()).unwrap()
    }

    fn allocation(class: &str, memory_bytes_each: u64) -> AcceleratorRequest {
        AcceleratorRequest {
            class: class.into(),
            count: 1,
            memory_bytes_each,
        }
    }

    #[test]
    fn acceleration_reconciliation_is_exact_per_phase_plan() {
        let first = identity("first-observation-plan");
        let second = identity("second-observation-plan");
        let first_allocation = allocation("gpu-a", 1 << 20);
        let second_allocation = allocation("gpu-b", 2 << 20);
        let claims = BTreeMap::from([
            (
                first,
                ObservationAccelerationClaim {
                    allocation: first_allocation.clone(),
                    execution_ns: 900_000,
                },
            ),
            (
                second,
                ObservationAccelerationClaim {
                    allocation: second_allocation.clone(),
                    execution_ns: 1_900_000,
                },
            ),
        ]);
        let reservations = BTreeMap::from([
            (
                first,
                ObservationAccelerationReservation {
                    allocation: first_allocation,
                    device_millis: 1,
                },
            ),
            (
                second,
                ObservationAccelerationReservation {
                    allocation: second_allocation,
                    device_millis: 2,
                },
            ),
        ]);
        reconcile_observation_acceleration_claims(&claims, &reservations).unwrap();

        let mut missing = reservations.clone();
        missing.remove(&second);
        assert!(reconcile_observation_acceleration_claims(&claims, &missing).is_err());

        let mut substituted = reservations.clone();
        substituted
            .get_mut(&first)
            .unwrap()
            .allocation
            .memory_bytes_each += 1;
        assert!(reconcile_observation_acceleration_claims(&claims, &substituted).is_err());

        let mut swapped = reservations.clone();
        let first_reservation = swapped[&first].clone();
        let second_reservation = swapped[&second].clone();
        swapped.insert(first, second_reservation);
        swapped.insert(second, first_reservation);
        assert!(reconcile_observation_acceleration_claims(&claims, &swapped).is_err());

        let mut insufficient = reservations;
        insufficient.get_mut(&second).unwrap().device_millis = 1;
        assert!(reconcile_observation_acceleration_claims(&claims, &insufficient).is_err());
    }

    #[test]
    fn scheduler_accounting_must_retain_the_exact_accelerator_reservation() {
        let planned = allocation("exact-f64", 8 << 30);
        let mut request = ResourceRequest {
            cpu_millicores: 1_000,
            memory_bytes: 1 << 30,
            scratch_bytes: 0,
            max_wall_millis: 100,
            max_artifact_bytes: 0,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: vec![planned.clone()],
            required_capabilities: BTreeSet::new(),
        };
        let exact = ResourceAccounting::for_reserved_interval(request.clone(), 1, 0, 0, 0).unwrap();
        validate_exact_accelerator_allocation(std::slice::from_ref(&planned), &exact).unwrap();

        request.accelerators[0].memory_bytes_each -= 1;
        let underallocated =
            ResourceAccounting::for_reserved_interval(request, 1, 0, 0, 0).unwrap();
        assert!(validate_exact_accelerator_allocation(
            std::slice::from_ref(&planned),
            &underallocated,
        )
        .is_err());
    }
}
