//! Reconcile physical observation claims with their exact execution reservations.
use crate::{AnalysisBridgeError, AnalysisBridgeResult};
use runmat_analysis_fea::{
    planning::resource::{FeaResourcePhase, FeaResourcePlanIdentity, ResourceUsageEvidence},
    results::observation::{ObservationPublicationCatalog, ObservationPublicationEntry},
};
use runmat_execution::resource::AcceleratorRequest;
use std::collections::{BTreeMap, BTreeSet};

/// Requires source-derived partition/reduction/checkpoint/finalization/assembly accounting,
/// then reconciles governed acceleration with its exact admitted allocations.
pub fn validate_observation_resource_usage(
    observations: &ObservationPublicationCatalog,
    usage: &ResourceUsageEvidence,
) -> AnalysisBridgeResult<()> {
    runmat_analysis_fea::results::publication::validate_observation_publication_usage(
        observations,
        usage,
    )?;
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
    // Reconcile the entire publication once: individual samples cannot each account for
    // reservations belonging to other samples in the same admitted task graph.
    for summary in observations
        .entries()
        .iter()
        .filter_map(|entry| match entry {
            ObservationPublicationEntry::Sample(sample) => Some(&sample.manifest),
            ObservationPublicationEntry::History(_) => None,
        })
        .flat_map(|manifest| &manifest.results)
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
    if observations
        .entries()
        .iter()
        .any(|entry| matches!(entry, ObservationPublicationEntry::History(_)))
        && accelerated_plans
            .keys()
            .any(|identity| !claims.contains_key(identity))
    {
        return Err(AnalysisBridgeError::Invalid(
            "history publication has no acceleration summary for its admitted observation phases"
                .into(),
        ));
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

    fn allocation(class: &str, minimum_allocation_bytes: u64) -> AcceleratorRequest {
        AcceleratorRequest {
            class: runmat_execution::resource::AcceleratorClass::new(class).unwrap(),
            count: 1,
            minimum_allocation_bytes,
            provider: None,
            required_features: BTreeSet::new(),
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
            .minimum_allocation_bytes += 1;
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
        // Accelerator requests are accounted against the leased device that satisfies them.
        let device = crate::test_fixtures::accelerator_device("exact-f64", 8 << 30);
        let assignment = runmat_execution::resource::ResourceAssignment {
            accelerator_leases: vec![
                runmat_execution::resource::AcceleratorDeviceLease::for_attempt(
                    runmat_execution::identity::AttemptId::derive(&[b"exact-accelerator-attempt"]),
                    1,
                    &device,
                )
                .unwrap(),
            ],
        };
        let exact =
            ResourceAccounting::for_assigned_interval(request.clone(), &assignment, 1, 0, 0, 0)
                .unwrap();
        validate_exact_accelerator_allocation(std::slice::from_ref(&planned), &exact).unwrap();

        request.accelerators[0].minimum_allocation_bytes -= 1;
        let underallocated =
            ResourceAccounting::for_assigned_interval(request, &assignment, 1, 0, 0, 0).unwrap();
        assert!(validate_exact_accelerator_allocation(
            std::slice::from_ref(&planned),
            &underallocated,
        )
        .is_err());
    }
}
