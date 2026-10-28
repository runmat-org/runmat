use super::*;
use crate::{
    algebra::SparsePatternStatistics,
    solve::general::{GeneralSolveEnvelope, GeneralSolvePolicy},
};
fn policy() -> GeneralSolvePolicy {
    GeneralSolvePolicy::bounded(1e-10)
}
fn general(policy: GeneralSolvePolicy, extra: u64) -> GeneralSparseSolveResourceDemand {
    let mut envelope = GeneralSolveEnvelope::for_counts(3, 9, policy).unwrap();
    envelope.resident_bytes += extra;
    GeneralSparseSolveResourceDemand::derive(
        SparsePatternStatistics {
            rows: 3,
            columns: 3,
            nonzeros: 9,
            maximum_row_nonzeros: 3,
        },
        policy,
        Count(7),
        envelope,
    )
    .unwrap()
}
// witness: FEA-SHARED-RESOURCES-001/post_projection_resource_admission @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Three-row general sparse count fixture verifies cold canonical resource plan,
//   additive resident storage and tolerance identity; estimates are requested storage, not measured
//   RSS or whole-task admission.
#[test]
fn governed_general_workspace_cold_identity_and_original_controls_are_rederived() {
    let mut planned = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut planned).solve =
        SolveResourcePolicy::GeneralSparse(general(policy(), 1_000_000));
    let first = build_resource_plan(planned.clone()).unwrap();
    let decoded = FeaResourcePlan::from_canonical_bytes(&first.canonical_bytes().unwrap()).unwrap();
    assert_eq!(first, decoded);
    let mut larger = planned.clone();
    operator_mut(&mut larger).solve =
        SolveResourcePolicy::GeneralSparse(general(policy(), 2_000_000));
    let second = build_resource_plan(larger).unwrap();
    assert_eq!(
        second.request.memory_bytes,
        first.request.memory_bytes + 1_000_000
    );
    assert_ne!(second.identity, first.identity);
    let mut changed = policy();
    changed.relative_tolerance *= 0.5;
    let mut altered = planned;
    operator_mut(&mut altered).solve =
        SolveResourcePolicy::GeneralSparse(general(changed, 1_000_000));
    let altered = build_resource_plan(altered).unwrap();
    assert_eq!(altered.request.memory_bytes, first.request.memory_bytes);
    assert_ne!(altered.identity, first.identity);
}
#[test]
fn general_demand_rejects_understated_proof_work_and_unsupported_partitioning() {
    let mut planned = input(PartitioningPolicy::Monolithic);
    let d = general(policy(), 0);
    let mut encoded = serde_json::to_value(d).unwrap();
    let work = encoded["admission"]["scalar_operations"].as_u64().unwrap();
    encoded["admission"]["scalar_operations"] = serde_json::json!(work - 1);
    let short = serde_json::from_value(encoded).unwrap();
    operator_mut(&mut planned).solve = SolveResourcePolicy::GeneralSparse(short);
    assert!(build_resource_plan(planned).is_err());
    let mut planned = input(PartitioningPolicy::OperatorDomain {
        maximum_partitions: 2,
        replicated_memory_bytes_per_partition: ByteCount(0),
        halo_memory_bytes_per_partition: ByteCount(0),
        replicated_scratch_bytes_per_partition: ByteCount(0),
        halo_scratch_bytes_per_partition: ByteCount(0),
    });
    operator_mut(&mut planned).solve = SolveResourcePolicy::GeneralSparse(d);
    assert!(build_resource_plan(planned).is_err());
}

// witness: FEA-SHARED-RESOURCES-001/post_projection_resource_admission @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: General-solve demand rejects the original 100-byte accuracy ceiling; no physical
//   operator or remote admission evidence.
#[test]
fn governed_general_demand_obeys_the_original_resource_ceiling() {
    let mut planned = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut planned).solve =
        SolveResourcePolicy::GeneralSparse(general(policy(), 1_000_000));
    planned.accuracy_ceiling.maximum_resident_bytes = Some(100);
    assert!(matches!(
        build_resource_plan(planned),
        Err(ResourcePlanningError::AccuracyCeiling {
            resource: "resident memory bytes",
            ..
        })
    ));
}

// witness: FEA-SHARED-ACCELERATORS-001/provider_owned_native_sparse_work @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Host general-solver demand plus a synthetic GPU requirement rejects planning; no
//   actual accelerator kernel capability or performance proof.
#[test]
fn general_host_solver_demand_does_not_authorize_accelerator_execution() {
    let mut planned = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut planned).solve = SolveResourcePolicy::GeneralSparse(general(policy(), 0));
    planned.worker.accelerators.push(AcceleratorRequirement {
        class: runmat_execution::resource::AcceleratorClass::new("test-gpu").unwrap(),
        count: 1,
        memory_bytes_each: ByteCount(1024),
    });
    assert!(matches!(
        build_resource_plan(planned),
        Err(ResourcePlanningError::Invalid(_))
    ));
}
