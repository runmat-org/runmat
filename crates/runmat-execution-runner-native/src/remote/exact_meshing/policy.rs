//! Remote work is bounded by its allocation, independently of the driver host.
use super::{AllocationResources, MeshingExecutionPolicy, NativeExecutionResult};
use runmat_meshing_core::{MeshingExecutionCeilings, MeshingSizeLimits};

/// The allocation is the worker's capacity authority. Coordinator host discovery and the
/// portable fixture floor must never enlarge it. Mesh size still owns search/iteration work.
pub(super) fn execution_ceilings(
    resources: &AllocationResources,
    limits: &MeshingSizeLimits,
) -> NativeExecutionResult<MeshingExecutionCeilings> {
    let mut ceilings = MeshingExecutionCeilings::for_host(resources.memory_bytes, limits);
    ceilings.maximum_memory_bytes = resources.memory_bytes;
    ceilings.maximum_scratch_bytes = resources.scratch_bytes;
    ceilings.maximum_artifact_bytes = resources.memory_bytes;
    ceilings.maximum_wall_time_ms = ceilings
        .maximum_wall_time_ms
        .min(resources.maximum_wall_millis);
    ceilings.validate().map_err(super::protocol)?;
    Ok(ceilings)
}

pub(super) fn execution_policy(
    resources: &AllocationResources,
) -> NativeExecutionResult<MeshingExecutionPolicy> {
    let task_resources = super::super::pool_resources::task_resources(resources)?;
    let policy = MeshingExecutionPolicy {
        cpu_millicores: task_resources.cpu_millicores,
        maximum_egress_bytes: task_resources.max_egress_bytes,
        maximum_relay_bytes: task_resources.max_relay_bytes,
        deadline_unix_millis: Some(super::super::pool_resources::task_deadline_unix_millis(
            resources,
        )?),
        ..MeshingExecutionPolicy::default()
    };
    policy.validate().map_err(super::protocol)?;
    crate::meshing_host::admitted_cpu_threads(policy.cpu_millicores, resources.memory_bytes)?;
    Ok(policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocation(cpu_millicores: u64) -> AllocationResources {
        AllocationResources {
            cpu_millicores,
            memory_bytes: 1 << 30,
            scratch_bytes: 1 << 30,
            accelerators: Vec::new(),
            maximum_wall_millis: 180_000,
        }
    }

    fn limits() -> MeshingSizeLimits {
        MeshingSizeLimits {
            maximum_nodes: 100_000,
            maximum_elements: 1_000_000,
            maximum_recursion_depth: 64,
        }
    }

    #[test]
    fn remote_meshing_policy_uses_allocated_cpu_and_deadline() {
        for cpu in [1_000, 8_000] {
            let resources = allocation(cpu);
            let before =
                super::super::super::pool_resources::task_deadline_unix_millis(&resources).unwrap();
            let policy = execution_policy(&resources).unwrap();
            let after =
                super::super::super::pool_resources::task_deadline_unix_millis(&resources).unwrap();
            assert_eq!(u64::from(policy.cpu_millicores), cpu);
            let task_resources =
                super::super::super::pool_resources::task_resources(&resources).unwrap();
            assert_eq!(policy.maximum_egress_bytes, task_resources.max_egress_bytes);
            assert_eq!(policy.maximum_relay_bytes, task_resources.max_relay_bytes);
            assert!((before..=after).contains(&policy.deadline_unix_millis.unwrap()));
        }
    }

    #[test]
    fn remote_meshing_policy_rejects_zero_and_overflowing_cpu() {
        for cpu in [
            0,
            500,
            1_500,
            (runmat_cpu_execution::MAX_THREADS as u64 + 1) * 1_000,
            u64::from(u32::MAX) + 1,
        ] {
            assert!(execution_policy(&allocation(cpu)).is_err());
        }
    }

    #[test]
    fn remote_meshing_ceilings_preserve_each_admitted_capacity_without_a_portable_floor() {
        for (memory, scratch, wall) in [
            (1 << 30, 2 << 30, 180_000),
            (64 << 20, 16 << 20, 5_000),
            (16 << 30, 2 << 30, 7_200_000),
        ] {
            let mut resources = allocation(1_000);
            resources.memory_bytes = memory;
            resources.scratch_bytes = scratch;
            resources.maximum_wall_millis = wall;
            let limits = limits();
            let ceilings = execution_ceilings(&resources, &limits).unwrap();
            let requested_work = MeshingExecutionCeilings::for_host(memory, &limits);
            assert_eq!(ceilings.maximum_memory_bytes, memory);
            assert_eq!(ceilings.maximum_scratch_bytes, scratch);
            assert_eq!(ceilings.maximum_artifact_bytes, memory);
            assert_eq!(
                ceilings.maximum_wall_time_ms,
                wall.min(requested_work.maximum_wall_time_ms)
            );
            assert_eq!(
                ceilings.maximum_search_work,
                requested_work.maximum_search_work
            );
            assert_eq!(
                ceilings.maximum_iterations,
                requested_work.maximum_iterations
            );
        }
    }

    #[test]
    fn remote_meshing_ceilings_reject_missing_capacity_instead_of_inventing_it() {
        for absent in 0..3 {
            let mut resources = allocation(1_000);
            match absent {
                0 => resources.memory_bytes = 0,
                1 => resources.scratch_bytes = 0,
                _ => resources.maximum_wall_millis = 0,
            }
            assert!(execution_ceilings(&resources, &limits()).is_err());
        }
    }
}
