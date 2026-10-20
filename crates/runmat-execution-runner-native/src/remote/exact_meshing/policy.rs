//! Remote work is bounded by its allocation, independently of the driver host.
use super::{AllocationResources, MeshingExecutionPolicy, NativeExecutionResult};

pub(super) fn execution_policy(
    resources: &AllocationResources,
) -> NativeExecutionResult<MeshingExecutionPolicy> {
    let policy = MeshingExecutionPolicy {
        cpu_millicores: super::super::pool_resources::task_resources(resources)?.cpu_millicores,
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
}
