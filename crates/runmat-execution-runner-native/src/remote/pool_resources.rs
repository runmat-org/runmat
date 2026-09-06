use runmat_execution::resource::{
    AcceleratorRequest, Capability, ResourceInventory, ResourceRequest as TaskResources,
};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) fn inventory(
    resources: &AllocationResources,
) -> NativeExecutionResult<ResourceInventory> {
    Ok(ResourceInventory {
        cpu_millicores: u32::try_from(resources.cpu_millicores)
            .map_err(|_| protocol("worker CPU request overflows scheduler units"))?,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes,
        accelerators: accelerators(resources)?,
        capabilities: native_worker_capabilities()?,
    })
}

pub(super) fn native_worker_capabilities(
) -> NativeExecutionResult<std::collections::BTreeSet<Capability>> {
    let mut capabilities = runmat_analysis_execution::analysis_worker_capabilities(
        std::iter::empty(),
        std::iter::empty(),
        false,
    );
    capabilities.insert(Capability::ProcessIsolation);
    if let Some(kernel_abi) = runmat_geometry_io::native_exact_kernel_abi() {
        let cohort = crate::meshing_platform::native_meshing_capability_cohort();
        capabilities.extend(
            runmat_meshing_execution::exact_meshing_worker_capability_inventory(
                &[kernel_abi],
                &runmat_meshing_core::production_algorithm_versions(),
                &[
                    runmat_meshing_core::ElementOrder::Tet4,
                    runmat_meshing_core::ElementOrder::Tet10,
                ],
                [cohort.as_str()],
            )
            .map_err(protocol)?,
        );
    }
    Ok(capabilities)
}

pub(super) fn pool_inventory(
    resources: &AllocationResources,
    workers: u32,
) -> NativeExecutionResult<ResourceInventory> {
    let mut inventory = inventory(resources)?;
    inventory.cpu_millicores = inventory
        .cpu_millicores
        .checked_mul(workers)
        .ok_or_else(|| protocol("worker pool CPU capacity overflows scheduler units"))?;
    inventory.memory_bytes = inventory
        .memory_bytes
        .checked_mul(u64::from(workers))
        .ok_or_else(|| protocol("worker pool memory capacity overflows scheduler units"))?;
    inventory.scratch_bytes = inventory
        .scratch_bytes
        .checked_mul(u64::from(workers))
        .ok_or_else(|| protocol("worker pool scratch capacity overflows scheduler units"))?;
    for accelerator in &mut inventory.accelerators {
        accelerator.count =
            accelerator
                .count
                .checked_mul(u16::try_from(workers).map_err(|_| {
                    protocol("worker pool size overflows accelerator scheduler units")
                })?)
                .ok_or_else(|| protocol("worker pool accelerators overflow scheduler units"))?;
    }
    Ok(inventory)
}

pub(super) fn task_resources(
    resources: &AllocationResources,
) -> NativeExecutionResult<TaskResources> {
    Ok(TaskResources {
        cpu_millicores: u32::try_from(resources.cpu_millicores)
            .map_err(|_| protocol("worker CPU request overflows scheduler units"))?,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes,
        max_wall_millis: resources.maximum_wall_millis,
        max_artifact_bytes: 64 * 1024 * 1024,
        max_egress_bytes: 64 * 1024 * 1024,
        max_relay_bytes: 4 * 1024 * 1024 * 1024,
        accelerators: accelerators(resources)?,
        required_capabilities: Default::default(),
    })
}

pub(super) fn task_deadline_unix_millis(
    resources: &AllocationResources,
) -> NativeExecutionResult<u64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| protocol("system clock predates the Unix epoch"))?;
    u64::try_from(now.as_millis())
        .map_err(|_| protocol("current time overflows scheduler milliseconds"))?
        .checked_add(resources.maximum_wall_millis)
        .ok_or_else(|| protocol("task deadline overflows scheduler milliseconds"))
}

fn accelerators(resources: &AllocationResources) -> NativeExecutionResult<Vec<AcceleratorRequest>> {
    match (
        resources.accelerator_count,
        resources.accelerator_class.as_ref(),
    ) {
        (0, _) => Ok(Vec::new()),
        (count, Some(class)) => Ok(vec![AcceleratorRequest {
            class: class.clone(),
            count: u16::try_from(count)
                .map_err(|_| protocol("worker accelerator count overflows scheduler units"))?,
            memory_bytes_each: resources.accelerator_memory_bytes / u64::from(count),
        }]),
        _ => Err(protocol("worker accelerator class is missing")),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use runmat_execution::resource::Capability;

    use super::native_worker_capabilities;

    #[test]
    fn native_inventory_always_advertises_analysis_and_process_isolation() {
        let inventory = native_worker_capabilities().expect("native worker inventory");
        let analysis = runmat_analysis_execution::analysis_worker_capabilities(
            std::iter::empty(),
            std::iter::empty(),
            false,
        );

        assert!(analysis.is_subset(&inventory));
        assert!(inventory.contains(&Capability::ProcessIsolation));
    }

    #[test]
    fn native_inventory_advertises_exact_meshing_only_with_occt() {
        let inventory = native_worker_capabilities().expect("native worker inventory");
        let has_meshing_capability = inventory.iter().any(|capability| {
            matches!(
                capability,
                Capability::Custom(identity) if identity.starts_with("runmat.meshing.")
            )
        });
        let has_exact_kernel_capability = inventory.iter().any(|capability| {
            matches!(
                capability,
                Capability::Custom(identity)
                    if identity.starts_with("runmat.meshing.exact-cad:")
            )
        });
        let has_admission_capability = inventory.contains(&Capability::Custom(
            runmat_meshing_execution::EXACT_CAD_ADMISSION_CAPABILITY.into(),
        ));
        let occt_compiled = cfg!(all(feature = "occt-native", not(target_arch = "wasm32")));

        assert_eq!(
            runmat_geometry_io::native_exact_kernel_abi().is_some(),
            occt_compiled
        );
        assert_eq!(has_meshing_capability, occt_compiled);
        assert_eq!(has_exact_kernel_capability, occt_compiled);
        assert_eq!(has_admission_capability, occt_compiled);
    }

    #[test]
    fn production_exact_workload_is_covered_by_native_inventory() {
        let Some(kernel_abi) = runmat_geometry_io::native_exact_kernel_abi() else {
            return;
        };
        let (mut document, _, _) = runmat_geometry_fixtures::exact_tetrahedron();
        let runmat_geometry_core::GeometryModel::ExactBRep { model } = &mut document.model else {
            panic!("exact tetrahedron fixture must remain an exact B-rep")
        };
        model.kernel_abi = kernel_abi;
        let request = runmat_meshing_core::resolve_meshing_request(
            document.tolerance,
            runmat_meshing_core::MeshingRequestSettings::default(),
        )
        .expect("production exact meshing request");
        let cohort = crate::meshing_platform::native_meshing_capability_cohort();
        let required = runmat_meshing_execution::exact_meshing_worker_capabilities(
            &document,
            &request,
            Some(&cohort),
        )
        .expect("production exact workload capabilities");
        let inventory = native_worker_capabilities().expect("native worker inventory");

        assert!(required.is_subset(&inventory));
    }
}
