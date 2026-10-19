use super::*;

#[test]
fn modal_cpu_grant_binds_identity_pool_storage_and_unchanged_accuracy_ceiling() {
    let (workload, mesh, constraints, observations) = modal_case(1, ComputeBackend::Cpu);
    let plan = |cpu| {
        let mut facts = prepared(&mesh, &constraints, &workload);
        facts.local_execution = super::super::LocalExecutionPlan::from_cpu_millicores(cpu).unwrap();
        build_structural_modal_resource_plan(
            Some(&mesh),
            &constraints,
            &observations,
            &workload,
            None,
            &[],
            facts,
        )
        .unwrap()
    };
    let serial = plan(1_000);
    let parallel = plan(4_000);
    assert_eq!(parallel.request.cpu_millicores, 4_000);
    assert_eq!(serial.request.cpu_millicores, 1_000);
    assert_ne!(serial.identity, parallel.identity);
    assert!(parallel.request.memory_bytes > serial.request.memory_bytes);
    assert_eq!(
        parallel.input.accuracy_ceiling,
        serial.input.accuracy_ceiling
    );
    parallel.validate_identity().unwrap();
}
