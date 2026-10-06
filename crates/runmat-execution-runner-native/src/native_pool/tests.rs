use super::*;
use runmat_execution_runner::DriverConfig;

#[test]
fn independent_inventory_is_exact_and_registration_does_not_infer_task_capacity() {
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    config.max_workers = 3;
    config.worker_cpu_millicores = 500;
    config.worker_memory_bytes = 19;
    config.worker_scratch_bytes = 23;
    let scope_id = ExecutionScopeId::derive(&[b"native-inventory"]);
    let mut driver = Driver::new(DriverConfig::default(), 1).unwrap();
    let pool_id = register(&mut driver, &config, scope_id).unwrap();
    let snapshot = driver.snapshot();
    snapshot.validate().unwrap();
    assert!(snapshot.tasks.is_empty());
    let pool = &snapshot.pools[&pool_id];
    assert_eq!(pool.spec.resource_limit.cpu_millicores, 1500);
    assert_eq!(pool.spec.resource_limit.memory_bytes, 57);
    assert_eq!(pool.spec.resource_limit.scratch_bytes, 69);
    assert_eq!(pool.workers.len(), 3);
    for worker in pool.workers.values() {
        assert_eq!(worker.spec.resources.memory_bytes, 19);
        assert_eq!(worker.spec.resources.scratch_bytes, 23);
    }
}

#[test]
fn overflowing_native_inventory_is_rejected_before_any_registration() {
    for dimension in 0..3 {
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.max_workers = 2;
        match dimension {
            0 => config.worker_cpu_millicores = u32::MAX,
            1 => config.worker_memory_bytes = u64::MAX,
            _ => config.worker_scratch_bytes = u64::MAX,
        }
        let mut driver = Driver::new(DriverConfig::default(), 1).unwrap();
        let before = driver.snapshot();
        assert!(matches!(
            register(
                &mut driver,
                &config,
                ExecutionScopeId::derive(&[b"overflow"])
            ),
            Err(NativeExecutionError::Configuration(_))
        ));
        assert_eq!(driver.snapshot(), before);
    }
}
