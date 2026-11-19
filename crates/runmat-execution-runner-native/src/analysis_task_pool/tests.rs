use super::*;

#[test]
fn local_analysis_pool_uses_explicit_worker_budget_not_transport_defaults() {
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    let request = ResourceRequest {
        cpu_millicores: 2_000,
        memory_bytes: 2 * 1024 * 1024 * 1024,
        scratch_bytes: 256 * 1024 * 1024,
        max_wall_millis: 1_000,
        max_artifact_bytes: 0,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    };
    request.validate().unwrap();
    configure_local_analysis_envelope(&mut config, &request);
    assert_eq!(config.worker_cpu_millicores, request.cpu_millicores);
    assert_eq!(config.worker_memory_bytes, request.memory_bytes);
    assert_eq!(config.worker_scratch_bytes, request.scratch_bytes);
    assert!(config
        .worker_capabilities
        .contains(&Capability::ProcessIsolation));
    config.validate().unwrap();
    let mut no_scratch = request;
    no_scratch.scratch_bytes = 0;
    configure_local_analysis_envelope(&mut config, &no_scratch);
    assert_eq!(config.worker_scratch_bytes, 1);
    config.validate().unwrap();
}

#[test]
fn fallback_task_hints_do_not_resize_independently_configured_pool() {
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    let capacity = (
        config.worker_cpu_millicores,
        config.worker_memory_bytes,
        config.worker_scratch_bytes,
    );
    let hint = ResourceRequest {
        cpu_millicores: 1,
        memory_bytes: 1,
        scratch_bytes: 0,
        max_wall_millis: 1,
        max_artifact_bytes: 0,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::new(),
    };
    configure_local_analysis(&mut config, &hint);
    assert_eq!(
        capacity,
        (
            config.worker_cpu_millicores,
            config.worker_memory_bytes,
            config.worker_scratch_bytes,
        )
    );
    config.validate().unwrap();
}

/// Any pool operation would fabricate execution evidence for an empty wave.
struct UntouchedPool;

impl CacheImport for UntouchedPool {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        panic!("empty wave must not read objects")
    }
}

#[async_trait]
impl AnalysisTaskPool for UntouchedPool {
    async fn begin_wave_observation(&mut self, _: &[TaskId]) -> NativeExecutionResult<()> {
        panic!("empty wave must not begin an observation")
    }

    fn scope_id(&self) -> ExecutionScopeId {
        panic!("empty wave must not request a scope")
    }

    fn pool_id(&self) -> PoolId {
        panic!("empty wave must not request a pool")
    }

    fn register_program(&mut self, _: &PreparedAnalysisProgram) -> NativeExecutionResult<()> {
        panic!("empty wave must not register programs")
    }

    fn register_result_objects(
        &mut self,
        _: &runmat_analysis_execution::PreparedAnalysisObjects,
        _: &ArtifactAccess,
    ) -> NativeExecutionResult<()> {
        panic!("empty wave must not register results")
    }

    fn register_recovery_object(
        &mut self,
        _: &runmat_execution_artifact::ProgramExecutionObject,
    ) -> NativeExecutionResult<()> {
        panic!("empty wave must not register recovery objects")
    }

    async fn execute_batch(
        &mut self,
        _: AdmittedAnalysisBatch,
    ) -> NativeExecutionResult<Vec<AnalysisTaskCompletion>> {
        panic!("empty wave must not submit a graph")
    }
}

#[tokio::test]
async fn empty_analysis_wave_returns_no_execution_or_attempt_evidence() {
    let result = execute_analysis_task_batch(
        &mut UntouchedPool,
        Vec::new(),
        AnalysisTaskBatchPolicy {
            retry: RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
            deadline_unix_millis: Some(0),
            priority: 0,
        },
    )
    .await
    .unwrap();
    assert!(result.is_empty());
}

fn reserved_interval(wall_millis: u64) -> runmat_execution_runner::AttemptResourceAccounting {
    runmat_execution_runner::AttemptResourceAccounting::new(
        runmat_execution::identity::AttemptId::derive(&[b"replaced-driver-attempt"]),
        runmat_execution::identity::WorkerId::derive(&[b"replaced-driver-worker"]),
        runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            ResourceRequest {
                cpu_millicores: 1_000,
                memory_bytes: 2_048,
                scratch_bytes: 1_024,
                max_wall_millis: 10_000,
                max_artifact_bytes: 4_096,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: BTreeSet::new(),
            },
            wall_millis,
            0,
            0,
            0,
        )
        .unwrap(),
    )
    .unwrap()
}

/// Receipt a9866c3531: the replaced driver measured a committed wave at 1,978 ms. The
/// replacement driver restored that wave from the checkpoint and observed 1,828 ms for its
/// whole rerun of the workflow.
#[test]
fn workflow_wall_encloses_a_wave_measured_under_a_replaced_driver() {
    let restored_interval = reserved_interval(1_978);
    let rerun_observed = Duration::from_millis(1_828);

    let observed_only = workflow_wall_millis(rerun_observed, 0).unwrap();
    let error = runmat_execution_runner::AttemptResourceAggregate::new(
        vec![restored_interval.clone()],
        observed_only,
    )
    .unwrap_err();
    assert!(
        error.contains(
            "graph wall time is shorter than a measured interval: graph=1828 ms, interval=1978 ms"
        ),
        "{error}"
    );

    // The restored wave returned within 2 ms of local observation.
    let carried = carried_wave_millis([Some(1_978)], Duration::from_millis(2));
    assert_eq!(carried, 1_976);
    let wall = workflow_wall_millis(rerun_observed, carried).unwrap();
    assert_eq!(wall, 3_804);
    let aggregate =
        runmat_execution_runner::AttemptResourceAggregate::new(vec![restored_interval], wall)
            .unwrap();
    assert_eq!(aggregate.summary.usage.wall_millis, 3_804);
}

#[test]
fn only_wave_time_before_local_observation_is_carried() {
    // A wave measured within this process's observation began here and carries nothing.
    assert_eq!(
        carried_wave_millis([Some(40), None], Duration::from_millis(45)),
        0
    );
    assert_eq!(
        carried_wave_millis([None, None], Duration::from_millis(45)),
        0
    );
    assert_eq!(carried_wave_millis([], Duration::ZERO), 0);
    // A wave interrupted by the replacement keeps the durable clock it began with.
    assert_eq!(
        carried_wave_millis([Some(900), Some(2_400)], Duration::from_millis(600)),
        1_800
    );
    // A pool whose waves always begin in this process carries no time.
    assert_eq!(UntouchedPool.carried_wave_millis(), 0);
}
