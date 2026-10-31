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
