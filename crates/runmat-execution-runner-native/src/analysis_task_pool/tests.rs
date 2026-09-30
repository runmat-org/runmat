use super::*;

/// Any pool operation would fabricate execution evidence for an empty wave.
struct UntouchedPool;

impl CacheImport for UntouchedPool {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        panic!("empty wave must not read objects")
    }
}

#[async_trait]
impl AnalysisTaskPool for UntouchedPool {
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
        _: Vec<(ProgramExecutionRequest, TaskSubmission)>,
        _: u64,
    ) -> NativeExecutionResult<Vec<(ProgramExecutionResponse, Option<AttemptSuccess>)>> {
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
