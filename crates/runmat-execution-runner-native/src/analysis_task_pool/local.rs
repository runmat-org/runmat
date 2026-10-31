use super::*;
use std::time::{Duration, Instant};

pub(crate) struct LocalAnalysisTaskPool {
    session: NativeProgramSession,
    store: FilesystemObjectStore,
}

impl LocalAnalysisTaskPool {
    pub(crate) fn new(session: NativeProgramSession) -> Self {
        Self {
            store: session.object_store(),
            session,
        }
    }

    pub(crate) fn object_source(&self) -> crate::NativeProgramObjectSource {
        self.session.object_source()
    }
}

impl CacheImport for LocalAnalysisTaskPool {
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified_bounded(digest, maximum_bytes)
    }
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified(digest)
    }
}

#[async_trait]
impl AnalysisTaskPool for LocalAnalysisTaskPool {
    async fn begin_wave_observation(
        &mut self,
        _planned_tasks: &[TaskId],
    ) -> NativeExecutionResult<()> {
        Ok(())
    }
    fn scope_id(&self) -> ExecutionScopeId {
        self.session.scope_id()
    }

    fn pool_id(&self) -> PoolId {
        self.session.pool_id()
    }

    fn register_program(&mut self, program: &PreparedAnalysisProgram) -> NativeExecutionResult<()> {
        program.write_to(&mut self.store).map_err(protocol)
    }

    fn register_result_objects(
        &mut self,
        objects: &runmat_analysis_execution::PreparedAnalysisObjects,
        _access: &ArtifactAccess,
    ) -> NativeExecutionResult<()> {
        runmat_execution_artifact::cache::CacheExport::write_verified_batch(
            &mut self.store,
            &objects.objects,
        )
        .map_err(protocol)
    }

    fn register_recovery_object(
        &mut self,
        object: &runmat_execution_artifact::ProgramExecutionObject,
    ) -> NativeExecutionResult<()> {
        object.validate().map_err(protocol)?;
        let logical = runmat_execution_artifact::LogicalObject::new(
            runmat_execution_artifact::ObjectNamespace::Checkpoint,
            "analysis-recovery-object",
            object.reference.media_type.clone(),
            object.bytes.clone(),
        )
        .map_err(protocol)?;
        if logical.descriptor.digest != object.reference.logical_digest {
            return Err(protocol(
                "analysis recovery object bytes differ from their portable reference",
            ));
        }
        runmat_execution_artifact::cache::CacheExport::write_verified(&mut self.store, &logical)
            .map_err(protocol)
    }

    async fn execute_batch(
        &mut self,
        tasks: AdmittedAnalysisBatch,
    ) -> NativeExecutionResult<Vec<AnalysisTaskCompletion>> {
        let tasks = self.session.submit_graph(tasks.into_submissions())?;
        let mut responses = (0..tasks.len()).map(|_| None).collect::<Vec<_>>();
        let mut cancellation_started = None;
        while responses.iter().any(Option::is_none) {
            if cancellation_started.is_none() && runmat_runtime::interrupt::is_cancelled() {
                self.session.cancel(CancellationReason::User);
                cancellation_started = Some(Instant::now());
            }
            for (index, task) in tasks.iter().enumerate() {
                if responses[index].is_some() {
                    continue;
                }
                if let Some(completion) = task.try_completion() {
                    if completion.delivery_result().is_err() && cancellation_started.is_none() {
                        self.session.cancel(CancellationReason::ParentScope);
                        cancellation_started = Some(Instant::now());
                    }
                    responses[index] = Some(AnalysisTaskCompletion::from_native(completion)?);
                }
            }
            if responses.iter().any(Option::is_none) {
                // Cleanup is bounded separately; queued/retried work is governed by
                // the shared driver's per-attempt and absolute task deadlines.
                if cancellation_started
                    .is_some_and(|at: Instant| at.elapsed() >= Duration::from_secs(5))
                {
                    return Err(protocol(
                        "local analysis terminal collection exceeded its cancellation cleanup bound",
                    ));
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        responses
            .into_iter()
            .map(|response| {
                response.ok_or_else(|| protocol("local analysis task has no terminal response"))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution_artifact::{
        cache::CacheExport, ArtifactError, LogicalObject, ObjectNamespace,
    };

    #[test]
    fn local_analysis_pool_preserves_bounded_object_read() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.store_root = directory.path().join("sessions");
        let session = NativeProgramSession::new(config).unwrap();
        let mut store = session.object_store();
        let object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "bounded",
            "test/v1",
            b"data".to_vec(),
        )
        .unwrap();
        store.write_verified(&object).unwrap();
        let pool = LocalAnalysisTaskPool::new(session);
        for limit in [0, 3] {
            assert!(matches!(
                pool.read_verified_bounded(object.descriptor.digest, limit),
                Err(ArtifactError::Limit(_))
            ));
        }
        assert_eq!(
            pool.read_verified_bounded(object.descriptor.digest, 4)
                .unwrap(),
            Some(object.bytes)
        );
    }
}
