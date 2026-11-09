use super::*;
use futures::{stream::FuturesUnordered, StreamExt};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) struct RemoteAnalysisTaskPool {
    pool: Arc<RemotePoolDriver>,
    pending_clock: Option<crate::remote::RemoteObservedWave>,
}

impl RemoteAnalysisTaskPool {
    pub(crate) fn new(pool: Arc<RemotePoolDriver>) -> Self {
        Self {
            pool,
            pending_clock: None,
        }
    }
}

impl CacheImport for RemoteAnalysisTaskPool {
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.pool.read_verified_bounded(digest, maximum_bytes)
    }
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.pool.read_verified(digest)
    }
}

#[async_trait]
impl AnalysisTaskPool for RemoteAnalysisTaskPool {
    async fn begin_wave_observation(
        &mut self,
        planned_tasks: &[TaskId],
    ) -> NativeExecutionResult<()> {
        if self.pending_clock.is_some() {
            return Err(protocol("remote wave observer is already pending"));
        }
        self.pending_clock = Some(
            crate::remote::RemoteObservedWave::begin(Arc::clone(&self.pool), planned_tasks).await?,
        );
        Ok(())
    }
    fn permitted_trust_tiers(
        &self,
    ) -> std::collections::BTreeSet<runmat_execution::security::ExecutionTrustTier> {
        crate::remote::pool_tasks::admitted_trust()
    }

    fn retained_committed_tasks(&self, candidates: &BTreeSet<TaskId>) -> BTreeSet<TaskId> {
        self.pool.retained_committed_task_ids(candidates)
    }

    fn scope_id(&self) -> ExecutionScopeId {
        self.pool.scope_id()
    }

    fn pool_id(&self) -> PoolId {
        self.pool.pool_id()
    }

    fn register_program(&mut self, program: &PreparedAnalysisProgram) -> NativeExecutionResult<()> {
        for (reference, bytes) in program.objects() {
            self.pool
                .register_execution_object(reference.clone(), bytes.to_vec())?;
        }
        self.pool.register_program_input_inventory(
            &program.request,
            program
                .objects()
                .map(|(reference, _)| reference.clone())
                .collect(),
        )
    }

    fn register_result_objects(
        &mut self,
        objects: &runmat_analysis_execution::PreparedAnalysisObjects,
        access: &ArtifactAccess,
    ) -> NativeExecutionResult<()> {
        let references = objects
            .references(access, runmat_execution::value::ValueRefKind::ResultObject)
            .map_err(protocol)?;
        for (object, reference) in objects.objects.iter().zip(references) {
            self.pool
                .register_execution_object(reference, object.bytes.clone())?;
        }
        Ok(())
    }

    fn register_recovery_object(
        &mut self,
        object: &runmat_execution_artifact::ProgramExecutionObject,
    ) -> NativeExecutionResult<()> {
        self.pool
            .register_execution_object(object.reference.clone(), object.bytes.clone())
    }

    async fn execute_batch(
        &mut self,
        tasks: AdmittedAnalysisBatch,
    ) -> NativeExecutionResult<Vec<AnalysisTaskCompletion>> {
        let tasks = tasks.into_submissions();
        let task_ids = tasks
            .iter()
            .map(|(_, submission)| submission.request.id)
            .collect::<Vec<_>>();
        let mut clock = self
            .pending_clock
            .take()
            .ok_or_else(|| protocol("remote analysis wave has no original begin"))?;
        clock.arm(&task_ids)?;
        let completions = self.pool.submit_graph(tasks)?;
        clock.checkpoint().await?;
        if completions.len() != task_ids.len() {
            return Err(protocol(
                "remote analysis completion inventory differs from submission",
            ));
        }
        let mut waiting = completions
            .into_iter()
            .zip(task_ids)
            .enumerate()
            .map(|(index, (completion, task_id))| async move {
                let original_graph_wall_millis = completion.original_graph_wall_millis();
                (
                    index,
                    task_id,
                    original_graph_wall_millis,
                    completion
                        .wait()
                        .await
                        .map_err(|failure| failure.to_string()),
                )
            })
            .collect::<FuturesUnordered<_>>();
        let mut responses = (0..waiting.len()).map(|_| None).collect::<Vec<_>>();
        let mut cancellation_started = None;
        while !waiting.is_empty() {
            if cancellation_started.is_none() && runmat_runtime::interrupt::is_cancelled() {
                self.pool.cancel(CancellationReason::User)?;
                cancellation_started = Some(Instant::now());
            }
            if cancellation_started
                .is_some_and(|at: Instant| at.elapsed() >= Duration::from_secs(5))
            {
                return Err(protocol(
                    "remote analysis terminal collection exceeded its cancellation cleanup bound",
                ));
            }
            tokio::select! {
                terminal = waiting.next() => {
                    let (index, task_id, original_graph_wall_millis, result) = terminal.ok_or_else(|| protocol("remote completion stream ended early"))?;
                    if result.is_err() && cancellation_started.is_none() {
                        self.pool.cancel(CancellationReason::ParentScope)?;
                        cancellation_started = Some(Instant::now());
                    }
                    let evidence = self.pool.task_execution_evidence(task_id).map_err(protocol)?;
                    let mut completion = AnalysisTaskCompletion::new(evidence, result)?;
                    completion.original_graph_wall_millis = original_graph_wall_millis;
                    responses[index] = Some(completion);
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
        }
        if responses
            .iter()
            .flatten()
            .any(|response| response.evidence.task.cancellation.is_some())
        {
            let control_acknowledged = tokio::time::timeout(
                Duration::from_secs(5),
                self.pool.wait_for_cancellation_control(),
            )
            .await
            .unwrap_or(false);
            for response in responses.iter_mut().flatten() {
                if let Some(failure) = &mut response.failure {
                    failure.observe_remote_control_acknowledgement(control_acknowledged);
                }
            }
        }
        if responses.iter().flatten().all(|response| {
            response.failure.is_none()
                && response.evidence.task.state == runmat_execution::state::TaskState::Succeeded
        }) {
            let observed_tasks = clock.observed_tasks().to_vec();
            if let Some(wall) = clock.close().await? {
                for response in responses.iter_mut().flatten() {
                    if observed_tasks.contains(&response.evidence.task.id()) {
                        response.original_graph_wall_millis = Some(wall);
                    }
                }
            }
        }
        responses
            .into_iter()
            .map(|response| {
                response.ok_or_else(|| protocol("remote analysis task has no terminal response"))
            })
            .collect()
    }
}
