use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use runmat_analysis_execution::{AnalysisHostWorkload, PreparedAnalysisProgram};
use runmat_analysis_fea::execution::AnalysisTaskIdentity;
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::resource::ResourceRequest;
use runmat_execution::task::RetryPolicy;
use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::cache::{CacheImport, FilesystemObjectStore};
use runmat_execution_artifact::{
    ArtifactAccess, ArtifactResult, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_execution_runner::{AttemptSuccess, TaskSubmission};

use crate::analysis_task::{prepare_analysis_task, AnalysisTaskPolicy};
use crate::{NativeExecutionError, NativeExecutionResult, NativeProgramSession, RemotePoolDriver};

/// Placement-neutral pool used by FEA task-graph coordinators.
///
/// Physics coordinators prepare exact programs, dependencies, and resource requests. This boundary
/// owns only program registration, task submission, cancellation, and terminal collection.
#[async_trait]
pub(crate) trait AnalysisTaskPool: CacheImport {
    fn scope_id(&self) -> ExecutionScopeId;
    fn pool_id(&self) -> PoolId;
    fn register_program(&mut self, program: &PreparedAnalysisProgram) -> NativeExecutionResult<()>;
    fn register_result_objects(
        &mut self,
        objects: &runmat_analysis_execution::PreparedAnalysisObjects,
        access: &ArtifactAccess,
    ) -> NativeExecutionResult<()>;
    async fn execute_batch(
        &mut self,
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
        maximum_wall_millis: u64,
    ) -> NativeExecutionResult<Vec<(ProgramExecutionResponse, Option<AttemptSuccess>)>>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AnalysisTaskDependency {
    pub identity: AnalysisTaskIdentity,
    pub task_id: TaskId,
}

pub(crate) struct PreparedAnalysisTask<'a> {
    pub host: &'a AnalysisHostWorkload,
    pub program: &'a PreparedAnalysisProgram,
    pub resource_plan: Option<&'a FeaResourcePlan>,
    pub fallback_resources: ResourceRequest,
    pub dependencies: Vec<AnalysisTaskDependency>,
}

pub(crate) struct AnalysisTaskBatchPolicy {
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

pub(crate) struct ExecutedAnalysisTask {
    pub identity: AnalysisTaskIdentity,
    pub task_id: TaskId,
    pub response: ProgramExecutionResponse,
    pub attempt: Option<AttemptSuccess>,
}

/// Registers and executes one independent task wave through the shared task substrate.
pub(crate) async fn execute_analysis_task_batch(
    pool: &mut impl AnalysisTaskPool,
    tasks: Vec<PreparedAnalysisTask<'_>>,
    policy: AnalysisTaskBatchPolicy,
) -> NativeExecutionResult<Vec<ExecutedAnalysisTask>> {
    let mut submissions = Vec::with_capacity(tasks.len());
    let mut admitted = Vec::with_capacity(tasks.len());
    for task in tasks {
        pool.register_program(task.program)?;
        let dependency_identities = task
            .dependencies
            .iter()
            .map(|dependency| dependency.identity)
            .collect::<Vec<_>>();
        let dependency_tasks = task
            .dependencies
            .iter()
            .map(|dependency| dependency.task_id)
            .collect::<BTreeSet<_>>();
        let resources = task
            .resource_plan
            .map_or(task.fallback_resources, |plan| plan.request.clone());
        let (submission, identity) = prepare_analysis_task(
            pool.scope_id(),
            pool.pool_id(),
            task.host,
            task.program,
            dependency_identities,
            dependency_tasks,
            AnalysisTaskPolicy {
                resources,
                resource_plan: task.resource_plan,
                retry: policy.retry,
                deadline_unix_millis: policy.deadline_unix_millis,
                priority: policy.priority,
            },
        )?;
        let task_id = submission.request.id;
        admitted.push((identity, task_id));
        submissions.push((task.program.request.clone(), submission));
    }
    let maximum_wall_millis = submissions
        .iter()
        .map(|(_, submission)| submission.request.resources.max_wall_millis)
        .max()
        .unwrap_or(0);
    let responses = pool.execute_batch(submissions, maximum_wall_millis).await?;
    if responses.len() != admitted.len() {
        return Err(NativeExecutionError::Protocol(
            "analysis task response inventory differs from its submitted wave".into(),
        ));
    }
    Ok(admitted
        .into_iter()
        .zip(responses)
        .map(
            |((identity, task_id), (response, attempt))| ExecutedAnalysisTask {
                identity,
                task_id,
                response,
                attempt,
            },
        )
        .collect())
}

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

    pub(crate) fn object_store(&self) -> FilesystemObjectStore {
        self.store.clone()
    }
}

impl CacheImport for LocalAnalysisTaskPool {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified(digest)
    }
}

#[async_trait]
impl AnalysisTaskPool for LocalAnalysisTaskPool {
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
        for object in &objects.objects {
            runmat_execution_artifact::cache::CacheExport::write_verified(&mut self.store, object)
                .map_err(protocol)?;
        }
        Ok(())
    }

    async fn execute_batch(
        &mut self,
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
        maximum_wall_millis: u64,
    ) -> NativeExecutionResult<Vec<(ProgramExecutionResponse, Option<AttemptSuccess>)>> {
        let tasks = self.session.submit_graph(tasks)?;
        let started = Instant::now();
        let maximum_wall = Duration::from_millis(maximum_wall_millis);
        let mut responses = (0..tasks.len()).map(|_| None).collect::<Vec<_>>();
        let mut failure = None;
        let mut deadline_exceeded = false;
        let mut user_cancelled = runmat_runtime::interrupt::is_cancelled();
        let mut cancellation_started = user_cancelled.then(Instant::now);
        if user_cancelled {
            self.session.cancel(CancellationReason::User);
        }
        while responses.iter().any(Option::is_none) {
            for (index, task) in tasks.iter().enumerate() {
                if responses[index].is_some() {
                    continue;
                }
                if let Some(result) = task.try_result() {
                    match result {
                        Ok(success) => {
                            responses[index] =
                                Some((success_response(success.clone()), Some(success)))
                        }
                        Err(message) => {
                            if !deadline_exceeded && !user_cancelled {
                                if failure.is_none() {
                                    self.session.cancel(CancellationReason::ParentScope);
                                    cancellation_started = Some(Instant::now());
                                }
                                failure.get_or_insert(message);
                            }
                            responses[index] = Some((
                                ProgramExecutionResponse::Failure {
                                    message: "analysis peer failed".into(),
                                },
                                None,
                            ));
                        }
                    }
                }
            }
            if responses.iter().any(Option::is_none) {
                if !deadline_exceeded
                    && failure.is_none()
                    && !user_cancelled
                    && runmat_runtime::interrupt::is_cancelled()
                {
                    self.session.cancel(CancellationReason::User);
                    user_cancelled = true;
                    cancellation_started = Some(Instant::now());
                }
                if !deadline_exceeded
                    && !user_cancelled
                    && failure.is_none()
                    && started.elapsed() >= maximum_wall
                {
                    self.session.cancel(CancellationReason::Deadline);
                    deadline_exceeded = true;
                    cancellation_started = Some(Instant::now());
                }
                if cancellation_started
                    .is_some_and(|cancelled| cancelled.elapsed() >= Duration::from_secs(5))
                {
                    return Err(NativeExecutionError::Protocol(
                        "local analysis cancellation did not reach quiescence".into(),
                    ));
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        if user_cancelled {
            return Err(NativeExecutionError::Cancelled);
        }
        if let Some(message) = failure {
            return Err(NativeExecutionError::Protocol(format!(
                "local analysis task failed: {message}"
            )));
        }
        if deadline_exceeded {
            return Err(NativeExecutionError::Protocol(
                "local analysis batch exceeded its wall-time limit after reaching quiescence"
                    .into(),
            ));
        }
        responses
            .into_iter()
            .map(|response| {
                response.ok_or_else(|| {
                    NativeExecutionError::Protocol(
                        "local analysis task has no terminal response".into(),
                    )
                })
            })
            .collect()
    }
}

pub(crate) struct RemoteAnalysisTaskPool {
    pool: Arc<RemotePoolDriver>,
}

impl RemoteAnalysisTaskPool {
    pub(crate) fn new(pool: Arc<RemotePoolDriver>) -> Self {
        Self { pool }
    }

    pub(crate) fn register_execution_object(
        &self,
        reference: runmat_execution::value::ValueRef,
        bytes: impl Into<Arc<[u8]>>,
    ) -> NativeExecutionResult<()> {
        self.pool.register_execution_object(reference, bytes)
    }
}

impl CacheImport for RemoteAnalysisTaskPool {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.pool.read_verified(digest)
    }
}

#[async_trait]
impl AnalysisTaskPool for RemoteAnalysisTaskPool {
    fn scope_id(&self) -> ExecutionScopeId {
        self.pool.scope_id()
    }

    fn pool_id(&self) -> PoolId {
        self.pool.pool_id()
    }

    fn register_program(&mut self, program: &PreparedAnalysisProgram) -> NativeExecutionResult<()> {
        for object in &program.objects {
            self.pool
                .register_execution_object(object.reference.clone(), object.bytes.clone())?;
        }
        Ok(())
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

    async fn execute_batch(
        &mut self,
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
        maximum_wall_millis: u64,
    ) -> NativeExecutionResult<Vec<(ProgramExecutionResponse, Option<AttemptSuccess>)>> {
        let completions = self.pool.submit_graph(tasks)?;
        let mut responses = Vec::with_capacity(completions.len());
        let mut failure = None;
        let started = Instant::now();
        let maximum_wall = Duration::from_millis(maximum_wall_millis);
        let mut deadline_exceeded = false;
        let mut cancellation_started: Option<Instant> = None;
        for completion in completions {
            let mut waiting = Box::pin(completion.wait());
            let terminal = if let Some(cancelled) = cancellation_started {
                let remaining = Duration::from_secs(5).saturating_sub(cancelled.elapsed());
                tokio::time::timeout(remaining, &mut waiting)
                    .await
                    .map_err(|_| {
                        NativeExecutionError::Protocol(
                            "remote analysis cancellation did not reach quiescence".into(),
                        )
                    })?
            } else {
                let remaining = maximum_wall.saturating_sub(started.elapsed());
                match tokio::time::timeout(remaining, &mut waiting).await {
                    Ok(terminal) => terminal,
                    Err(_) => {
                        self.pool.cancel(CancellationReason::Deadline)?;
                        deadline_exceeded = true;
                        cancellation_started = Some(Instant::now());
                        tokio::time::timeout(Duration::from_secs(5), &mut waiting)
                            .await
                            .map_err(|_| {
                                NativeExecutionError::Protocol(
                                    "remote analysis deadline cancellation did not reach quiescence"
                                        .into(),
                                )
                            })?
                    }
                }
            };
            match terminal {
                Ok(success) => responses.push((success_response(success.clone()), Some(success))),
                Err(message) => {
                    if !deadline_exceeded {
                        failure.get_or_insert(message);
                    }
                    responses.push((
                        ProgramExecutionResponse::Failure {
                            message: "analysis peer failed".into(),
                        },
                        None,
                    ));
                }
            }
        }
        if let Some(message) = failure {
            return Err(NativeExecutionError::Protocol(format!(
                "remote analysis task failed: {message}"
            )));
        }
        if deadline_exceeded {
            return Err(NativeExecutionError::Protocol(
                "remote analysis batch exceeded its wall-time limit after reaching quiescence"
                    .into(),
            ));
        }
        Ok(responses)
    }
}

fn success_response(success: AttemptSuccess) -> ProgramExecutionResponse {
    ProgramExecutionResponse::ExternalizedSuccess {
        outputs: success.outputs,
        result_objects: success.result_objects,
    }
}

pub(crate) fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
