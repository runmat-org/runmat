use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use runmat_analysis_execution::{
    build_analysis_task_submission, AnalysisExecutionContext, AnalysisHostWorkload,
    CompletedStructuralSequence, PreparedAnalysisProgram, PreparedStructuralSolveWave,
    PreparedStructuralStateWave, StructuralSequenceCoordinator, StructuralSequenceNode,
    StructuralSequencePreparation,
};
use runmat_analysis_fea::execution::AnalysisTaskIdentity;
use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValuePayload;
use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::cache::{CacheImport, FilesystemObjectStore};
use runmat_execution_artifact::{
    ArtifactResult, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_execution_runner::{AttemptSuccess, TaskSubmission};

use crate::{
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver,
};

#[derive(Clone, Debug)]
pub struct StructuralSequenceExecutionPolicy {
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

impl StructuralSequenceExecutionPolicy {
    fn validate(&self) -> NativeExecutionResult<()> {
        self.resources
            .validate()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))
    }
}

pub async fn execute_structural_sequence_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralSequencePreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralSequence> {
    policy.validate()?;
    config
        .worker_capabilities
        .extend(policy.resources.required_capabilities.iter().cloned());
    config
        .worker_capabilities
        .insert(Capability::ProcessIsolation);
    config
        .worker_capabilities
        .insert(Capability::Custom(format!(
            "runmat.analysis.host:{}",
            runmat_analysis_execution::ANALYSIS_HOST_ABI
        )));
    let session = NativeProgramSession::new(config)?;
    execute_with_pool(
        &mut LocalSequencePool {
            store: session.object_store(),
            session,
        },
        preparation,
        policy,
    )
    .await
}

pub async fn execute_structural_sequence_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralSequencePreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralSequence> {
    policy.validate()?;
    execute_with_pool(&mut RemoteSequencePool { pool }, preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut (impl SequencePool + CacheImport),
    preparation: StructuralSequencePreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralSequence> {
    let mut coordinator = StructuralSequenceCoordinator::new(preparation).map_err(protocol)?;
    let mut completed_nodes = BTreeMap::new();
    while !coordinator.is_complete() {
        let solve_wave = coordinator.prepare_next_solve_wave().map_err(protocol)?;
        let solve_responses =
            execute_solve_wave(pool, &solve_wave, &policy, &mut completed_nodes).await?;
        let state_wave = coordinator
            .complete_solve_wave(pool, solve_wave, solve_responses)
            .map_err(protocol)?;
        let state_responses =
            execute_state_wave(pool, &state_wave, &policy, &mut completed_nodes).await?;
        coordinator
            .complete_state_wave(pool, state_wave, state_responses)
            .map_err(protocol)?;
    }
    coordinator.finish().map_err(protocol)
}

async fn execute_solve_wave(
    pool: &mut (impl SequencePool + CacheImport),
    wave: &PreparedStructuralSolveWave,
    policy: &StructuralSequenceExecutionPolicy,
    completed_nodes: &mut BTreeMap<StructuralSequenceNode, (AnalysisTaskIdentity, TaskId)>,
) -> NativeExecutionResult<BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>> {
    let tasks = wave.tasks.iter().map(|task| TaskView {
        node: &task.node,
        dependencies: &task.dependencies,
        host: &task.prepared.host,
        program: &task.prepared.program,
    });
    execute_wave(pool, tasks, policy, completed_nodes).await
}

async fn execute_state_wave(
    pool: &mut (impl SequencePool + CacheImport),
    wave: &PreparedStructuralStateWave,
    policy: &StructuralSequenceExecutionPolicy,
    completed_nodes: &mut BTreeMap<StructuralSequenceNode, (AnalysisTaskIdentity, TaskId)>,
) -> NativeExecutionResult<BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>> {
    let tasks = wave.tasks.iter().map(|task| TaskView {
        node: &task.node,
        dependencies: &task.dependencies,
        host: &task.prepared.host,
        program: &task.prepared.program,
    });
    execute_wave(pool, tasks, policy, completed_nodes).await
}

async fn execute_wave<'a>(
    pool: &mut (impl SequencePool + CacheImport),
    tasks: impl IntoIterator<Item = TaskView<'a>>,
    policy: &StructuralSequenceExecutionPolicy,
    completed_nodes: &mut BTreeMap<StructuralSequenceNode, (AnalysisTaskIdentity, TaskId)>,
) -> NativeExecutionResult<BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>> {
    let mut submissions = Vec::new();
    let mut admitted = Vec::new();
    for task in tasks {
        pool.register_program(task.program)?;
        let dependency_records = task
            .dependencies
            .iter()
            .map(|dependency| {
                completed_nodes.get(dependency).copied().ok_or_else(|| {
                    NativeExecutionError::Protocol(format!(
                        "sequence node {dependency:?} has not completed"
                    ))
                })
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        let dependency_identities = dependency_records
            .iter()
            .map(|(identity, _)| *identity)
            .collect::<Vec<_>>();
        let dependency_tasks = dependency_records
            .iter()
            .map(|(_, task_id)| *task_id)
            .collect::<BTreeSet<_>>();
        let roots = task
            .program
            .request
            .arguments
            .iter()
            .map(|argument| match argument {
                ValuePayload::Object(root) => Ok((**root).clone()),
                ValuePayload::Inline(_) => Err(NativeExecutionError::Protocol(
                    "sequence analysis input was unexpectedly inline".into(),
                )),
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        let mut resources = policy.resources.clone();
        resources.required_capabilities.extend(
            task.host
                .workload
                .generic_capabilities()
                .map_err(protocol)?,
        );
        resources
            .required_capabilities
            .insert(Capability::ProcessIsolation);
        let submission = build_analysis_task_submission(
            task.host,
            roots,
            dependency_identities.clone(),
            dependency_tasks,
            AnalysisExecutionContext {
                scope_id: pool.scope_id(),
                pool_id: pool.pool_id(),
                program_artifact_id: ArtifactId::derive(&[task
                    .program
                    .request
                    .artifact
                    .id
                    .0
                    .bytes()]),
                resources,
                retry: policy.retry,
                deadline_unix_millis: policy.deadline_unix_millis,
                priority: policy.priority,
            },
        )
        .map_err(protocol)?;
        let identity =
            AnalysisTaskIdentity::new(task.host.workload.identity, dependency_identities)
                .map_err(protocol)?;
        if submission.request.id != TaskId::derive(&[identity.digest().bytes()]) {
            return Err(NativeExecutionError::Protocol(
                "sequence task identity does not converge".into(),
            ));
        }
        if completed_nodes.contains_key(task.node)
            || admitted
                .iter()
                .any(|(node, _, _): &(StructuralSequenceNode, _, _)| node == task.node)
        {
            return Err(NativeExecutionError::Protocol(
                "sequence node was submitted more than once".into(),
            ));
        }
        admitted.push((task.node.clone(), identity, submission.request.id));
        submissions.push((task.program.request.clone(), submission));
    }
    let responses = pool
        .execute_batch(submissions, policy.resources.max_wall_millis)
        .await?;
    if responses.len() != admitted.len() {
        return Err(NativeExecutionError::Protocol(
            "sequence task responses differ from the submitted wave".into(),
        ));
    }
    let mut by_node = BTreeMap::new();
    for ((node, identity, task_id), response) in admitted.into_iter().zip(responses) {
        if by_node.insert(node.clone(), response).is_some() {
            return Err(NativeExecutionError::Protocol(
                "sequence wave returned a duplicate node".into(),
            ));
        }
        completed_nodes.insert(node, (identity, task_id));
    }
    Ok(by_node)
}

struct TaskView<'a> {
    node: &'a StructuralSequenceNode,
    dependencies: &'a [StructuralSequenceNode],
    host: &'a AnalysisHostWorkload,
    program: &'a PreparedAnalysisProgram,
}

#[async_trait]
trait SequencePool {
    fn scope_id(&self) -> ExecutionScopeId;
    fn pool_id(&self) -> PoolId;
    fn register_program(&mut self, program: &PreparedAnalysisProgram) -> NativeExecutionResult<()>;
    async fn execute_batch(
        &mut self,
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
        maximum_wall_millis: u64,
    ) -> NativeExecutionResult<Vec<ProgramExecutionResponse>>;
}

struct LocalSequencePool {
    session: NativeProgramSession,
    store: FilesystemObjectStore,
}

impl CacheImport for LocalSequencePool {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified(digest)
    }
}

#[async_trait]
impl SequencePool for LocalSequencePool {
    fn scope_id(&self) -> ExecutionScopeId {
        self.session.scope_id()
    }

    fn pool_id(&self) -> PoolId {
        self.session.pool_id()
    }

    fn register_program(&mut self, program: &PreparedAnalysisProgram) -> NativeExecutionResult<()> {
        program.write_to(&mut self.store).map_err(protocol)
    }

    async fn execute_batch(
        &mut self,
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
        maximum_wall_millis: u64,
    ) -> NativeExecutionResult<Vec<ProgramExecutionResponse>> {
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
                        Ok(success) => responses[index] = Some(success_response(success)),
                        Err(message) => {
                            if !deadline_exceeded && !user_cancelled {
                                if failure.is_none() {
                                    self.session.cancel(CancellationReason::ParentScope);
                                    cancellation_started = Some(Instant::now());
                                }
                                failure.get_or_insert(message);
                            }
                            responses[index] = Some(ProgramExecutionResponse::Failure {
                                message: "sequence peer failed".into(),
                            });
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
                        "local structural sequence cancellation did not reach quiescence".into(),
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
                "local structural sequence task failed: {message}"
            )));
        }
        if deadline_exceeded {
            return Err(NativeExecutionError::Protocol(
                "local structural sequence wave exceeded its wall-time limit after reaching quiescence"
                    .into(),
            ));
        }
        responses
            .into_iter()
            .map(|response| {
                response.ok_or_else(|| {
                    NativeExecutionError::Protocol(
                        "local sequence task has no terminal response".into(),
                    )
                })
            })
            .collect()
    }
}

struct RemoteSequencePool {
    pool: Arc<RemotePoolDriver>,
}

impl CacheImport for RemoteSequencePool {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.pool.read_verified(digest)
    }
}

#[async_trait]
impl SequencePool for RemoteSequencePool {
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

    async fn execute_batch(
        &mut self,
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
        maximum_wall_millis: u64,
    ) -> NativeExecutionResult<Vec<ProgramExecutionResponse>> {
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
                            "remote structural sequence cancellation did not reach quiescence"
                                .into(),
                        )
                    })?
            } else {
                let remaining = maximum_wall.saturating_sub(started.elapsed());
                match tokio::time::timeout(remaining, &mut waiting).await {
                    Ok(terminal) => terminal,
                    Err(_) => {
                        self.pool.cancel(CancellationReason::Deadline)?;
                        deadline_exceeded = true;
                        let cancelled = Instant::now();
                        cancellation_started = Some(cancelled);
                        tokio::time::timeout(Duration::from_secs(5), &mut waiting)
                            .await
                            .map_err(|_| {
                                NativeExecutionError::Protocol(
                                    "remote structural sequence deadline cancellation did not reach quiescence"
                                        .into(),
                                )
                            })?
                    }
                }
            };
            match terminal {
                Ok(success) => responses.push(success_response(success)),
                Err(message) => {
                    if !deadline_exceeded {
                        failure.get_or_insert(message);
                    }
                    responses.push(ProgramExecutionResponse::Failure {
                        message: "sequence peer failed".into(),
                    });
                }
            }
        }
        if let Some(message) = failure {
            return Err(NativeExecutionError::Protocol(format!(
                "remote structural sequence task failed: {message}"
            )));
        }
        if deadline_exceeded {
            return Err(NativeExecutionError::Protocol(
                "remote structural sequence wave exceeded its wall-time limit after reaching quiescence"
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

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
