use std::collections::BTreeSet;

use async_trait::async_trait;
use runmat_analysis_execution::{
    AnalysisHostWorkload, PlannedResourceUsageAttempt, PreparedAnalysisProgram,
    PreparedAnalysisTaskClosure,
};
use runmat_analysis_fea::execution::AnalysisTaskIdentity;
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::cache::{CacheImport, FilesystemObjectStore};
use runmat_execution_artifact::{
    ArtifactAccess, ArtifactResult, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_execution_runner::{AttemptSuccess, TaskExecutionEvidence, TaskSubmission};

use crate::analysis_task::{analysis_task_id, prepare_analysis_task, AnalysisTaskPolicy};
use crate::{
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver,
};

mod completion;
mod local;
mod phase;
mod remote;
pub(crate) use completion::{delivered_attempt, AnalysisTaskCompletion};
pub(crate) use local::LocalAnalysisTaskPool;
pub(crate) use remote::RemoteAnalysisTaskPool;
#[cfg(test)]
pub(crate) mod phase_fixture;
#[cfg(test)]
mod tests;
pub(crate) use phase::{execute_governed_analysis_phase, execute_governed_analysis_task};

pub(crate) fn configure_local_analysis(
    config: &mut NativeExecutionConfig,
    resources: &ResourceRequest,
) {
    config.worker_capabilities.extend(
        resources
            .accelerators
            .iter()
            .map(|accelerator| Capability::Accelerator(accelerator.class.clone())),
    );
    config
        .worker_capabilities
        .extend(resources.required_capabilities.iter().cloned());
    config
        .worker_capabilities
        .insert(Capability::ProcessIsolation);
    config
        .worker_capabilities
        .insert(Capability::Custom(format!(
            "runmat.analysis.host:{}",
            runmat_analysis_execution::ANALYSIS_HOST_ABI
        )));
}

/// Configure an explicit authored per-worker envelope, not a fallback task request.
/// Existing placement policies may supply task hints while retaining independently
/// configured worker capacity; they must not implicitly resize their pools.
pub(crate) fn configure_local_analysis_envelope(
    config: &mut NativeExecutionConfig,
    envelope: &ResourceRequest,
) {
    config.worker_cpu_millicores = envelope.cpu_millicores;
    config.worker_memory_bytes = envelope.memory_bytes;
    // Native workers require a nonempty scratch inventory even for tasks reserving none.
    config.worker_scratch_bytes = envelope.scratch_bytes.max(1);
    configure_local_analysis(config, envelope);
}

/// Placement-neutral pool used by FEA task-graph coordinators.
///
/// Physics coordinators prepare exact programs, dependencies, and resource requests. This boundary
/// owns only program registration, task submission, cancellation, and terminal collection.
#[async_trait]
pub(crate) trait AnalysisTaskPool: CacheImport {
    fn scope_id(&self) -> ExecutionScopeId;
    fn pool_id(&self) -> PoolId;
    /// Begin one wave before program registration, submission lowering, or queue admission.
    /// Typed host/program plans may already exist; this is not their preparation interval.
    async fn begin_wave_observation(
        &mut self,
        planned_tasks: &[TaskId],
    ) -> NativeExecutionResult<()>;
    /// Trust tiers this pool's workers may hold; task host requirements narrow to them.
    fn permitted_trust_tiers(
        &self,
    ) -> std::collections::BTreeSet<runmat_execution::security::ExecutionTrustTier> {
        crate::program_host::local_trust()
    }
    /// A portable recovery closure can outlive its original scheduler. Only a
    /// retained, committed scheduler record may restore a physical dependency edge.
    fn retained_committed_tasks(&self, _candidates: &BTreeSet<TaskId>) -> BTreeSet<TaskId> {
        BTreeSet::new()
    }
    fn register_program(&mut self, program: &PreparedAnalysisProgram) -> NativeExecutionResult<()>;
    fn register_result_objects(
        &mut self,
        objects: &runmat_analysis_execution::PreparedAnalysisObjects,
        access: &ArtifactAccess,
    ) -> NativeExecutionResult<()>;
    fn register_recovery_object(
        &mut self,
        object: &runmat_execution_artifact::ProgramExecutionObject,
    ) -> NativeExecutionResult<()>;
    async fn execute_batch(
        &mut self,
        tasks: AdmittedAnalysisBatch,
    ) -> NativeExecutionResult<Vec<AnalysisTaskCompletion>>;
}

/// Only begin_analysis_batch can issue this pre-submission wave authority.
/// The submitted inventory must be the exact one observed before lowering.
pub(crate) struct AnalysisBatchAdmission {
    planned_tasks: Vec<TaskId>,
}

pub(crate) struct AdmittedAnalysisBatch {
    submissions: Vec<(ProgramExecutionRequest, TaskSubmission)>,
}

impl AnalysisBatchAdmission {
    pub(crate) fn admit(
        self,
        submissions: Vec<(ProgramExecutionRequest, TaskSubmission)>,
    ) -> NativeExecutionResult<AdmittedAnalysisBatch> {
        if submissions.len() != self.planned_tasks.len()
            || submissions
                .iter()
                .zip(&self.planned_tasks)
                .any(|((_, submission), expected)| submission.request.id != *expected)
        {
            return Err(protocol(
                "analysis submission differs from its pre-preparation wave",
            ));
        }
        Ok(AdmittedAnalysisBatch { submissions })
    }
}

impl AdmittedAnalysisBatch {
    pub(crate) fn into_submissions(self) -> Vec<(ProgramExecutionRequest, TaskSubmission)> {
        self.submissions
    }
}

pub(crate) async fn begin_analysis_batch(
    pool: &mut impl AnalysisTaskPool,
    planned_tasks: &[TaskId],
) -> NativeExecutionResult<AnalysisBatchAdmission> {
    let unique = planned_tasks.iter().copied().collect::<BTreeSet<_>>();
    if planned_tasks.is_empty() || unique.len() != planned_tasks.len() {
        return Err(protocol(
            "analysis wave requires a nonempty unique task inventory",
        ));
    }
    pool.begin_wave_observation(planned_tasks).await?;
    Ok(AnalysisBatchAdmission {
        planned_tasks: planned_tasks.to_vec(),
    })
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
    pub prepared_closure: Option<PreparedAnalysisTaskClosure<'a>>,
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
    pub evidence: TaskExecutionEvidence,
    pub graph_wall_millis: u64,
    pub failure: Option<crate::task_outcome::NativeTaskFailure>,
}

pub(crate) fn planned_resource_attempt(
    plan: &FeaResourcePlan,
    task: &ExecutedAnalysisTask,
) -> NativeExecutionResult<PlannedResourceUsageAttempt> {
    task.evidence.validate().map_err(protocol)?;
    if task.evidence.task.id() != task.task_id {
        return Err(protocol(
            "analysis task history belongs to another admitted task",
        ));
    }
    let success = delivered_attempt(&task.response, task.attempt.as_ref(), task.failure.as_ref())?;
    success.validate_accounting().map_err(protocol)?;
    let accounting = success.resources().cloned().ok_or_else(|| {
        NativeExecutionError::Protocol("analysis task lacks scheduler resource accounting".into())
    })?;
    if !runmat_analysis_fea::planning::resource::allocation_matches_plan(
        &accounting.resources.allocation,
        &plan.request,
    ) {
        return Err(NativeExecutionError::Protocol(format!(
            "analysis scheduler allocation differs from its admitted resource plan: actual={:?}, planned={:?}",
            accounting.resources.allocation, plan.request
        )));
    }
    Ok(PlannedResourceUsageAttempt {
        plan: plan.clone(),
        accounting,
    })
}

pub(crate) fn planned_phase_execution(
    plan: &FeaResourcePlan,
    task: &ExecutedAnalysisTask,
) -> NativeExecutionResult<runmat_analysis_execution::AnalysisPhaseExecutionEvidence> {
    // Delivery errors retain their original diagnostic before any usage gate.
    delivered_attempt(&task.response, task.attempt.as_ref(), task.failure.as_ref())?;
    if task.evidence.task.id() != task.task_id {
        return Err(protocol("phase evidence belongs to another admitted task"));
    }
    let execution = runmat_analysis_execution::AnalysisPhaseExecutionEvidence::new(
        task.evidence.clone(),
        task.graph_wall_millis,
    )
    .map_err(protocol)?;
    execution.resource_graph(plan).map_err(protocol)?;
    Ok(execution)
}

/// Registers and executes one independent task wave through the shared task substrate.
pub(crate) async fn execute_analysis_task_batch(
    pool: &mut impl AnalysisTaskPool,
    tasks: Vec<PreparedAnalysisTask<'_>>,
    policy: AnalysisTaskBatchPolicy,
) -> NativeExecutionResult<Vec<ExecutedAnalysisTask>> {
    let started = std::time::Instant::now();
    // Optional observation/history waves can contain no admitted work. Do not submit an
    // empty native graph or manufacture scheduler attempts for either placement.
    if tasks.is_empty() {
        return Ok(Vec::new());
    }
    let planned_tasks = tasks
        .iter()
        .map(|task| {
            let dependencies = task
                .dependencies
                .iter()
                .map(|dep| dep.identity)
                .collect::<Vec<_>>();
            AnalysisTaskIdentity::new(task.host.workload.identity, dependencies)
                .map(analysis_task_id)
                .map_err(protocol)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let admission = begin_analysis_batch(pool, &planned_tasks).await?;
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
                prepared_closure: task.prepared_closure,
                retry: policy.retry,
                deadline_unix_millis: policy.deadline_unix_millis,
                priority: policy.priority,
                permitted_trust_tiers: pool.permitted_trust_tiers(),
            },
        )?;
        let task_id = submission.request.id;
        admitted.push((identity, task_id));
        submissions.push((task.program.request.clone(), submission));
    }
    let responses = pool.execute_batch(admission.admit(submissions)?).await?;
    if responses.len() != admitted.len() {
        return Err(NativeExecutionError::Protocol(
            "analysis task response inventory differs from its submitted wave".into(),
        ));
    }
    if runmat_runtime::interrupt::is_cancelled() {
        return Err(NativeExecutionError::Cancelled);
    }
    let graph_wall_millis = u64::try_from(started.elapsed().as_millis())
        .map_err(|_| protocol("analysis wave observed interval overflowed"))?
        .max(1);
    admitted
        .into_iter()
        .zip(responses)
        .map(|((identity, task_id), completion)| {
            if completion.evidence.task.id() != task_id {
                return Err(protocol(
                    "analysis completion belongs to another submitted task",
                ));
            }
            Ok(ExecutedAnalysisTask {
                identity,
                task_id,
                response: completion.response,
                attempt: completion.attempt,
                evidence: completion.evidence,
                graph_wall_millis: completion
                    .original_graph_wall_millis
                    .unwrap_or(graph_wall_millis),
                failure: completion.failure,
            })
        })
        .collect()
}

fn success_response(success: AttemptSuccess) -> ProgramExecutionResponse {
    match success {
        AttemptSuccess::Values {
            outputs,
            result_objects,
            ..
        } => ProgramExecutionResponse::from_success_outputs(outputs, result_objects),
        AttemptSuccess::Graph { graph, .. } => ProgramExecutionResponse::GraphSuccess { graph },
        AttemptSuccess::Spmd { outputs } => ProgramExecutionResponse::SpmdSuccess { outputs },
    }
}

pub(crate) fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
