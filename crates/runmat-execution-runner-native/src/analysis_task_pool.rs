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

use crate::analysis_task::{prepare_analysis_task, AnalysisTaskPolicy};
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
mod tests;
pub(crate) use phase::{execute_governed_analysis_phase, execute_governed_analysis_task};

pub(crate) fn configure_local_analysis(
    config: &mut NativeExecutionConfig,
    resources: &ResourceRequest,
) {
    config.worker_accelerators = resources.accelerators.clone();
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
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
    ) -> NativeExecutionResult<Vec<AnalysisTaskCompletion>>;
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
    let success = delivered_attempt(&task.response, task.attempt.as_ref())?;
    success.validate_accounting().map_err(protocol)?;
    let accounting = success.resources.as_deref().cloned().ok_or_else(|| {
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

/// Registers and executes one independent task wave through the shared task substrate.
pub(crate) async fn execute_analysis_task_batch(
    pool: &mut impl AnalysisTaskPool,
    tasks: Vec<PreparedAnalysisTask<'_>>,
    policy: AnalysisTaskBatchPolicy,
) -> NativeExecutionResult<Vec<ExecutedAnalysisTask>> {
    // Optional observation/history waves can contain no admitted work. Do not submit an
    // empty native graph or manufacture scheduler attempts for either placement.
    if tasks.is_empty() {
        return Ok(Vec::new());
    }
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
            },
        )?;
        let task_id = submission.request.id;
        admitted.push((identity, task_id));
        submissions.push((task.program.request.clone(), submission));
    }
    let responses = pool.execute_batch(submissions).await?;
    if responses.len() != admitted.len() {
        return Err(NativeExecutionError::Protocol(
            "analysis task response inventory differs from its submitted wave".into(),
        ));
    }
    if runmat_runtime::interrupt::is_cancelled() {
        return Err(NativeExecutionError::Cancelled);
    }
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
            })
        })
        .collect()
}

fn success_response(success: AttemptSuccess) -> ProgramExecutionResponse {
    ProgramExecutionResponse::from_success_outputs(success.outputs, success.result_objects)
}

pub(crate) fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
