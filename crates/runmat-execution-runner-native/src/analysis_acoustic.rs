//! Whole-study acoustic placement through the shared admitted analysis task pool.
pub(crate) mod source;

use runmat_analysis_execution::AnalysisPhaseExecutionEvidence;
use runmat_analysis_execution::{
    AcousticStudyCoordinator, AcousticStudyPreparation, AnalysisHostWorkload,
    CompletedAcousticStudy, PreparedAnalysisProgram,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy};

use crate::{
    analysis_task_pool::{
        execute_analysis_task_batch, execute_governed_analysis_phase, planned_phase_execution,
        AnalysisTaskBatchPolicy, AnalysisTaskDependency, AnalysisTaskPool, ExecutedAnalysisTask,
        PreparedAnalysisTask,
    },
    NativeExecutionError, NativeExecutionResult,
};

#[derive(Clone, Debug)]
pub struct AcousticStudyExecutionPolicy {
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

impl AcousticStudyExecutionPolicy {
    fn validate(&self) -> NativeExecutionResult<()> {
        self.resources.validate().map_err(protocol)
    }
}

pub(crate) async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: AcousticStudyPreparation,
    policy: AcousticStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedAcousticStudy> {
    policy.validate()?;
    let scope_id = pool.scope_id();
    let pool_id = pool.pool_id();
    let mut coordinator = AcousticStudyCoordinator::new(preparation).map_err(protocol)?;
    let started = std::time::Instant::now();
    let mut preceding = vec![None; coordinator.entries().len()];
    while coordinator
        .entries()
        .iter()
        .any(|entry| !entry.is_complete())
    {
        validate_pool(pool, scope_id, pool_id)?;
        let mut ordinals = Vec::new();
        let count = coordinator.entries().len();
        for ordinal in 0..count {
            if coordinator.entries()[ordinal].is_complete() {
                continue;
            }
            // Observation fan-out imports retained field partitions through the pool.
            coordinator.entries_mut().map_err(protocol)?[ordinal]
                .prepare_next(&*pool)
                .map_err(protocol)?;
            ordinals.push(ordinal);
        }
        let tasks = ordinals
            .iter()
            .map(|&ordinal| {
                let (host, program, plan) = coordinator.entries()[ordinal]
                    .pending_task()
                    .map_err(protocol)?;
                Ok(PreparedAnalysisTask {
                    host,
                    program,
                    resource_plan: Some(plan),
                    prepared_closure: None,
                    fallback_resources: policy.resources.clone(),
                    dependencies: preceding[ordinal].into_iter().collect(),
                })
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        let executed = execute_analysis_task_batch(
            pool,
            tasks,
            AnalysisTaskBatchPolicy {
                retry: policy.retry,
                deadline_unix_millis: policy.deadline_unix_millis,
                priority: policy.priority,
            },
        )
        .await
        .map_err(|error| {
            let requests = ordinals
                .iter()
                .map(|&ordinal| {
                    coordinator.entries()[ordinal]
                        .pending_task()
                        .map(|(_, _, plan)| (ordinal, plan.input.phase, plan.request.clone()))
                })
                .collect::<Result<Vec<_>, _>>();
            protocol(format!(
                "acoustic entry wave failed: {error}; admitted requests: {requests:?}"
            ))
        })?;
        validate_pool(pool, scope_id, pool_id)?;
        if executed.len() != ordinals.len() {
            return Err(protocol(
                "acoustic task wave response inventory differs from submitted entries",
            ));
        }
        for (ordinal, executed) in ordinals.into_iter().zip(executed) {
            let (_, _, plan) = coordinator.entries()[ordinal]
                .pending_task()
                .map_err(protocol)?;
            let attempt = planned_phase_execution(plan, &executed)?;
            let dependency = dependency(&executed);
            coordinator
                .entry_mut(ordinal)
                .map_err(protocol)?
                .complete_next(pool, executed.response, attempt)
                .map_err(protocol)?;
            preceding[ordinal] = Some(dependency);
        }
    }
    let mut terminal = preceding
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| protocol("acoustic assembly is missing a terminal entry dependency"))?;
    // Histories span the frequencies: checkpoint and finalize them once every entry is done,
    // one wave at a time; the tasks of a wave are independent.
    loop {
        validate_pool(pool, scope_id, pool_id)?;
        let tasks = coordinator.prepare_history().map_err(protocol)?;
        if tasks.is_empty() {
            break;
        }
        let prepared = tasks
            .iter()
            .map(|task| PreparedAnalysisTask {
                host: task.host,
                program: task.program,
                resource_plan: Some(task.resource_plan),
                prepared_closure: None,
                fallback_resources: policy.resources.clone(),
                dependencies: terminal.clone(),
            })
            .collect::<Vec<_>>();
        let executed = execute_analysis_task_batch(
            pool,
            prepared,
            AnalysisTaskBatchPolicy {
                retry: policy.retry,
                deadline_unix_millis: policy.deadline_unix_millis,
                priority: policy.priority,
            },
        )
        .await
        .map_err(|error| protocol(format!("acoustic history wave failed: {error}")))?;
        if executed.len() != tasks.len() {
            return Err(protocol(
                "acoustic history wave response inventory differs from submitted tasks",
            ));
        }
        let mut wave = Vec::with_capacity(executed.len());
        let mut dependencies = Vec::with_capacity(executed.len());
        for (task, executed) in tasks.iter().zip(executed) {
            let attempt = planned_phase_execution(task.resource_plan, &executed)?;
            dependencies.push(dependency(&executed));
            wave.push((executed.response, attempt));
        }
        drop(tasks);
        coordinator.complete_history(pool, wave).map_err(protocol)?;
        terminal = dependencies;
    }
    let assembly = coordinator.prepare_assembly().map_err(protocol)?;
    validate_pool(pool, scope_id, pool_id)?;
    let (executed, accounting) = execute_one(
        pool,
        &assembly.host,
        &assembly.program,
        &assembly.resource_plan,
        terminal,
        &policy,
    )
    .await?;
    let elapsed = u64::try_from(started.elapsed().as_millis())
        .map_err(|_| protocol("acoustic task graph elapsed time overflow"))?;
    coordinator
        .complete_assembly(pool, executed.response, accounting, elapsed)
        .map_err(protocol)?;
    if pool.scope_id() != scope_id || pool.pool_id() != pool_id {
        return Err(protocol(
            "acoustic lifecycle execution pool or scope changed between phases",
        ));
    }
    coordinator.into_completed().map_err(protocol)
}

async fn execute_one(
    pool: &mut impl AnalysisTaskPool,
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    plan: &FeaResourcePlan,
    dependencies: Vec<AnalysisTaskDependency>,
    policy: &AcousticStudyExecutionPolicy,
) -> NativeExecutionResult<(ExecutedAnalysisTask, AnalysisPhaseExecutionEvidence)> {
    execute_governed_analysis_phase(
        pool,
        host,
        program,
        plan,
        dependencies,
        policy.resources.clone(),
        AnalysisTaskBatchPolicy {
            retry: policy.retry,
            deadline_unix_millis: policy.deadline_unix_millis,
            priority: policy.priority,
        },
    )
    .await
    .map_err(|error| {
        protocol(format!(
            "acoustic aggregate phase {:?} failed: {error}; admitted request: {:?}",
            plan.input.phase, plan.request,
        ))
    })
}

fn dependency(executed: &ExecutedAnalysisTask) -> AnalysisTaskDependency {
    AnalysisTaskDependency {
        identity: executed.identity,
        task_id: executed.task_id,
    }
}

fn validate_pool(
    pool: &impl AnalysisTaskPool,
    scope_id: runmat_execution::ExecutionScopeId,
    pool_id: runmat_execution::PoolId,
) -> NativeExecutionResult<()> {
    if pool.scope_id() != scope_id || pool.pool_id() != pool_id {
        return Err(protocol(
            "acoustic execution pool or scope changed between phases",
        ));
    }
    Ok(())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
