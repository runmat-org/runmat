//! Whole-study thermal placement through the shared admitted analysis task pool.
#[cfg(test)]
mod tests;

use std::sync::Arc;

use runmat_analysis_execution::{
    AnalysisHostWorkload, CompletedThermalStudy, PreparedAnalysisProgram, ThermalStudyCoordinator,
    ThermalStudyPreparation,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy};
use runmat_execution_runner::AttemptResourceAccounting;

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, planned_resource_attempt,
        AnalysisTaskBatchPolicy, AnalysisTaskDependency, AnalysisTaskPool, ExecutedAnalysisTask,
        LocalAnalysisTaskPool, PreparedAnalysisTask, RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver,
};

#[derive(Clone, Debug)]
pub struct ThermalStudyExecutionPolicy {
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

impl ThermalStudyExecutionPolicy {
    fn validate(&self) -> NativeExecutionResult<()> {
        self.resources.validate().map_err(protocol)
    }
}

pub async fn execute_thermal_study_locally(
    mut config: NativeExecutionConfig,
    preparation: ThermalStudyPreparation,
    policy: ThermalStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedThermalStudy> {
    policy.validate()?;
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    execute_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
    )
    .await
}

pub async fn execute_thermal_study_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: ThermalStudyPreparation,
    policy: ThermalStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedThermalStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: ThermalStudyPreparation,
    policy: ThermalStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedThermalStudy> {
    policy.validate()?;
    let mut coordinator = ThermalStudyCoordinator::new(preparation).map_err(protocol)?;
    let state_count = coordinator.compiled_study().model.sequence.steps[0]
        .state_outputs
        .len();
    let solve = coordinator.prepare_solve().map_err(protocol)?;
    let (executed, accounting) = execute_one(
        pool,
        &solve.host,
        &solve.program,
        &solve.resource_plan,
        Vec::new(),
        &policy,
    )
    .await?;
    let mut preceding = dependency(&executed);
    coordinator
        .complete_solve(pool, executed.response, accounting)
        .map_err(protocol)?;

    while coordinator.phase() == runmat_analysis_execution::ThermalStudyPhase::ReadyForObservations
    {
        let Some(task) = coordinator
            .prepare_next_observation(pool)
            .map_err(protocol)?
        else {
            break;
        };
        let (executed, accounting) = execute_one(
            pool,
            task.host,
            task.program,
            task.resource_plan,
            vec![preceding],
            &policy,
        )
        .await?;
        preceding = dependency(&executed);
        coordinator
            .complete_observation(pool, executed.response, accounting)
            .map_err(protocol)?;
    }

    let accuracy = coordinator.prepare_accuracy().map_err(protocol)?;
    let (executed, accounting) = execute_one(
        pool,
        &accuracy.host,
        &accuracy.program,
        &accuracy.resource_plan,
        vec![preceding],
        &policy,
    )
    .await?;
    preceding = dependency(&executed);
    coordinator
        .complete_accuracy(pool, executed.response, accounting)
        .map_err(protocol)?;

    for _ in 0..state_count {
        let state = coordinator.prepare_next_state().map_err(protocol)?;
        let (executed, accounting) = execute_one(
            pool,
            &state.host,
            &state.program,
            &state.resource_plan,
            vec![preceding],
            &policy,
        )
        .await?;
        preceding = dependency(&executed);
        coordinator
            .complete_next_state(pool, executed.response, accounting)
            .map_err(protocol)?;
    }

    coordinator.freeze_baseline().map_err(protocol)?;
    let publication = coordinator.prepare_publication().map_err(protocol)?;
    let (executed, accounting) = execute_one(
        pool,
        &publication.host,
        &publication.program,
        &publication.resource_plan,
        vec![preceding],
        &policy,
    )
    .await?;
    coordinator
        .complete_publication(pool, executed.response, accounting)
        .map_err(protocol)?;
    coordinator.into_completed().map_err(protocol)
}

async fn execute_one(
    pool: &mut impl AnalysisTaskPool,
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    plan: &FeaResourcePlan,
    dependencies: Vec<AnalysisTaskDependency>,
    policy: &ThermalStudyExecutionPolicy,
) -> NativeExecutionResult<(ExecutedAnalysisTask, AttemptResourceAccounting)> {
    let [executed] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host,
            program,
            resource_plan: Some(plan),
            fallback_resources: policy.resources.clone(),
            dependencies,
        }],
        AnalysisTaskBatchPolicy {
            retry: policy.retry,
            deadline_unix_millis: policy.deadline_unix_millis,
            priority: policy.priority,
        },
    )
    .await?
    .try_into()
    .map_err(|_| protocol("thermal phase returned an invalid task inventory"))?;
    let accounting = planned_resource_attempt(plan, &executed)?.accounting;
    Ok((executed, accounting))
}

fn dependency(executed: &ExecutedAnalysisTask) -> AnalysisTaskDependency {
    AnalysisTaskDependency {
        identity: executed.identity,
        task_id: executed.task_id,
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
