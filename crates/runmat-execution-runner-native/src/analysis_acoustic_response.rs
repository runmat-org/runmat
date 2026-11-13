//! Whole-study acoustic response placement through the shared admitted analysis task pool.
pub(crate) mod source;

use runmat_analysis_execution::AnalysisPhaseExecutionEvidence;
use runmat_analysis_execution::{
    AcousticResponseStudyCoordinator, AcousticResponseStudyPreparation, AnalysisHostWorkload,
    CompletedAcousticResponseStudy, PreparedAnalysisProgram,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy};

use crate::{
    analysis_task_pool::{
        execute_governed_analysis_phase, AnalysisTaskBatchPolicy, AnalysisTaskDependency,
        AnalysisTaskPool, ExecutedAnalysisTask,
    },
    NativeExecutionError, NativeExecutionResult,
};

#[derive(Clone, Debug)]
pub struct AcousticResponseStudyExecutionPolicy {
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

impl AcousticResponseStudyExecutionPolicy {
    fn validate(&self) -> NativeExecutionResult<()> {
        self.resources.validate().map_err(protocol)
    }
}

pub(crate) async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: AcousticResponseStudyPreparation,
    policy: AcousticResponseStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedAcousticResponseStudy> {
    policy.validate()?;
    let scope_id = pool.scope_id();
    let pool_id = pool.pool_id();
    let mut coordinator = AcousticResponseStudyCoordinator::new(preparation).map_err(protocol)?;
    let state_ids: Vec<_> = coordinator
        .selected_step()
        .state_outputs
        .iter()
        .map(|declaration| declaration.state_id.clone())
        .collect();
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

    if coordinator.phase()
        == runmat_analysis_execution::AcousticResponseStudyPhase::ReadyForFieldRecovery
    {
        let recovery = coordinator.prepare_field_recovery().map_err(protocol)?;
        let (executed, accounting) = execute_one(
            pool,
            &recovery.host,
            &recovery.program,
            &recovery.resource_plan,
            vec![preceding],
            &policy,
        )
        .await?;
        preceding = dependency(&executed);
        coordinator
            .complete_field_recovery(pool, executed.response, accounting)
            .map_err(protocol)?;
        loop {
            let Some(observation) = coordinator
                .prepare_next_observation(pool)
                .map_err(protocol)?
            else {
                break;
            };
            let (executed, accounting) = execute_one(
                pool,
                observation.host,
                observation.program,
                observation.resource_plan,
                vec![preceding],
                &policy,
            )
            .await?;
            preceding = dependency(&executed);
            coordinator
                .complete_observation(pool, executed.response, accounting)
                .map_err(protocol)?;
        }
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

    for _state_id in state_ids {
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
    if pool.scope_id() != scope_id || pool.pool_id() != pool_id {
        return Err(protocol(
            "acoustic response lifecycle execution pool or scope changed between phases",
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
    policy: &AcousticResponseStudyExecutionPolicy,
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
