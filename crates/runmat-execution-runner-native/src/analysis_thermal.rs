//! Whole-study thermal placement through the shared admitted analysis task pool.
#[cfg(test)]
pub(crate) mod tests;

use runmat_analysis_core::StateId;
use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_analysis_execution::AnalysisPhaseExecutionEvidence;
use runmat_analysis_execution::{
    AnalysisHostWorkload, CompletedThermalStudy, PreparedAnalysisProgram, ThermalStudyCoordinator,
    ThermalStudyPreparation,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_governed_analysis_phase, AnalysisTaskBatchPolicy,
        AnalysisTaskDependency, AnalysisTaskPool, ExecutedAnalysisTask, LocalAnalysisTaskPool,
        RemoteAnalysisTaskPool,
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
    config: NativeExecutionConfig,
    preparation: ThermalStudyPreparation,
    policy: ThermalStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedThermalStudy> {
    execute_thermal_study_locally_with_source(config, preparation, policy)
        .await
        .map(|executed| executed.completed)
}

/// Completed scientific artifacts and a read-only lease on their declared immutable closure.
/// The source retains session storage, not the producing worker pool. Persist the published
/// closure separately before dropping it when results must survive beyond this process.
#[derive(Clone)]
pub struct CompletedThermalStudyWithSource {
    pub completed: CompletedThermalStudy,
    pub source: crate::NativeProgramObjectSource,
}

/// Execute the same admitted lifecycle while retaining its cache-backed output descendants.
/// This does not add a solver call, reconstruct lost fields, or qualify numerical restart.
pub async fn execute_thermal_study_locally_with_source(
    mut config: NativeExecutionConfig,
    preparation: ThermalStudyPreparation,
    policy: ThermalStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedThermalStudyWithSource> {
    policy.validate()?;
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    let mut pool = LocalAnalysisTaskPool::new(session);
    let executed = execute_with_pool(&mut pool, preparation, policy).await?;
    let source = pool.object_source();
    Ok(CompletedThermalStudyWithSource {
        completed: executed.completed,
        source,
    })
}

pub async fn execute_thermal_study_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: ThermalStudyPreparation,
    policy: ThermalStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedThermalStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy)
        .await
        .map(|executed| executed.completed)
}

pub(crate) struct ExecutedThermalStudy {
    pub completed: CompletedThermalStudy,
    pub state_publications: BTreeMap<StateId, AnalysisTaskDependency>,
    pub scope_id: runmat_execution::ExecutionScopeId,
    pub pool_id: runmat_execution::PoolId,
}

pub(crate) async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: ThermalStudyPreparation,
    policy: ThermalStudyExecutionPolicy,
) -> NativeExecutionResult<ExecutedThermalStudy> {
    policy.validate()?;
    let scope_id = pool.scope_id();
    let pool_id = pool.pool_id();
    let mut coordinator = ThermalStudyCoordinator::new(preparation).map_err(protocol)?;
    let state_ids: Vec<_> = coordinator
        .selected_step()
        .state_outputs
        .iter()
        .map(|declaration| declaration.state_id.clone())
        .collect();
    let mut state_publications = BTreeMap::new();
    let solve = coordinator
        .prepare_solve()
        .map_err(|e| protocol(format!("thermal solve preparation: {e}")))?;
    crate::machine_capacity::trace_footprint("thermal solve prepared");
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
            .map_err(|e| protocol(format!("thermal observation preparation: {e}")))?
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

    let accuracy = coordinator
        .prepare_accuracy()
        .map_err(|e| protocol(format!("thermal accuracy preparation: {e}")))?;
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

    for state_id in state_ids {
        let state = coordinator
            .prepare_next_state()
            .map_err(|e| protocol(format!("thermal state preparation: {e}")))?;
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
        if state_publications.insert(state_id, preceding).is_some() {
            return Err(protocol(
                "thermal lifecycle published a duplicate named state",
            ));
        }
    }

    coordinator.freeze_baseline().map_err(protocol)?;
    let publication = coordinator
        .prepare_publication()
        .map_err(|e| protocol(format!("thermal publication preparation: {e}")))?;
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
            "thermal lifecycle execution pool or scope changed between phases",
        ));
    }
    Ok(ExecutedThermalStudy {
        completed: coordinator.into_completed().map_err(protocol)?,
        state_publications,
        scope_id,
        pool_id,
    })
}

async fn execute_one(
    pool: &mut impl AnalysisTaskPool,
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    plan: &FeaResourcePlan,
    dependencies: Vec<AnalysisTaskDependency>,
    policy: &ThermalStudyExecutionPolicy,
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
