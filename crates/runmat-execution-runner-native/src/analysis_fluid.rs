//! Native placement of the governed steady finite-volume Fluid study lifecycle.
pub(crate) mod source;
use crate::{
    analysis_task_pool::{
        execute_governed_analysis_phase, execute_governed_analysis_task, AnalysisTaskBatchPolicy,
        AnalysisTaskDependency, AnalysisTaskPool, ExecutedAnalysisTask, PreparedAnalysisTask,
    },
    NativeExecutionError, NativeExecutionResult,
};
use runmat_analysis_execution::AnalysisPhaseExecutionEvidence;
use runmat_analysis_execution::{
    AnalysisHostWorkload, FluidStudyCoordinator, FluidStudyPhase, FluidStudyPreparation,
    PlannedFluidStudyPreparation, PreparedAnalysisProgram, PreparedAnalysisTaskClosure,
    VerifiedFluidStudy,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy};

#[derive(Clone, Debug)]
pub struct FluidStudyExecutionPolicy {
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

/// Execute all governed fluid phases through the shared native task pool.
/// Retain `session.object_source()` before passing the session for cold result access.
pub async fn execute_fluid_study_locally(
    session: crate::NativeProgramSession,
    preparation: FluidStudyPreparation,
    policy: FluidStudyExecutionPolicy,
) -> NativeExecutionResult<VerifiedFluidStudy> {
    execute_with_pool(
        &mut crate::analysis_task_pool::LocalAnalysisTaskPool::new(session),
        PlannedFluidStudyPreparation::verify(preparation).map_err(protocol)?,
        policy,
    )
    .await
}

pub(crate) async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: PlannedFluidStudyPreparation,
    policy: FluidStudyExecutionPolicy,
) -> NativeExecutionResult<VerifiedFluidStudy> {
    policy.resources.validate().map_err(protocol)?;
    let scope_id = pool.scope_id();
    let pool_id = pool.pool_id();
    let output_policy = preparation.preparation().output_policy;
    let mut coordinator = FluidStudyCoordinator::planned(preparation).map_err(protocol)?;
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

    // Each authored observation phase (partition, reduction, history checkpoint, assembly,
    // history finalization) runs after its predecessor, sample by retained sample.
    while coordinator.phase() == FluidStudyPhase::ReadyForObservations {
        let Some(task) = coordinator
            .prepare_next_observation(pool)
            .map_err(|e| protocol(format!("fluid observation preparation: {e}")))?
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

    while matches!(coordinator.phase(), FluidStudyPhase::ReadyForState { .. }) {
        let state = coordinator
            .prepare_next_state()
            .map_err(|error| protocol(format!("fluid semantic state preparation: {error}")))?;
        let (executed, accounting) = execute_governed_analysis_task(
            pool,
            PreparedAnalysisTask {
                host: &state.host,
                program: &state.program,
                resource_plan: Some(&state.resource_plan),
                prepared_closure: Some(PreparedAnalysisTaskClosure {
                    inputs: &state.inputs,
                    output_policy,
                }),
                dependencies: vec![preceding],
                fallback_resources: policy.resources.clone(),
            },
            AnalysisTaskBatchPolicy {
                retry: policy.retry,
                deadline_unix_millis: policy.deadline_unix_millis,
                priority: policy.priority,
            },
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
            "fluid lifecycle execution pool or scope changed between phases",
        ));
    }
    coordinator.into_verified().map_err(protocol)
}

async fn execute_one(
    pool: &mut impl AnalysisTaskPool,
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    plan: &FeaResourcePlan,
    dependencies: Vec<AnalysisTaskDependency>,
    policy: &FluidStudyExecutionPolicy,
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
    .inspect_err(|error| {
        tracing::error!(
            work_kind = ?host.workload.kind,
            admitted_resources = ?plan.request,
            error = %error,
            "governed fluid phase failed"
        );
    })
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
