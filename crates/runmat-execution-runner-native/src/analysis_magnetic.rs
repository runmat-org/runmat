//! Native placement of the governed magnetostatic study lifecycle.
pub(crate) mod source;
use crate::{
    analysis_task_pool::{
        execute_governed_analysis_phase, AnalysisTaskBatchPolicy, AnalysisTaskDependency,
        AnalysisTaskPool, ExecutedAnalysisTask,
    },
    NativeExecutionError, NativeExecutionResult,
};
use runmat_analysis_execution::AnalysisPhaseExecutionEvidence;
use runmat_analysis_execution::{
    AnalysisHostWorkload, CompletedMagneticStudy, MagneticStudyCoordinator, MagneticStudyPhase,
    MagneticStudyPreparation, PreparedAnalysisProgram,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy};

#[derive(Clone, Debug)]
pub struct MagneticStudyExecutionPolicy {
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

pub(crate) async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: MagneticStudyPreparation,
    policy: MagneticStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedMagneticStudy> {
    policy.resources.validate().map_err(protocol)?;
    let scope_id = pool.scope_id();
    let pool_id = pool.pool_id();
    let mut coordinator = MagneticStudyCoordinator::new(preparation).map_err(protocol)?;
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

    if coordinator.phase() == MagneticStudyPhase::ReadyForObservationFanout {
        // Partition tasks share the recovery predecessor, not artificial serial scientific
        // dependencies. Placement may serialize them without changing their logical lineage.
        let fanout = coordinator
            .prepare_observation_fanout(pool)
            .map_err(protocol)?;
        let mut completed = Vec::with_capacity(fanout.partitions.len());
        for (ordinal, task) in fanout.partitions.iter().enumerate() {
            let (executed, accounting) = execute_one(
                pool,
                &task.host,
                &task.program,
                &task.resource_plan,
                vec![preceding],
                &policy,
            )
            .await?;
            completed.push((ordinal, executed, accounting));
        }
        let mut partition_dependencies = Vec::with_capacity(completed.len());
        for (ordinal, executed, accounting) in completed {
            partition_dependencies.push(dependency(&executed));
            coordinator
                .complete_observation_partition(pool, ordinal, executed.response, accounting)
                .map_err(protocol)?;
        }
        let mut reductions = Vec::new();
        while matches!(
            coordinator.phase(),
            MagneticStudyPhase::ReadyForObservationReduction { .. }
        ) {
            let reduction = coordinator
                .prepare_observation_reduction()
                .map_err(protocol)?;
            let (executed, accounting) = execute_one(
                pool,
                &reduction.host,
                &reduction.program,
                &reduction.resource_plan,
                partition_dependencies.clone(),
                &policy,
            )
            .await?;
            reductions.push(dependency(&executed));
            coordinator
                .complete_observation_reduction(pool, executed.response, accounting)
                .map_err(protocol)?;
        }
        let assembly = coordinator
            .prepare_observation_assembly()
            .map_err(protocol)?;
        let (executed, accounting) = execute_one(
            pool,
            &assembly.host,
            &assembly.program,
            &assembly.resource_plan,
            reductions,
            &policy,
        )
        .await?;
        preceding = dependency(&executed);
        coordinator
            .complete_observation_assembly(pool, executed.response, accounting)
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

    while matches!(
        coordinator.phase(),
        MagneticStudyPhase::ReadyForState { .. }
    ) {
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
            "magnetic lifecycle execution pool or scope changed between phases",
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
    policy: &MagneticStudyExecutionPolicy,
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
            "governed magnetic phase failed"
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
