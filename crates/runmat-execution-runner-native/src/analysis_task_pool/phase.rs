//! One exact resource-governed phase; no physics or family dispatch lives here.
use super::*;

pub(crate) async fn execute_governed_analysis_phase(
    pool: &mut impl AnalysisTaskPool,
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    plan: &FeaResourcePlan,
    dependencies: Vec<AnalysisTaskDependency>,
    resources: ResourceRequest,
    policy: AnalysisTaskBatchPolicy,
) -> NativeExecutionResult<(
    ExecutedAnalysisTask,
    runmat_execution_runner::AttemptResourceAccounting,
)> {
    execute_governed_analysis_task(
        pool,
        PreparedAnalysisTask {
            host,
            program,
            resource_plan: Some(plan),
            prepared_closure: None,
            fallback_resources: resources,
            dependencies,
        },
        policy,
    )
    .await
}

pub(crate) async fn execute_governed_analysis_task(
    pool: &mut impl AnalysisTaskPool,
    task: PreparedAnalysisTask<'_>,
    policy: AnalysisTaskBatchPolicy,
) -> NativeExecutionResult<(
    ExecutedAnalysisTask,
    runmat_execution_runner::AttemptResourceAccounting,
)> {
    let plan = task
        .resource_plan
        .ok_or_else(|| protocol("governed analysis task requires an admitted resource plan"))?;
    let [executed] = execute_analysis_task_batch(pool, vec![task], policy)
        .await?
        .try_into()
        .map_err(|_| protocol("analysis phase returned an invalid task inventory"))?;
    let accounting = planned_resource_attempt(plan, &executed)?.accounting;
    Ok((executed, accounting))
}
