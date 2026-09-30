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
    let [executed] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host,
            program,
            resource_plan: Some(plan),
            fallback_resources: resources,
            dependencies,
        }],
        policy,
    )
    .await?
    .try_into()
    .map_err(|_| protocol("analysis phase returned an invalid task inventory"))?;
    let accounting = planned_resource_attempt(plan, &executed)?.accounting;
    Ok((executed, accounting))
}
