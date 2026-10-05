//! Native placement of one original temporal block through the shared task pool.
use crate::{
    analysis_task_pool::{
        execute_governed_analysis_task, AnalysisTaskBatchPolicy, AnalysisTaskPool,
        LocalAnalysisTaskPool, PreparedAnalysisTask,
    },
    NativeExecutionError, NativeExecutionResult, NativeProgramSession,
};
use runmat_analysis_execution::{
    import_fluid_transient_checkpoint, AnalysisHostWorkload, ImportedFluidTransientCheckpoint,
    PlannedResourceUsageAttempt, PreparedAnalysisTaskClosure, PreparedFluidTransientTask,
};
use runmat_execution::task::RetryPolicy;
use runmat_execution_artifact::ProgramExecutionResponse;

#[derive(Clone, Copy, Debug)]
pub struct FluidTransientBlockExecutionPolicy {
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

/// Original checkpoint, fenced response and scheduler accounting for the successful
/// attempt. This is not a ledger of unsuccessful attempts or a completed fluid study.
pub struct CompletedFluidTransientBlockExecution {
    pub original: ImportedFluidTransientCheckpoint,
    pub host: AnalysisHostWorkload,
    pub response: ProgramExecutionResponse,
    pub attempt: PlannedResourceUsageAttempt,
}

pub async fn execute_fluid_transient_block_locally(
    session: NativeProgramSession,
    task: &PreparedFluidTransientTask,
    policy: FluidTransientBlockExecutionPolicy,
) -> NativeExecutionResult<CompletedFluidTransientBlockExecution> {
    execute_with_pool(&mut LocalAnalysisTaskPool::new(session), task, policy).await
}

pub(crate) async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    task: &PreparedFluidTransientTask,
    policy: FluidTransientBlockExecutionPolicy,
) -> NativeExecutionResult<CompletedFluidTransientBlockExecution> {
    let scope = pool.scope_id();
    let placement = pool.pool_id();
    let (executed, accounting) = execute_governed_analysis_task(
        pool,
        PreparedAnalysisTask {
            host: &task.host,
            program: &task.program,
            resource_plan: Some(&task.resource_plan),
            prepared_closure: Some(PreparedAnalysisTaskClosure {
                inputs: &task.inputs,
                output_policy: task.output_policy,
            }),
            fallback_resources: task.resource_plan.request.clone(),
            dependencies: Vec::new(),
        },
        AnalysisTaskBatchPolicy {
            retry: policy.retry,
            deadline_unix_millis: policy.deadline_unix_millis,
            priority: policy.priority,
        },
    )
    .await?;
    if pool.scope_id() != scope || pool.pool_id() != placement {
        return Err(protocol(
            "fluid temporal block execution pool/scope changed",
        ));
    }
    runmat_analysis_execution::validate_analysis_phase_accounting(
        &task.host,
        &task.resource_plan,
        &executed.response,
        &accounting,
    )
    .map_err(protocol)?;
    let original = import_fluid_transient_checkpoint(pool, executed.response.clone(), task)
        .map_err(protocol)?;
    Ok(CompletedFluidTransientBlockExecution {
        original,
        host: task.host.clone(),
        response: executed.response,
        attempt: PlannedResourceUsageAttempt {
            plan: task.resource_plan.clone(),
            accounting,
        },
    })
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
