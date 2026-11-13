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
    PlannedResourceUsageGraph, PreparedAnalysisTaskClosure, PreparedFluidTransientTask,
};
use runmat_execution::task::RetryPolicy;
use runmat_execution_artifact::ProgramExecutionResponse;

#[derive(Clone, Copy, Debug)]
pub struct FluidTransientBlockExecutionPolicy {
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

/// Original physical checkpoint plus complete portable task lineage and usage.
/// Missing earlier consumption blocks this result; this is not a completed fluid study.
pub struct CompletedFluidTransientBlockExecution {
    pub original: ImportedFluidTransientCheckpoint,
    pub host: AnalysisHostWorkload,
    pub response: ProgramExecutionResponse,
    pub execution: runmat_execution_runner::TaskExecutionEvidence,
    pub usage: PlannedResourceUsageGraph,
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
    let started = std::time::Instant::now();
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
    let wall_millis = u64::try_from(started.elapsed().as_millis())
        .map_err(|_| protocol("fluid block observed wall interval overflowed"))?;
    let usage = PlannedResourceUsageGraph::from_execution(
        &task.resource_plan,
        &executed.evidence,
        wall_millis,
    )
    .map_err(protocol)?;
    Ok(CompletedFluidTransientBlockExecution {
        original,
        host: task.host.clone(),
        response: executed.response,
        execution: executed.evidence,
        usage,
    })
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
