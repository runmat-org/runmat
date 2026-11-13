//! Runs a statistical step's authored observations (field recovery, sampled fan-out, ordered
//! reductions, assembly) through the shared task substrate, between assembly and accuracy.
use runmat_analysis_execution::{PlannedResourceUsageAttempt, StatisticalStudyObservations};

use crate::{
    analysis_task_pool::{
        execute_analysis_task_batch, planned_phase_execution, AnalysisTaskBatchPolicy,
        AnalysisTaskDependency, AnalysisTaskPool, ExecutedAnalysisTask, PreparedAnalysisTask,
    },
    NativeExecutionError, NativeExecutionResult, StructuralSequenceExecutionPolicy,
};

/// Returns the dependency accuracy must wait on: the observation assembly.
pub(crate) async fn execute_statistical_observations(
    pool: &mut impl AnalysisTaskPool,
    observations: &mut StatisticalStudyObservations,
    policy: &StructuralSequenceExecutionPolicy,
    assembly: AnalysisTaskDependency,
    resource_attempts: &mut Vec<PlannedResourceUsageAttempt>,
) -> NativeExecutionResult<AnalysisTaskDependency> {
    let recovery = observations.prepare_field_recovery().map_err(protocol)?;
    let [executed] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &recovery.host,
            program: &recovery.program,
            resource_plan: Some(&recovery.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![assembly],
        }],
        task_policy(policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("statistical field recovery returned an invalid inventory"))?;
    let recovery_dependency = dependency(&executed);
    record(resource_attempts, &recovery.resource_plan, &executed)?;
    observations
        .complete_field_recovery(pool, executed.response)
        .map_err(protocol)?;

    let fanout = observations
        .prepare_observation_fanout(pool)
        .map_err(protocol)?;
    let observation_count = fanout.source_request.plan.observations.len();
    let executed_partitions = execute_analysis_task_batch(
        pool,
        fanout
            .partitions
            .iter()
            .map(|task| PreparedAnalysisTask {
                host: &task.host,
                program: &task.program,
                resource_plan: Some(&task.resource_plan),
                prepared_closure: None,
                fallback_resources: policy.resources.clone(),
                dependencies: vec![recovery_dependency],
            })
            .collect(),
        task_policy(policy),
    )
    .await?;
    if executed_partitions.len() != fanout.partitions.len() {
        return Err(protocol(
            "statistical observation partitions returned an invalid inventory",
        ));
    }
    let mut partition_dependencies = Vec::with_capacity(executed_partitions.len());
    let mut partition_responses = Vec::with_capacity(executed_partitions.len());
    for (task, executed) in fanout.partitions.iter().zip(executed_partitions) {
        record(resource_attempts, &task.resource_plan, &executed)?;
        partition_dependencies.push(dependency(&executed));
        partition_responses.push(executed.response);
    }
    for (ordinal, response) in partition_responses.into_iter().enumerate() {
        observations
            .complete_observation_partition(pool, ordinal, response)
            .map_err(protocol)?;
    }

    let mut reduction_dependencies = Vec::with_capacity(observation_count);
    for _ in 0..observation_count {
        let reduction = observations
            .prepare_observation_reduction()
            .map_err(protocol)?;
        let [executed] = execute_analysis_task_batch(
            pool,
            vec![PreparedAnalysisTask {
                host: &reduction.host,
                program: &reduction.program,
                resource_plan: Some(&reduction.resource_plan),
                prepared_closure: None,
                fallback_resources: policy.resources.clone(),
                dependencies: partition_dependencies.clone(),
            }],
            task_policy(policy),
        )
        .await?
        .try_into()
        .map_err(|_| protocol("statistical reduction returned an invalid inventory"))?;
        record(resource_attempts, &reduction.resource_plan, &executed)?;
        reduction_dependencies.push(dependency(&executed));
        observations
            .complete_observation_reduction(pool, executed.response)
            .map_err(protocol)?;
    }

    let assembly = observations
        .prepare_observation_assembly()
        .map_err(protocol)?;
    let [executed] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &assembly.host,
            program: &assembly.program,
            resource_plan: Some(&assembly.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: reduction_dependencies,
        }],
        task_policy(policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("statistical observation assembly returned an invalid inventory"))?;
    record(resource_attempts, &assembly.resource_plan, &executed)?;
    let completed = dependency(&executed);
    observations
        .complete_observation_assembly(pool, executed.response)
        .map_err(protocol)?;
    Ok(completed)
}

fn record(
    attempts: &mut Vec<PlannedResourceUsageAttempt>,
    plan: &runmat_analysis_fea::planning::resource::FeaResourcePlan,
    executed: &ExecutedAnalysisTask,
) -> NativeExecutionResult<()> {
    attempts.extend(
        planned_phase_execution(plan, executed)?
            .resource_graph(plan)
            .map_err(protocol)?
            .attempts,
    );
    Ok(())
}

fn dependency(executed: &ExecutedAnalysisTask) -> AnalysisTaskDependency {
    AnalysisTaskDependency {
        identity: executed.identity,
        task_id: executed.task_id,
    }
}

fn task_policy(policy: &StructuralSequenceExecutionPolicy) -> AnalysisTaskBatchPolicy {
    AnalysisTaskBatchPolicy {
        retry: policy.retry,
        deadline_unix_millis: policy.deadline_unix_millis,
        priority: policy.priority,
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
