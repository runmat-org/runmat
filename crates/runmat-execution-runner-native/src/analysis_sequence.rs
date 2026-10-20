use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_analysis_execution::{
    AnalysisHostWorkload, CompletedStructuralSequence, PlannedResourceUsageAttempt,
    PreparedAnalysisProgram, PreparedStructuralSolveWave, PreparedStructuralStateWave,
    StructuralSequenceCoordinator, StructuralSequenceNode, StructuralSequencePreparation,
};
use runmat_analysis_fea::execution::AnalysisTaskIdentity;
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::resource::ResourceRequest;
use runmat_execution::task::RetryPolicy;
use runmat_execution::TaskId;
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_execution_runner::AttemptSuccess;

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, planned_resource_attempt, protocol,
        AnalysisTaskBatchPolicy, AnalysisTaskDependency, AnalysisTaskPool, LocalAnalysisTaskPool,
        PreparedAnalysisTask, RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver,
};

#[derive(Clone, Debug)]
pub struct StructuralSequenceExecutionPolicy {
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

impl StructuralSequenceExecutionPolicy {
    pub(crate) fn validate(&self) -> NativeExecutionResult<()> {
        self.resources
            .validate()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))
    }
}

pub async fn execute_structural_sequence_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralSequencePreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralSequence> {
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

pub async fn execute_structural_sequence_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralSequencePreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralSequence> {
    policy.validate()?;
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralSequencePreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralSequence> {
    validate_accelerator_allocation(
        &preparation.accelerator_requests,
        &policy.resources.accelerators,
    )?;
    let mut coordinator = StructuralSequenceCoordinator::new(preparation).map_err(protocol)?;
    let mut completed_nodes = BTreeMap::new();
    while !coordinator.is_complete() {
        let solve_wave = coordinator.prepare_next_solve_wave().map_err(protocol)?;
        let solve_execution =
            execute_solve_wave(pool, &solve_wave, &policy, &mut completed_nodes).await?;
        let state_wave = coordinator
            .complete_metered_solve_wave(
                pool,
                solve_wave,
                solve_execution.responses,
                solve_execution.attempts,
            )
            .map_err(protocol)?;
        let state_execution =
            execute_state_wave(pool, &state_wave, &policy, &mut completed_nodes).await?;
        coordinator
            .complete_metered_state_wave(
                pool,
                state_wave,
                state_execution.responses,
                state_execution.planned_attempts,
            )
            .map_err(protocol)?;
    }
    coordinator.finish().map_err(protocol)
}

fn validate_accelerator_allocation(
    prepared: &[runmat_execution::resource::AcceleratorRequest],
    allocated: &[runmat_execution::resource::AcceleratorRequest],
) -> NativeExecutionResult<()> {
    if prepared != allocated {
        return Err(NativeExecutionError::Protocol(
            "structural preparation and execution policy name different accelerator requests"
                .into(),
        ));
    }
    Ok(())
}

async fn execute_solve_wave(
    pool: &mut impl AnalysisTaskPool,
    wave: &PreparedStructuralSolveWave,
    policy: &StructuralSequenceExecutionPolicy,
    completed_nodes: &mut BTreeMap<StructuralSequenceNode, (AnalysisTaskIdentity, TaskId)>,
) -> NativeExecutionResult<ExecutedWave> {
    let tasks = wave.tasks.iter().map(|task| TaskView {
        node: &task.node,
        dependencies: &task.dependencies,
        host: &task.prepared.host,
        program: &task.prepared.program,
        resource_plan: &task.prepared.resource_plan,
    });
    execute_wave(pool, tasks, policy, completed_nodes).await
}

async fn execute_state_wave(
    pool: &mut impl AnalysisTaskPool,
    wave: &PreparedStructuralStateWave,
    policy: &StructuralSequenceExecutionPolicy,
    completed_nodes: &mut BTreeMap<StructuralSequenceNode, (AnalysisTaskIdentity, TaskId)>,
) -> NativeExecutionResult<ExecutedWave> {
    let tasks = wave.tasks.iter().map(|task| TaskView {
        node: &task.node,
        dependencies: &task.dependencies,
        host: &task.prepared.host,
        program: &task.prepared.program,
        resource_plan: &task.prepared.resource_plan,
    });
    execute_wave(pool, tasks, policy, completed_nodes).await
}

async fn execute_wave<'a>(
    pool: &mut impl AnalysisTaskPool,
    tasks: impl IntoIterator<Item = TaskView<'a>>,
    policy: &StructuralSequenceExecutionPolicy,
    completed_nodes: &mut BTreeMap<StructuralSequenceNode, (AnalysisTaskIdentity, TaskId)>,
) -> NativeExecutionResult<ExecutedWave> {
    let mut prepared_tasks = Vec::new();
    let mut admitted_tasks = Vec::new();
    for task in tasks {
        let dependency_records = task
            .dependencies
            .iter()
            .map(|dependency| {
                completed_nodes.get(dependency).copied().ok_or_else(|| {
                    NativeExecutionError::Protocol(format!(
                        "sequence node {dependency:?} has not completed"
                    ))
                })
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        let dependencies = dependency_records
            .iter()
            .map(|(identity, task_id)| AnalysisTaskDependency {
                identity: *identity,
                task_id: *task_id,
            })
            .collect();
        if completed_nodes.contains_key(task.node)
            || admitted_tasks
                .iter()
                .any(|(node, _): &(StructuralSequenceNode, &FeaResourcePlan)| node == task.node)
        {
            return Err(NativeExecutionError::Protocol(
                "sequence node was submitted more than once".into(),
            ));
        }
        admitted_tasks.push((task.node.clone(), task.resource_plan));
        prepared_tasks.push(PreparedAnalysisTask {
            host: task.host,
            program: task.program,
            resource_plan: Some(task.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies,
        });
    }
    let responses = execute_analysis_task_batch(
        pool,
        prepared_tasks,
        AnalysisTaskBatchPolicy {
            retry: policy.retry,
            deadline_unix_millis: policy.deadline_unix_millis,
            priority: policy.priority,
        },
    )
    .await?;
    let mut by_node = BTreeMap::new();
    let mut attempts = BTreeMap::new();
    let mut planned_attempts = BTreeMap::new();
    for ((node, resource_plan), response) in admitted_tasks.into_iter().zip(responses) {
        let planned_attempt = planned_resource_attempt(resource_plan, &response)?;
        if by_node.insert(node.clone(), response.response).is_some() {
            return Err(NativeExecutionError::Protocol(
                "sequence wave returned a duplicate node".into(),
            ));
        }
        if let Some(attempt) = response.attempt {
            attempts.insert(node.clone(), attempt);
        }
        planned_attempts.insert(node.clone(), planned_attempt);
        completed_nodes.insert(node, (response.identity, response.task_id));
    }
    Ok(ExecutedWave {
        responses: by_node,
        attempts,
        planned_attempts,
    })
}

struct ExecutedWave {
    responses: BTreeMap<StructuralSequenceNode, ProgramExecutionResponse>,
    attempts: BTreeMap<StructuralSequenceNode, AttemptSuccess>,
    planned_attempts: BTreeMap<StructuralSequenceNode, PlannedResourceUsageAttempt>,
}

struct TaskView<'a> {
    node: &'a StructuralSequenceNode,
    dependencies: &'a [StructuralSequenceNode],
    host: &'a AnalysisHostWorkload,
    program: &'a PreparedAnalysisProgram,
    resource_plan: &'a FeaResourcePlan,
}

#[cfg(test)]
mod tests {
    use runmat_execution::resource::AcceleratorRequest;

    use super::validate_accelerator_allocation;

    fn accelerator(class: &str, memory_bytes_each: u64) -> AcceleratorRequest {
        AcceleratorRequest {
            class: runmat_execution::resource::AcceleratorClass::new(class).unwrap(),
            count: 1,
            minimum_allocation_bytes: memory_bytes_each,
            provider: None,
            required_features: Default::default(),
        }
    }

    #[test]
    fn structural_sequence_requires_the_exact_prepared_accelerator_allocation() {
        let prepared = [accelerator("cuda", 8 * 1024 * 1024 * 1024)];
        assert!(validate_accelerator_allocation(&prepared, &prepared).is_ok());
        assert!(validate_accelerator_allocation(&prepared, &[]).is_err());
        assert!(validate_accelerator_allocation(
            &prepared,
            &[accelerator("cuda", 4 * 1024 * 1024 * 1024)]
        )
        .is_err());
        assert!(validate_accelerator_allocation(
            &prepared,
            &[accelerator("metal", 8 * 1024 * 1024 * 1024)]
        )
        .is_err());
    }
}
