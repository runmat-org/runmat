use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use runmat_analysis_execution::{
    ImportedStructuralTransientResult, PreparedAnalysisProgram, StructuralTransientRecoveryClosure,
    StructuralTransientSequenceCoordinator, StructuralTransientSequencePreparation,
};
use runmat_analysis_fea::execution::AnalysisTaskIdentity;
use runmat_execution::TaskId;

use crate::analysis_sequence::StructuralSequenceExecutionPolicy;
use crate::analysis_task::{prepare_analysis_task, AnalysisTaskPolicy};
use crate::analysis_task_pool::{AnalysisTaskPool, RemoteAnalysisTaskPool};
use crate::{NativeExecutionError, NativeExecutionResult, RemotePoolDriver};

#[async_trait]
pub trait StructuralTransientRecoveryPublisher: Send {
    async fn publish(
        &mut self,
        closure: StructuralTransientRecoveryClosure,
    ) -> NativeExecutionResult<()>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralTransientSequence {
    pub result: ImportedStructuralTransientResult,
}

pub async fn execute_structural_transient_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralTransientSequencePreparation,
    policy: StructuralSequenceExecutionPolicy,
    recovery: Option<&StructuralTransientRecoveryClosure>,
    publisher: &mut (impl StructuralTransientRecoveryPublisher + ?Sized),
) -> NativeExecutionResult<CompletedStructuralTransientSequence> {
    policy.validate()?;
    let mut pool = RemoteAnalysisTaskPool::new(pool);
    let mut coordinator = match recovery {
        Some(recovery) => {
            for object in &recovery.objects {
                pool.register_execution_object(object.reference.clone(), object.bytes.clone())?;
            }
            StructuralTransientSequenceCoordinator::restore_from_closure(
                preparation,
                &recovery.root,
                &recovery.objects,
            )
            .map_err(protocol)?
        }
        None => StructuralTransientSequenceCoordinator::new(preparation).map_err(protocol)?,
    };
    let mut live_block_tasks = Vec::new();
    while coordinator.has_remaining_blocks() {
        let prepared = coordinator.prepare_next_block().map_err(protocol)?;
        let logical_dependencies = coordinator.next_block_dependency_identities();
        let live_dependencies = matching_live_dependencies(
            &logical_dependencies,
            live_block_tasks.last().copied().into_iter(),
        );
        let (response, identity, task_id) = execute_task(
            &mut pool,
            &prepared.host,
            &prepared.program,
            logical_dependencies,
            live_dependencies,
            &policy,
        )
        .await?;
        coordinator
            .complete_block(&pool, response, &prepared)
            .map_err(protocol)?;
        if coordinator.completed_block_task_identities().last() != Some(&identity) {
            return Err(NativeExecutionError::Protocol(
                "transient coordinator and remote task identities differ".into(),
            ));
        }
        let recovery = coordinator.recovery_closure(&pool).map_err(protocol)?;
        publisher.publish(recovery).await?;
        live_block_tasks.push((identity, task_id));
    }

    let prepared = coordinator.prepare_assembly().map_err(protocol)?;
    let logical_dependencies = coordinator.assembly_dependency_identities();
    let live_dependencies =
        matching_live_dependencies(&logical_dependencies, live_block_tasks.iter().copied());
    let (response, _, _) = execute_task(
        &mut pool,
        &prepared.host,
        &prepared.program,
        logical_dependencies,
        live_dependencies,
        &policy,
    )
    .await?;
    coordinator
        .complete_assembly(&pool, response, &prepared)
        .map_err(protocol)?;
    let result = coordinator.finish().map_err(protocol)?;
    Ok(CompletedStructuralTransientSequence { result })
}

async fn execute_task(
    pool: &mut RemoteAnalysisTaskPool,
    host: &runmat_analysis_execution::AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    dependency_identities: Vec<AnalysisTaskIdentity>,
    dependency_tasks: BTreeSet<TaskId>,
    policy: &StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<(
    runmat_execution_artifact::ProgramExecutionResponse,
    AnalysisTaskIdentity,
    TaskId,
)> {
    pool.register_program(program)?;
    let (submission, identity) = prepare_analysis_task(
        pool.scope_id(),
        pool.pool_id(),
        host,
        program,
        dependency_identities,
        dependency_tasks,
        AnalysisTaskPolicy {
            resources: policy.resources.clone(),
            resource_plan: None,
            retry: policy.retry,
            deadline_unix_millis: policy.deadline_unix_millis,
            priority: policy.priority,
        },
    )?;
    let task_id = submission.request.id;
    let mut responses = pool
        .execute_batch(
            vec![(program.request.clone(), submission)],
            policy.resources.max_wall_millis,
        )
        .await?;
    let (response, _) = responses.pop().ok_or_else(|| {
        NativeExecutionError::Protocol("transient task returned no response".into())
    })?;
    if !responses.is_empty() {
        return Err(NativeExecutionError::Protocol(
            "transient task returned more than one response".into(),
        ));
    }
    Ok((response, identity, task_id))
}

fn matching_live_dependencies(
    logical: &[AnalysisTaskIdentity],
    live: impl IntoIterator<Item = (AnalysisTaskIdentity, TaskId)>,
) -> BTreeSet<TaskId> {
    live.into_iter()
        .filter_map(|(identity, task)| logical.contains(&identity).then_some(task))
        .collect()
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_execution::Digest;

    use super::matching_live_dependencies;
    use runmat_analysis_fea::execution::{
        AnalysisTaskIdentity, AnalysisWorkKind, AnalysisWorkloadIdentity, ConstraintPlanIdentity,
        DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity,
        ProblemIdentity, SequenceIdentity, StepIdentity, StudyIdentity,
    };
    use runmat_execution::TaskId;

    fn task(label: &str) -> AnalysisTaskIdentity {
        let study = StudyIdentity::from_authored_digest(Digest::sha256(label), None).unwrap();
        let sequence = SequenceIdentity::new(Digest::sha256(format!("{label}-sequence"))).unwrap();
        let step = StepIdentity::new(sequence, Digest::sha256(format!("{label}-step"))).unwrap();
        let constraints = ConstraintPlanIdentity::new(
            step,
            Digest::sha256("mesh"),
            Digest::sha256("constraints"),
        )
        .unwrap();
        let problem =
            ProblemIdentity::new(step, Digest::sha256("model"), None, constraints, Vec::new())
                .unwrap();
        let discretization = DiscretizationIdentity::new(
            problem,
            Digest::sha256("mesh"),
            Digest::sha256("formulation"),
        )
        .unwrap();
        let operator = OperatorIdentity::new(discretization, Digest::sha256("operator")).unwrap();
        let numerical = NumericalPlanIdentity::new(operator, Digest::sha256("solver")).unwrap();
        let execution = ExecutionPlanIdentity::new(numerical, Digest::sha256("execution")).unwrap();
        let workload = AnalysisWorkloadIdentity::new(
            study,
            step,
            execution,
            AnalysisWorkKind::SolveStructuralTransientBlock,
            Digest::sha256(format!("{label}-contract")),
            Vec::new(),
        )
        .unwrap();
        AnalysisTaskIdentity::new(workload, Vec::new()).unwrap()
    }

    #[test]
    fn replacement_uses_only_dependencies_live_in_the_current_scheduler() {
        let durable = task("durable");
        let current = task("current");
        let unrelated = task("unrelated");
        let current_task = TaskId::derive(&[b"current"]);
        assert!(matching_live_dependencies(&[durable], []).is_empty());
        assert_eq!(
            matching_live_dependencies(
                &[durable, current],
                [
                    (current, current_task),
                    (unrelated, TaskId::derive(&[b"other"]))
                ],
            ),
            BTreeSet::from([current_task])
        );
    }
}
