use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use runmat_analysis_execution::{
    complete_structural_transient_sequence_preparation, import_structural_transient_scope,
    prepare_structural_transient_scope_task, ImportedStructuralTransientResult,
    PlannedResourceUsageAttempt, PreparedAnalysisProgram, StructuralTransientRecoveryClosure,
    StructuralTransientScopeTaskPreparation, StructuralTransientSequenceCoordinator,
    StructuralTransientSequencePlan,
};
use runmat_analysis_fea::execution::AnalysisTaskIdentity;
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::TaskId;

use crate::analysis_sequence::StructuralSequenceExecutionPolicy;
use crate::analysis_task::{analysis_task_id, prepare_analysis_task, AnalysisTaskPolicy};
use crate::analysis_task_pool::{
    begin_analysis_batch, configure_local_analysis, planned_resource_attempt, AnalysisTaskPool,
    ExecutedAnalysisTask, LocalAnalysisTaskPool, RemoteAnalysisTaskPool,
};
use crate::{
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver,
};

#[async_trait]
pub trait StructuralTransientRecoveryPublisher: Send {
    async fn publish(
        &mut self,
        closure: StructuralTransientRecoveryClosure,
    ) -> NativeExecutionResult<()>;
}

#[derive(Default)]
pub struct RetainedStructuralTransientRecovery {
    latest: Option<StructuralTransientRecoveryClosure>,
}

impl RetainedStructuralTransientRecovery {
    pub fn latest(&self) -> Option<&StructuralTransientRecoveryClosure> {
        self.latest.as_ref()
    }
}

#[async_trait]
impl StructuralTransientRecoveryPublisher for RetainedStructuralTransientRecovery {
    async fn publish(
        &mut self,
        closure: StructuralTransientRecoveryClosure,
    ) -> NativeExecutionResult<()> {
        self.latest = Some(closure);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralTransientSequence {
    pub result: ImportedStructuralTransientResult,
    pub resource_attempts: Vec<PlannedResourceUsageAttempt>,
}

pub async fn execute_structural_transient_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralTransientSequencePlan,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralTransientSequence> {
    policy.validate()?;
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    let mut publisher = RetainedStructuralTransientRecovery::default();
    execute_structural_transient_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
        None,
        &mut publisher,
        Vec::new(),
    )
    .await
}

pub async fn execute_structural_transient_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralTransientSequencePlan,
    policy: StructuralSequenceExecutionPolicy,
    recovery: Option<&StructuralTransientRecoveryClosure>,
    publisher: &mut (impl StructuralTransientRecoveryPublisher + ?Sized),
) -> NativeExecutionResult<CompletedStructuralTransientSequence> {
    policy.validate()?;
    let mut pool = RemoteAnalysisTaskPool::new(pool);
    execute_structural_transient_with_pool(
        &mut pool,
        preparation,
        policy,
        recovery,
        publisher,
        Vec::new(),
    )
    .await
}

pub(crate) async fn execute_structural_transient_with_pool(
    pool: &mut impl AnalysisTaskPool,
    plan: StructuralTransientSequencePlan,
    policy: StructuralSequenceExecutionPolicy,
    recovery: Option<&StructuralTransientRecoveryClosure>,
    publisher: &mut (impl StructuralTransientRecoveryPublisher + ?Sized),
    initial_live_tasks: Vec<(AnalysisTaskIdentity, TaskId)>,
) -> NativeExecutionResult<CompletedStructuralTransientSequence> {
    policy.validate()?;
    let prepared_scope =
        prepare_structural_transient_scope_task(StructuralTransientScopeTaskPreparation {
            local_execution: plan.local_execution,
            workload: plan.workload.clone(),
            mesh: plan.mesh.clone(),
            modal_state: plan.modal_state.clone(),
            artifact_access: plan.artifact_access.clone(),
            program_revision: plan.program_revision.clone(),
            chunk_bytes: plan.chunk_bytes,
            inventory_limits: plan.inventory_limits,
        })
        .map_err(protocol)?;
    let scope_dependencies = matching_live_dependencies(
        &plan.initial_dependencies,
        initial_live_tasks.iter().copied(),
    );
    let (scope_response, scope_identity, scope_task_id, scope_attempt) = execute_task(
        pool,
        &prepared_scope.host,
        &prepared_scope.program,
        plan.initial_dependencies.clone(),
        scope_dependencies,
        Some(&prepared_scope.resource_plan),
        &policy,
    )
    .await?;
    let mut resource_attempts = vec![scope_attempt.ok_or_else(|| {
        NativeExecutionError::Protocol("transient scope task lost its planned attempt".into())
    })?];
    let imported_scope = import_structural_transient_scope(
        pool,
        scope_response,
        &prepared_scope,
        plan.inventory_limits,
    )
    .map_err(protocol)?;
    let preparation = complete_structural_transient_sequence_preparation(
        plan,
        prepared_scope,
        imported_scope,
        scope_identity,
    )
    .map_err(protocol)?;
    let mut coordinator = match recovery {
        Some(recovery) => {
            for object in &recovery.objects {
                pool.register_recovery_object(object)?;
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
    let restored_blocks = coordinator.completed_block_task_identities();
    let candidates = restored_blocks
        .iter()
        .map(|identity| analysis_task_id(*identity))
        .collect::<BTreeSet<_>>();
    let retained = if candidates.is_empty() {
        BTreeSet::new()
    } else {
        pool.retained_committed_tasks(&candidates)
    };
    let mut live_block_tasks = initial_live_tasks;
    live_block_tasks.push((scope_identity, scope_task_id));
    live_block_tasks.extend(restored_retained_tasks(restored_blocks, &retained));
    while coordinator.has_remaining_blocks() {
        let prepared = coordinator.prepare_next_block().map_err(protocol)?;
        let logical_dependencies = coordinator.next_block_dependency_identities();
        let live_dependencies =
            matching_live_dependencies(&logical_dependencies, live_block_tasks.iter().copied());
        let (response, identity, task_id, block_attempt) = execute_task(
            pool,
            &prepared.host,
            &prepared.program,
            logical_dependencies,
            live_dependencies,
            Some(&prepared.resource_plan),
            &policy,
        )
        .await?;
        resource_attempts.push(block_attempt.ok_or_else(|| {
            NativeExecutionError::Protocol("transient block task lost its planned attempt".into())
        })?);
        coordinator
            .complete_block(pool, response, &prepared)
            .map_err(protocol)?;
        if coordinator.completed_block_task_identities().last() != Some(&identity) {
            return Err(NativeExecutionError::Protocol(
                "transient coordinator and remote task identities differ".into(),
            ));
        }
        let recovery = coordinator.recovery_closure(pool).map_err(protocol)?;
        publisher.publish(recovery).await?;
        live_block_tasks.push((identity, task_id));
    }

    let prepared = coordinator.prepare_assembly().map_err(protocol)?;
    let logical_dependencies = coordinator.assembly_dependency_identities();
    let live_dependencies =
        matching_live_dependencies(&logical_dependencies, live_block_tasks.iter().copied());
    let (response, _, _, assembly_attempt) = execute_task(
        pool,
        &prepared.host,
        &prepared.program,
        logical_dependencies,
        live_dependencies,
        Some(&prepared.resource_plan),
        &policy,
    )
    .await?;
    resource_attempts.push(assembly_attempt.ok_or_else(|| {
        NativeExecutionError::Protocol("transient assembly task lost its planned attempt".into())
    })?);
    coordinator
        .complete_assembly(pool, response, &prepared)
        .map_err(protocol)?;
    let result = coordinator.finish().map_err(protocol)?;
    Ok(CompletedStructuralTransientSequence {
        result,
        resource_attempts,
    })
}

async fn execute_task(
    pool: &mut impl AnalysisTaskPool,
    host: &runmat_analysis_execution::AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    dependency_identities: Vec<AnalysisTaskIdentity>,
    dependency_tasks: BTreeSet<TaskId>,
    resource_plan: Option<&FeaResourcePlan>,
    policy: &StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<(
    runmat_execution_artifact::ProgramExecutionResponse,
    AnalysisTaskIdentity,
    TaskId,
    Option<PlannedResourceUsageAttempt>,
)> {
    let started = std::time::Instant::now();
    let expected_identity =
        AnalysisTaskIdentity::new(host.workload.identity, dependency_identities.clone())
            .map_err(protocol)?;
    let expected_task_id = analysis_task_id(expected_identity);
    let admission = begin_analysis_batch(pool, &[expected_task_id]).await?;
    pool.register_program(program)?;
    let resources = resource_plan
        .map(|plan| plan.request.clone())
        .unwrap_or_else(|| policy.resources.clone());
    let (submission, identity) = prepare_analysis_task(
        pool.scope_id(),
        pool.pool_id(),
        host,
        program,
        dependency_identities,
        dependency_tasks,
        AnalysisTaskPolicy {
            resources: resources.clone(),
            resource_plan,
            prepared_closure: None,
            retry: policy.retry,
            deadline_unix_millis: policy.deadline_unix_millis,
            priority: policy.priority,
            permitted_trust_tiers: pool.permitted_trust_tiers(),
        },
    )?;
    let task_id = submission.request.id;
    if identity != expected_identity || task_id != expected_task_id {
        return Err(protocol(
            "transient task changed identity after its observed wave began",
        ));
    }
    let mut responses = pool
        .execute_batch(admission.admit(vec![(program.request.clone(), submission)])?)
        .await?;
    let completion = responses.pop().ok_or_else(|| {
        NativeExecutionError::Protocol("transient task returned no response".into())
    })?;
    if !responses.is_empty() {
        return Err(NativeExecutionError::Protocol(
            "transient task returned more than one response".into(),
        ));
    }
    if completion.evidence.task.id() != task_id {
        return Err(protocol(
            "transient completion belongs to another submitted task",
        ));
    }
    if runmat_runtime::interrupt::is_cancelled() {
        return Err(NativeExecutionError::Cancelled);
    }
    let graph_wall_millis = completion.original_graph_wall_millis.unwrap_or(
        u64::try_from(started.elapsed().as_millis().max(1))
            .map_err(|_| protocol("transient execution interval overflow"))?,
    );
    let executed = ExecutedAnalysisTask {
        identity,
        task_id,
        response: completion.response,
        attempt: completion.attempt,
        evidence: completion.evidence,
        failure: completion.failure,
        graph_wall_millis,
    };
    let planned_attempt = resource_plan
        .map(|plan| planned_resource_attempt(plan, &executed))
        .transpose()?;
    Ok((executed.response, identity, task_id, planned_attempt))
}

fn restored_retained_tasks(
    identities: &[AnalysisTaskIdentity],
    retained: &BTreeSet<TaskId>,
) -> Vec<(AnalysisTaskIdentity, TaskId)> {
    identities
        .iter()
        .filter_map(|identity| {
            let id = analysis_task_id(*identity);
            retained.contains(&id).then_some((*identity, id))
        })
        .collect()
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

    use super::{matching_live_dependencies, restored_retained_tasks};
    use crate::analysis_task::analysis_task_id;
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
    fn replacement_uses_only_retained_committed_or_current_scheduler_dependencies() {
        let durable = task("durable");
        let current = task("current");
        let unrelated = task("unrelated");
        let current_task = TaskId::derive(&[b"current"]);
        let durable_task = analysis_task_id(durable);
        // Cold externalized prefixes retain semantic state but no old scheduler edge.
        assert!(restored_retained_tasks(&[durable], &BTreeSet::new()).is_empty());
        let restored = restored_retained_tasks(&[durable], &BTreeSet::from([durable_task]));
        assert_eq!(restored, vec![(durable, durable_task)]);
        assert_eq!(
            matching_live_dependencies(
                &[durable, current],
                restored.into_iter().chain([
                    (current, current_task),
                    (unrelated, TaskId::derive(&[b"other"])),
                ]),
            ),
            BTreeSet::from([durable_task, current_task])
        );
    }
}
