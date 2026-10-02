use std::sync::Arc;

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, planned_resource_attempt,
        AnalysisTaskBatchPolicy, AnalysisTaskDependency, AnalysisTaskPool, LocalAnalysisTaskPool,
        PreparedAnalysisTask, RemoteAnalysisTaskPool,
    },
    analysis_transient::{
        execute_structural_transient_with_pool, CompletedStructuralTransientSequence,
        StructuralTransientRecoveryPublisher,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver, StructuralSequenceExecutionPolicy,
};
use runmat_analysis_execution::{
    ImportedSemanticState, ImportedStructuralModalResult, PlannedResourceUsageAttempt,
    StructuralModalTransientStudyCoordinator, StructuralModalTransientStudyPreparation,
    StructuralTransientRecoveryClosure,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralModalTransientStudy {
    pub modal: ImportedStructuralModalResult,
    pub basis_state: ImportedSemanticState,
    pub state_publication_usage: PlannedResourceUsageAttempt,
    pub transient: CompletedStructuralTransientSequence,
}

pub async fn execute_structural_modal_transient_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralModalTransientStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
    recovery: Option<&StructuralTransientRecoveryClosure>,
    publisher: &mut (impl StructuralTransientRecoveryPublisher + ?Sized),
) -> NativeExecutionResult<CompletedStructuralModalTransientStudy> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    execute_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
        recovery,
        publisher,
    )
    .await
}

pub async fn execute_structural_modal_transient_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralModalTransientStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
    recovery: Option<&StructuralTransientRecoveryClosure>,
    publisher: &mut (impl StructuralTransientRecoveryPublisher + ?Sized),
) -> NativeExecutionResult<CompletedStructuralModalTransientStudy> {
    execute_with_pool(
        &mut RemoteAnalysisTaskPool::new(pool),
        preparation,
        policy,
        recovery,
        publisher,
    )
    .await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralModalTransientStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
    recovery: Option<&StructuralTransientRecoveryClosure>,
    publisher: &mut (impl StructuralTransientRecoveryPublisher + ?Sized),
) -> NativeExecutionResult<CompletedStructuralModalTransientStudy> {
    policy.validate()?;
    let mut coordinator =
        StructuralModalTransientStudyCoordinator::new(preparation).map_err(protocol)?;

    let modal = coordinator.prepare_modal().map_err(protocol)?;
    let [executed_modal] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &modal.host,
            program: &modal.program,
            resource_plan: Some(&modal.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: Vec::new(),
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("modal extraction returned an invalid task inventory"))?;
    coordinator
        .complete_modal(pool, executed_modal.response, &modal)
        .map_err(protocol)?;
    let modal_dependency = AnalysisTaskDependency {
        identity: executed_modal.identity,
        task_id: executed_modal.task_id,
    };
    if coordinator.modal_task_identity() != Some(executed_modal.identity) {
        return Err(protocol(
            "modal coordinator and task substrate derived different identities",
        ));
    }

    let state = coordinator.prepare_state().map_err(protocol)?;
    let [executed_state] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &state.host,
            program: &state.program,
            resource_plan: Some(&state.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![modal_dependency],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("modal-state publication returned an invalid task inventory"))?;
    let state_publication_usage = planned_resource_attempt(&state.resource_plan, &executed_state)?;
    coordinator
        .complete_state(pool, executed_state.response, &state)
        .map_err(protocol)?;
    if coordinator.state_task_identity() != Some(executed_state.identity) {
        return Err(protocol(
            "state coordinator and task substrate derived different identities",
        ));
    }

    let transient_preparation = coordinator.prepare_transient().map_err(protocol)?;
    let transient = execute_structural_transient_with_pool(
        pool,
        transient_preparation,
        policy,
        recovery,
        publisher,
        vec![(executed_state.identity, executed_state.task_id)],
    )
    .await?;
    Ok(CompletedStructuralModalTransientStudy {
        modal: coordinator
            .modal_result()
            .cloned()
            .ok_or_else(|| protocol("completed modal result is absent"))?,
        basis_state: coordinator
            .semantic_state()
            .cloned()
            .ok_or_else(|| protocol("published modal state is absent"))?,
        state_publication_usage,
        transient,
    })
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
