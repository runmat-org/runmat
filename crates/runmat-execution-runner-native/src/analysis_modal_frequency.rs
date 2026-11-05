use std::sync::Arc;

use runmat_analysis_execution::{
    ImportedSemanticState, ImportedStructuralModalResult, PlannedResourceUsageAttempt,
    StructuralModalFrequencyStudyCoordinator, StructuralModalFrequencyStudyPreparation,
};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, planned_resource_attempt,
        AnalysisTaskBatchPolicy, AnalysisTaskDependency, AnalysisTaskPool, LocalAnalysisTaskPool,
        PreparedAnalysisTask, RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver, StructuralFrequencyExecutionPreparation, StructuralSequenceExecutionPolicy,
};

pub struct CompletedStructuralModalFrequencyStudy {
    pub modal: ImportedStructuralModalResult,
    pub basis_state: ImportedSemanticState,
    pub state_publication_usage: PlannedResourceUsageAttempt,
    pub resolved_workload: runmat_analysis_fea::execution::StructuralFrequencyWorkload,
    pub frequency: crate::CompletedStructuralFrequencyStudy,
}

/// Runs the study locally and returns the session's object source, from which the frequency
/// publication's original objects are read when the study is published.
pub async fn execute_structural_modal_frequency_study_locally_with_source(
    mut config: NativeExecutionConfig,
    preparation: StructuralModalFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<(
    CompletedStructuralModalFrequencyStudy,
    crate::NativeProgramObjectSource,
)> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    let mut pool = LocalAnalysisTaskPool::new(session);
    let completion = execute_with_pool(&mut pool, preparation, policy).await?;
    Ok((completion, pool.object_source()))
}

pub async fn execute_structural_modal_frequency_study_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralModalFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralModalFrequencyStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralModalFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralModalFrequencyStudy> {
    policy.validate()?;
    if preparation.accelerator_requests != policy.resources.accelerators {
        return Err(protocol(
            "modal-frequency preparation and execution policy name different accelerator requests",
        ));
    }
    let local_execution = preparation.local_execution;
    let mesh = preparation.mesh.clone();
    let program_revision = preparation.program_revision.clone();
    let artifact_access = preparation.artifact_access.clone();
    let chunk_bytes = preparation.chunk_bytes;
    let inventory_limits = preparation.inventory_limits;
    let mut coordinator =
        StructuralModalFrequencyStudyCoordinator::new(preparation).map_err(protocol)?;

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
    if coordinator.modal_task_identity() != Some(executed_modal.identity) {
        return Err(protocol(
            "modal-frequency coordinator and task substrate derived different modal identities",
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
            dependencies: vec![AnalysisTaskDependency {
                identity: executed_modal.identity,
                task_id: executed_modal.task_id,
            }],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("modal basis publication returned an invalid task inventory"))?;
    let state_publication_usage = planned_resource_attempt(&state.resource_plan, &executed_state)?;
    coordinator
        .complete_state(pool, executed_state.response, &state)
        .map_err(protocol)?;
    if coordinator.state_task_identity() != Some(executed_state.identity) {
        return Err(protocol(
            "modal-frequency coordinator and task substrate derived different state identities",
        ));
    }

    let bound = coordinator.bind_frequency().map_err(protocol)?;
    if bound.state_task != executed_state.identity {
        return Err(protocol(
            "modal-frequency binding names another state publication task",
        ));
    }
    let resolved_workload = bound.workload.clone();
    let frequency = crate::analysis_frequency::execute_with_pool(
        pool,
        StructuralFrequencyExecutionPreparation {
            local_execution,
            workload: bound.workload,
            mesh,
            prestress_state: None,
            modal_state: Some(bound.modal_state),
            program_revision,
            artifact_access,
            chunk_bytes,
            inventory_limits,
            observation_limits:
                runmat_analysis_fea::results::observation::ObservationLimits::default(),
        },
        policy,
        vec![AnalysisTaskDependency {
            identity: executed_state.identity,
            task_id: executed_state.task_id,
        }],
    )
    .await?;

    Ok(CompletedStructuralModalFrequencyStudy {
        modal: coordinator
            .modal_result()
            .cloned()
            .ok_or_else(|| protocol("completed modal result is absent"))?,
        basis_state: coordinator
            .semantic_state()
            .cloned()
            .ok_or_else(|| protocol("published modal state is absent"))?,
        state_publication_usage,
        resolved_workload,
        frequency,
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
