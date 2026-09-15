use std::sync::Arc;

use runmat_analysis_execution::{
    ImportedSemanticState, ImportedStructuralLinearResult,
    StructuralPrestressedFrequencyStudyCoordinator, StructuralPrestressedFrequencyStudyPreparation,
};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, AnalysisTaskBatchPolicy,
        AnalysisTaskDependency, AnalysisTaskPool, LocalAnalysisTaskPool, PreparedAnalysisTask,
        RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver, StructuralFrequencyExecutionPreparation, StructuralSequenceExecutionPolicy,
};

pub struct CompletedStructuralPrestressedFrequencyStudy {
    pub preload: ImportedStructuralLinearResult,
    pub prestress_state: ImportedSemanticState,
    pub frequency: crate::CompletedStructuralFrequencyStudy,
}

pub async fn execute_structural_prestressed_frequency_study_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedFrequencyStudy> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    execute_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
    )
    .await
}

pub async fn execute_structural_prestressed_frequency_study_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedFrequencyStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedFrequencyStudy> {
    policy.validate()?;
    if preparation.accelerator_requests != policy.resources.accelerators {
        return Err(protocol(
            "prestressed-frequency preparation and execution policy name different accelerator requests",
        ));
    }
    let mesh = preparation.mesh.clone();
    let program_revision = preparation.program_revision.clone();
    let artifact_access = preparation.artifact_access.clone();
    let chunk_bytes = preparation.chunk_bytes;
    let inventory_limits = preparation.inventory_limits;
    let mut coordinator =
        StructuralPrestressedFrequencyStudyCoordinator::new(preparation).map_err(protocol)?;

    let preload = coordinator.prepare_preload().map_err(protocol)?;
    let [executed_preload] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &preload.host,
            program: &preload.program,
            resource_plan: Some(&preload.resource_plan),
            fallback_resources: policy.resources.clone(),
            dependencies: Vec::new(),
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("prestressed-frequency preload returned an invalid task inventory"))?;
    coordinator
        .complete_preload(pool, executed_preload.response, &preload)
        .map_err(protocol)?;
    if coordinator.preload_task_identity() != Some(executed_preload.identity) {
        return Err(protocol(
            "prestressed-frequency coordinator and preload substrate derived different identities",
        ));
    }

    let state = coordinator.prepare_state().map_err(protocol)?;
    let [executed_state] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &state.host,
            program: &state.program,
            resource_plan: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![AnalysisTaskDependency {
                identity: executed_preload.identity,
                task_id: executed_preload.task_id,
            }],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("prestress publication returned an invalid task inventory"))?;
    coordinator
        .complete_state(pool, executed_state.response, &state)
        .map_err(protocol)?;
    if coordinator.state_task_identity() != Some(executed_state.identity) {
        return Err(protocol(
            "prestressed-frequency coordinator and state substrate derived different identities",
        ));
    }

    let bound = coordinator.bind_frequency().map_err(protocol)?;
    if bound.state_task != executed_state.identity {
        return Err(protocol(
            "prestressed-frequency binding names another state publication task",
        ));
    }
    let frequency = crate::analysis_frequency::execute_with_pool(
        pool,
        StructuralFrequencyExecutionPreparation {
            workload: bound.workload,
            mesh,
            prestress_state: Some(bound.prestress_state),
            modal_state: None,
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

    Ok(CompletedStructuralPrestressedFrequencyStudy {
        preload: coordinator
            .preload_result()
            .cloned()
            .ok_or_else(|| protocol("completed prestressed-frequency preload is absent"))?,
        prestress_state: coordinator
            .semantic_state()
            .cloned()
            .ok_or_else(|| protocol("published prestress state is absent"))?,
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
