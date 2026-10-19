use std::sync::Arc;

use runmat_analysis_execution::{
    ImportedSemanticState, ImportedStructuralLinearResult, ImportedStructuralModalResult,
    PlannedResourceUsageAttempt, StructuralPrestressedModalStudyCoordinator,
    StructuralPrestressedModalStudyPreparation,
};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, planned_resource_attempt,
        AnalysisTaskBatchPolicy, AnalysisTaskDependency, AnalysisTaskPool, LocalAnalysisTaskPool,
        PreparedAnalysisTask, RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver, StructuralSequenceExecutionPolicy,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralPrestressedModalStudy {
    pub preload: ImportedStructuralLinearResult,
    pub prestress_state: ImportedSemanticState,
    pub state_publication_usage: PlannedResourceUsageAttempt,
    pub modal: ImportedStructuralModalResult,
}

pub async fn execute_structural_prestressed_modal_study_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralPrestressedModalStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedModalStudy> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    execute_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
    )
    .await
}

pub async fn execute_structural_prestressed_modal_study_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralPrestressedModalStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedModalStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralPrestressedModalStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedModalStudy> {
    execute_with_thermal_source(pool, preparation, policy, None).await
}

pub(crate) async fn execute_with_thermal_source(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralPrestressedModalStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
    thermal: Option<crate::analysis_thermal_prestress::BoundThermalExecution>,
) -> NativeExecutionResult<CompletedStructuralPrestressedModalStudy> {
    policy.validate()?;
    if preparation.accelerator_requests != policy.resources.accelerators {
        return Err(protocol(
            "prestressed-modal preparation and execution policy name different accelerator requests",
        ));
    }
    let (mut coordinator, preload_dependencies) = match thermal {
        Some(thermal) => {
            let (source, dependency) = thermal.into_parts(pool)?;
            (
                StructuralPrestressedModalStudyCoordinator::new_with_thermal_source(
                    preparation,
                    source,
                )
                .map_err(protocol)?,
                vec![dependency],
            )
        }
        None => (
            StructuralPrestressedModalStudyCoordinator::new(preparation).map_err(protocol)?,
            Vec::new(),
        ),
    };
    crate::analysis_thermal_prestress::validate_preload_dependencies(
        &coordinator.preload_dependencies(),
        &preload_dependencies,
    )?;

    let preload = coordinator.prepare_preload().map_err(protocol)?;
    let [executed_preload] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &preload.host,
            program: &preload.program,
            resource_plan: Some(&preload.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: preload_dependencies,
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("prestressed-modal preload returned an invalid task inventory"))?;
    coordinator
        .complete_preload(pool, executed_preload.response, &preload)
        .map_err(protocol)?;
    if coordinator.preload_task_identity() != Some(executed_preload.identity) {
        return Err(protocol(
            "prestressed-modal coordinator and preload substrate derived different identities",
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
                identity: executed_preload.identity,
                task_id: executed_preload.task_id,
            }],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("prestress publication returned an invalid task inventory"))?;
    let state_publication_usage = planned_resource_attempt(&state.resource_plan, &executed_state)?;
    coordinator
        .complete_state(pool, executed_state.response, &state)
        .map_err(protocol)?;
    if coordinator.state_task_identity() != Some(executed_state.identity) {
        return Err(protocol(
            "prestressed-modal coordinator and state substrate derived different identities",
        ));
    }

    let modal = coordinator.prepare_modal().map_err(protocol)?;
    let [executed_modal] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &modal.host,
            program: &modal.program,
            resource_plan: Some(&modal.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![AnalysisTaskDependency {
                identity: executed_state.identity,
                task_id: executed_state.task_id,
            }],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("prestressed-modal solve returned an invalid task inventory"))?;
    coordinator
        .complete_modal(pool, executed_modal.response, &modal)
        .map_err(protocol)?;
    if coordinator.modal_task_identity() != Some(executed_modal.identity) {
        return Err(protocol(
            "prestressed-modal coordinator and modal substrate derived different identities",
        ));
    }

    Ok(CompletedStructuralPrestressedModalStudy {
        preload: coordinator
            .preload_result()
            .cloned()
            .ok_or_else(|| protocol("completed prestressed-modal preload is absent"))?,
        prestress_state: coordinator
            .semantic_state()
            .cloned()
            .ok_or_else(|| protocol("published prestress state is absent"))?,
        state_publication_usage,
        modal: coordinator
            .modal_result()
            .cloned()
            .ok_or_else(|| protocol("completed prestressed-modal result is absent"))?,
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
