use std::sync::Arc;

use runmat_analysis_execution::{
    AnalysisHostWorkload, ImportedSemanticState, ImportedStructuralLinearResult,
    ImportedStructuralModalResult, PlannedResourceUsageAttempt, PreparedAnalysisProgram,
    StructuralPrestressedFrequencyStudyPreparation,
    StructuralPrestressedModalFrequencyStudyCoordinator,
};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, planned_resource_attempt,
        AnalysisTaskBatchPolicy, AnalysisTaskDependency, AnalysisTaskPool, ExecutedAnalysisTask,
        LocalAnalysisTaskPool, PreparedAnalysisTask, RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver, StructuralFrequencyExecutionPreparation, StructuralSequenceExecutionPolicy,
};

pub struct CompletedStructuralPrestressedModalFrequencyStudy {
    pub preload: ImportedStructuralLinearResult,
    pub prestress_state: ImportedSemanticState,
    pub modal: ImportedStructuralModalResult,
    pub basis_state: ImportedSemanticState,
    pub state_publication_usage: Vec<PlannedResourceUsageAttempt>,
    pub resolved_workload: runmat_analysis_fea::execution::StructuralFrequencyWorkload,
    pub frequency: crate::CompletedStructuralFrequencyStudy,
}

pub async fn execute_structural_prestressed_modal_frequency_study_locally_with_source(
    mut config: NativeExecutionConfig,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<(
    CompletedStructuralPrestressedModalFrequencyStudy,
    crate::FilesystemObjectStore,
)> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    let mut pool = LocalAnalysisTaskPool::new(session);
    let completion = execute_with_pool(&mut pool, preparation, policy).await?;
    Ok((completion, pool.object_store()))
}

pub async fn execute_structural_prestressed_modal_frequency_study_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedModalFrequencyStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedModalFrequencyStudy> {
    execute_with_thermal_source(pool, preparation, policy, None).await
}

pub(crate) async fn execute_with_thermal_source(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
    thermal: Option<crate::analysis_thermal_prestress::BoundThermalExecution>,
) -> NativeExecutionResult<CompletedStructuralPrestressedModalFrequencyStudy> {
    policy.validate()?;
    if preparation.accelerator_requests != policy.resources.accelerators {
        return Err(protocol(
            "prestressed modal-frequency preparation and policy name different accelerator requests",
        ));
    }
    let mesh = preparation.mesh.clone();
    let program_revision = preparation.program_revision.clone();
    let artifact_access = preparation.artifact_access.clone();
    let chunk_bytes = preparation.chunk_bytes;
    let inventory_limits = preparation.inventory_limits;
    let (mut coordinator, preload_dependencies) = match thermal {
        Some(thermal) => {
            let (source, dependency) = thermal.into_parts(pool)?;
            (
                StructuralPrestressedModalFrequencyStudyCoordinator::new_with_thermal_source(
                    preparation,
                    source,
                )
                .map_err(protocol)?,
                vec![dependency],
            )
        }
        None => (
            StructuralPrestressedModalFrequencyStudyCoordinator::new(preparation)
                .map_err(protocol)?,
            Vec::new(),
        ),
    };
    crate::analysis_thermal_prestress::validate_preload_dependencies(
        &coordinator.preload_dependencies(),
        &preload_dependencies,
    )?;

    let preload = coordinator.prepare_preload().map_err(protocol)?;
    let executed_preload = execute_one(
        pool,
        &preload.host,
        &preload.program,
        &preload.resource_plan,
        &policy,
        preload_dependencies,
        "preload",
    )
    .await?;
    let preload_dependency = dependency(&executed_preload);
    coordinator
        .complete_preload(pool, executed_preload.response, &preload)
        .map_err(protocol)?;
    verify_identity(
        coordinator.preload_task_identity(),
        executed_preload.identity,
        "preload",
    )?;

    let prestress = coordinator.prepare_prestress_state().map_err(protocol)?;
    let executed_prestress = execute_one(
        pool,
        &prestress.host,
        &prestress.program,
        &prestress.resource_plan,
        &policy,
        vec![preload_dependency],
        "prestress-state publication",
    )
    .await?;
    let prestress_usage = planned_resource_attempt(&prestress.resource_plan, &executed_prestress)?;
    let prestress_dependency = dependency(&executed_prestress);
    coordinator
        .complete_prestress_state(pool, executed_prestress.response, &prestress)
        .map_err(protocol)?;
    verify_identity(
        coordinator.prestress_state_task_identity(),
        executed_prestress.identity,
        "prestress-state publication",
    )?;

    let modal = coordinator.prepare_modal().map_err(protocol)?;
    let executed_modal = execute_one(
        pool,
        &modal.host,
        &modal.program,
        &modal.resource_plan,
        &policy,
        vec![prestress_dependency],
        "prestressed-modal solve",
    )
    .await?;
    let modal_dependency = dependency(&executed_modal);
    coordinator
        .complete_modal(pool, executed_modal.response, &modal)
        .map_err(protocol)?;
    verify_identity(
        coordinator.modal_task_identity(),
        executed_modal.identity,
        "prestressed-modal solve",
    )?;

    let basis = coordinator.prepare_basis_state().map_err(protocol)?;
    let executed_basis = execute_one(
        pool,
        &basis.host,
        &basis.program,
        &basis.resource_plan,
        &policy,
        vec![modal_dependency],
        "modal-basis publication",
    )
    .await?;
    let basis_usage = planned_resource_attempt(&basis.resource_plan, &executed_basis)?;
    let basis_dependency = dependency(&executed_basis);
    coordinator
        .complete_basis_state(pool, executed_basis.response, &basis)
        .map_err(protocol)?;
    verify_identity(
        coordinator.basis_state_task_identity(),
        executed_basis.identity,
        "modal-basis publication",
    )?;

    let bound = coordinator.bind_frequency().map_err(protocol)?;
    if bound.state_task != executed_basis.identity {
        return Err(protocol(
            "modal-frequency binding names another basis publication task",
        ));
    }
    let resolved_workload = bound.workload.clone();
    let modal_state = bound
        .modal_state
        .ok_or_else(|| protocol("modal-frequency binding omitted its modal state"))?;
    let frequency = crate::analysis_frequency::execute_with_pool(
        pool,
        StructuralFrequencyExecutionPreparation {
            workload: bound.workload,
            mesh,
            prestress_state: Some(bound.prestress_state),
            modal_state: Some(modal_state),
            program_revision,
            artifact_access,
            chunk_bytes,
            inventory_limits,
            observation_limits:
                runmat_analysis_fea::results::observation::ObservationLimits::default(),
        },
        policy,
        vec![basis_dependency],
    )
    .await?;

    Ok(CompletedStructuralPrestressedModalFrequencyStudy {
        preload: coordinator
            .preload_result()
            .cloned()
            .ok_or_else(|| protocol("completed preload is absent"))?,
        prestress_state: coordinator
            .prestress_state()
            .cloned()
            .ok_or_else(|| protocol("published prestress state is absent"))?,
        modal: coordinator
            .modal_result()
            .cloned()
            .ok_or_else(|| protocol("completed prestressed-modal result is absent"))?,
        basis_state: coordinator
            .basis_state()
            .cloned()
            .ok_or_else(|| protocol("published modal-basis state is absent"))?,
        state_publication_usage: vec![prestress_usage, basis_usage],
        resolved_workload,
        frequency,
    })
}

async fn execute_one<'a>(
    pool: &mut impl AnalysisTaskPool,
    host: &'a AnalysisHostWorkload,
    program: &'a PreparedAnalysisProgram,
    resource_plan: &'a runmat_analysis_fea::planning::resource::FeaResourcePlan,
    policy: &StructuralSequenceExecutionPolicy,
    dependencies: Vec<AnalysisTaskDependency>,
    phase: &str,
) -> NativeExecutionResult<ExecutedAnalysisTask> {
    let [executed] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host,
            program,
            resource_plan: Some(resource_plan),
            fallback_resources: policy.resources.clone(),
            dependencies,
        }],
        task_policy(policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol(format!("{phase} returned an invalid task inventory")))?;
    Ok(executed)
}

fn dependency(completed: &ExecutedAnalysisTask) -> AnalysisTaskDependency {
    AnalysisTaskDependency {
        identity: completed.identity,
        task_id: completed.task_id,
    }
}

fn verify_identity(
    coordinated: Option<runmat_analysis_fea::execution::AnalysisTaskIdentity>,
    executed: runmat_analysis_fea::execution::AnalysisTaskIdentity,
    phase: &str,
) -> NativeExecutionResult<()> {
    if coordinated != Some(executed) {
        return Err(protocol(format!(
            "modal-frequency coordinator and {phase} substrate derived different identities"
        )));
    }
    Ok(())
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
