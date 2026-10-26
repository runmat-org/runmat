use std::sync::Arc;

mod phases;

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
    crate::NativeProgramObjectSource,
)> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    let mut pool = LocalAnalysisTaskPool::new(session);
    let completion = execute_with_thermal_source(&mut pool, preparation, policy, None).await?;
    Ok((completion, pool.object_source()))
}

pub async fn execute_structural_prestressed_modal_frequency_study_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedModalFrequencyStudy> {
    execute_with_thermal_source(
        &mut RemoteAnalysisTaskPool::new(pool),
        preparation,
        policy,
        None,
    )
    .await
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
    let local_execution = preparation.local_execution;
    let mesh = preparation.mesh.clone();
    let program_revision = preparation.program_revision.clone();
    let artifact_access = preparation.artifact_access.clone();
    let chunk_bytes = preparation.chunk_bytes;
    let inventory_limits = preparation.inventory_limits;
    let (mut coordinator, preload_dependencies) = coordinator(pool, preparation, thermal)?;
    crate::analysis_thermal_prestress::validate_preload_dependencies(
        &coordinator.preload_dependencies(),
        &preload_dependencies,
    )?;

    let preload_dependency = Box::pin(phases::preload(
        pool,
        &mut coordinator,
        &policy,
        preload_dependencies,
    ))
    .await?;
    let (prestress_dependency, prestress_usage) = Box::pin(phases::prestress(
        pool,
        &mut coordinator,
        &policy,
        vec![preload_dependency],
    ))
    .await?;
    let modal_dependency = Box::pin(phases::modal(
        pool,
        &mut coordinator,
        &policy,
        vec![prestress_dependency],
    ))
    .await?;
    let (basis_dependency, basis_usage) = Box::pin(phases::basis(
        pool,
        &mut coordinator,
        &policy,
        vec![modal_dependency],
    ))
    .await?;

    let bound = coordinator.bind_frequency().map_err(protocol)?;
    if bound.state_task != basis_dependency.identity {
        return Err(protocol(
            "modal-frequency binding names another basis publication task",
        ));
    }
    let resolved_workload = bound.workload.clone();
    let modal_state = bound
        .modal_state
        .ok_or_else(|| protocol("modal-frequency binding omitted its modal state"))?;
    let frequency = Box::pin(crate::analysis_frequency::execute_with_pool(
        pool,
        StructuralFrequencyExecutionPreparation {
            local_execution,
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
    ))
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
            prepared_closure: None,
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

fn coordinator(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    thermal: Option<crate::analysis_thermal_prestress::BoundThermalExecution>,
) -> NativeExecutionResult<(
    Box<StructuralPrestressedModalFrequencyStudyCoordinator>,
    Vec<AnalysisTaskDependency>,
)> {
    let (coordinator, dependencies) = match thermal {
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
    Ok((Box::new(coordinator), dependencies))
}
