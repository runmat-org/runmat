use std::sync::Arc;

use runmat_analysis_execution::{
    ImportedSemanticState, ImportedStructuralLinearResult, PlannedResourceUsageAttempt,
    StructuralPrestressedFrequencyStudyCoordinator, StructuralPrestressedFrequencyStudyPreparation,
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

pub struct CompletedStructuralPrestressedFrequencyStudy {
    pub preload: ImportedStructuralLinearResult,
    pub prestress_state: ImportedSemanticState,
    pub state_publication_usage: PlannedResourceUsageAttempt,
    pub resolved_workload: runmat_analysis_fea::execution::StructuralFrequencyWorkload,
    pub frequency: crate::CompletedStructuralFrequencyStudy,
}

pub async fn execute_structural_prestressed_frequency_study_locally(
    config: NativeExecutionConfig,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedFrequencyStudy> {
    execute_structural_prestressed_frequency_study_locally_with_source(config, preparation, policy)
        .await
        .map(|(completion, _)| completion)
}

pub async fn execute_structural_prestressed_frequency_study_locally_with_source(
    mut config: NativeExecutionConfig,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<(
    CompletedStructuralPrestressedFrequencyStudy,
    crate::NativeProgramObjectSource,
)> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    let mut pool = LocalAnalysisTaskPool::new(session);
    let completion = execute_with_pool(&mut pool, preparation, policy).await?;
    Ok((completion, pool.object_source()))
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
    execute_with_thermal_source(pool, preparation, policy, None).await
}

/// Admission and preload construction finish before the suspended execution
/// begins. Keep their large owned carriers out of every subsequent poll frame.
struct PreparedPrestressedFrequencyExecution {
    coordinator: Box<StructuralPrestressedFrequencyStudyCoordinator>,
    preload: Box<runmat_analysis_execution::PreparedStructuralLinearProgram>,
    preload_dependencies: Vec<AnalysisTaskDependency>,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    mesh: runmat_meshing_core::SolverMeshArtifact,
    program_revision: runmat_execution::ProgramRevision,
    artifact_access: runmat_execution_artifact::ArtifactAccess,
    chunk_bytes: usize,
    inventory_limits: runmat_execution_artifact::object::ObjectInventoryLimits,
}

fn prepare_execution(
    pool: &impl AnalysisTaskPool,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: &StructuralSequenceExecutionPolicy,
    thermal: Option<crate::analysis_thermal_prestress::BoundThermalExecution>,
) -> NativeExecutionResult<PreparedPrestressedFrequencyExecution> {
    policy.validate()?;
    if preparation.accelerator_requests != policy.resources.accelerators {
        return Err(protocol(
            "prestressed-frequency preparation and execution policy name different accelerator requests",
        ));
    }
    let local_execution = preparation.local_execution;
    let mesh = preparation.mesh.clone();
    let program_revision = preparation.program_revision.clone();
    let artifact_access = preparation.artifact_access.clone();
    let chunk_bytes = preparation.chunk_bytes;
    let inventory_limits = preparation.inventory_limits;
    let (coordinator, preload_dependencies) = match thermal {
        Some(thermal) => {
            let (source, dependency) = thermal.into_parts(pool)?;
            (
                StructuralPrestressedFrequencyStudyCoordinator::new_with_thermal_source(
                    preparation,
                    source,
                )
                .map_err(protocol)?,
                vec![dependency],
            )
        }
        None => (
            StructuralPrestressedFrequencyStudyCoordinator::new(preparation).map_err(protocol)?,
            Vec::new(),
        ),
    };
    let mut coordinator = Box::new(coordinator);
    crate::analysis_thermal_prestress::validate_preload_dependencies(
        &coordinator.preload_dependencies(),
        &preload_dependencies,
    )?;

    let preload = Box::new(coordinator.prepare_preload().map_err(protocol)?);
    Ok(PreparedPrestressedFrequencyExecution {
        coordinator,
        preload,
        preload_dependencies,
        local_execution,
        mesh,
        program_revision,
        artifact_access,
        chunk_bytes,
        inventory_limits,
    })
}

pub(crate) async fn execute_with_thermal_source(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
    thermal: Option<crate::analysis_thermal_prestress::BoundThermalExecution>,
) -> NativeExecutionResult<CompletedStructuralPrestressedFrequencyStudy> {
    let prepared = prepare_execution(pool, preparation, &policy, thermal)?;
    execute_prepared(pool, prepared, policy).await
}

async fn execute_prepared(
    pool: &mut impl AnalysisTaskPool,
    prepared: PreparedPrestressedFrequencyExecution,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralPrestressedFrequencyStudy> {
    let PreparedPrestressedFrequencyExecution {
        mut coordinator,
        preload,
        preload_dependencies,
        local_execution,
        mesh,
        program_revision,
        artifact_access,
        chunk_bytes,
        inventory_limits,
    } = prepared;
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
            "prestressed-frequency coordinator and state substrate derived different identities",
        ));
    }

    let bound = coordinator.bind_frequency().map_err(protocol)?;
    if bound.state_task != executed_state.identity {
        return Err(protocol(
            "prestressed-frequency binding names another state publication task",
        ));
    }
    let resolved_workload = bound.workload.clone();
    let frequency = crate::analysis_frequency::execute_with_pool(
        pool,
        StructuralFrequencyExecutionPreparation {
            local_execution,
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
