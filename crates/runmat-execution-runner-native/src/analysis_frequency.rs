use std::sync::Arc;
use std::time::Instant;

use runmat_analysis_execution::{
    import_structural_frequency_entry, import_structural_frequency_recovered_fields,
    import_structural_frequency_result, prepare_analysis_objects,
    prepare_planned_resource_graph_usage, prepare_structural_frequency_fanout,
    prepare_structural_frequency_field_recovery,
    prepare_structural_frequency_result_assembly_request_objects,
    prepare_structural_frequency_result_task, prepare_structural_frequency_workload_objects,
    ImportedRecoveredFields, ImportedStructuralFrequencyEntry, ImportedStructuralFrequencyResult,
    PlannedResourceUsageGraph, PreparedResourceUsage, StructuralFrequencyFanoutPreparation,
    StructuralFrequencyFieldRecoveryPreparation, StructuralFrequencyResultPreparation,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, CanonicalSemanticState, SequenceIdentity,
    StructuralFrequencyPublicationHistory, StructuralFrequencyPublicationInput,
    StructuralFrequencyPublicationManifest, StructuralFrequencyPublicationSample,
    StructuralFrequencyResultAssemblyRequest, StructuralFrequencySolverPlan,
    StructuralFrequencyWorkload,
};
use runmat_analysis_fea::results::observation::ObservationLimits;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::analysis_frequency_observations::{
    execute_structural_frequency_observations, CompletedStructuralFrequencyHistory,
    CompletedStructuralFrequencyObservationSample, StructuralFrequencyObservationExecution,
};
use crate::analysis_sequence::StructuralSequenceExecutionPolicy;
use crate::analysis_task_pool::{
    configure_local_analysis, execute_analysis_task_batch, protocol, AnalysisTaskBatchPolicy,
    AnalysisTaskDependency, AnalysisTaskPool, ExecutedAnalysisTask, LocalAnalysisTaskPool,
    PreparedAnalysisTask, RemoteAnalysisTaskPool,
};
use crate::{
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver,
};

mod publication;

pub struct StructuralFrequencyExecutionPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub workload: StructuralFrequencyWorkload,
    pub mesh: SolverMeshArtifact,
    pub prestress_state: Option<CanonicalSemanticState>,
    pub modal_state: Option<CanonicalSemanticState>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub observation_limits: ObservationLimits,
}

pub struct CompletedStructuralFrequencyEntry {
    pub entry: ImportedStructuralFrequencyEntry,
    pub recovered_fields: ImportedRecoveredFields,
}

pub struct CompletedStructuralFrequencyStudy {
    pub result: ImportedStructuralFrequencyResult,
    pub entries: Vec<CompletedStructuralFrequencyEntry>,
    pub observation_samples: Vec<CompletedStructuralFrequencyObservationSample>,
    pub histories: Vec<CompletedStructuralFrequencyHistory>,
    pub resource_usage: PreparedResourceUsage,
    pub publication: StructuralFrequencyPublicationManifest,
}

pub async fn execute_structural_frequency_locally(
    config: NativeExecutionConfig,
    preparation: StructuralFrequencyExecutionPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralFrequencyStudy> {
    execute_structural_frequency_locally_with_source(config, preparation, policy)
        .await
        .map(|(completion, _)| completion)
}

pub async fn execute_structural_frequency_locally_with_source(
    mut config: NativeExecutionConfig,
    preparation: StructuralFrequencyExecutionPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<(
    CompletedStructuralFrequencyStudy,
    crate::NativeProgramObjectSource,
)> {
    policy.validate()?;
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    let mut pool = LocalAnalysisTaskPool::new(session);
    let completion = execute_with_pool(&mut pool, preparation, policy, Vec::new()).await?;
    Ok((completion, pool.object_source()))
}

pub async fn execute_structural_frequency_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralFrequencyExecutionPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralFrequencyStudy> {
    policy.validate()?;
    execute_with_pool(
        &mut RemoteAnalysisTaskPool::new(pool),
        preparation,
        policy,
        Vec::new(),
    )
    .await
}

pub(crate) async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralFrequencyExecutionPreparation,
    policy: StructuralSequenceExecutionPolicy,
    initial_dependencies: Vec<AnalysisTaskDependency>,
) -> NativeExecutionResult<CompletedStructuralFrequencyStudy> {
    preparation.workload.validate().map_err(protocol)?;
    preparation.mesh.validate_canonical().map_err(protocol)?;
    match (
        preparation.workload.prestress_transition(),
        preparation.prestress_state.as_ref(),
    ) {
        (None, None) => {}
        (Some(transition), Some(state)) if transition.source_state == state.identity => {}
        _ => {
            return Err(NativeExecutionError::Protocol(
                "structural frequency execution has the wrong prestress-state inventory".into(),
            ))
        }
    }
    match (
        preparation.workload.solver(),
        preparation.modal_state.as_ref(),
    ) {
        (StructuralFrequencySolverPlan::Direct { .. }, None) => {}
        (StructuralFrequencySolverPlan::ModalReduced { basis_state }, Some(state))
            if *basis_state == state.identity => {}
        _ => {
            return Err(NativeExecutionError::Protocol(
                "structural frequency execution has the wrong modal-state inventory".into(),
            ))
        }
    }
    let started = Instant::now();
    let workload_objects = prepare_structural_frequency_workload_objects(
        &preparation.workload,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )
    .map_err(protocol)?;
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*preparation.mesh.canonical_digest.bytes()),
        &preparation.mesh.canonical_encode().map_err(protocol)?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )
    .map_err(protocol)?;
    let modal_state_objects = preparation
        .modal_state
        .as_ref()
        .map(|state| {
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::SemanticState,
                state.identity.digest(),
                &state.canonical_bytes()?,
                preparation.chunk_bytes,
                ObjectNamespace::InputValue,
                preparation.inventory_limits,
            )
        })
        .transpose()
        .map_err(protocol)?;
    let prestress_state_objects = preparation
        .prestress_state
        .as_ref()
        .map(|state| {
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::SemanticState,
                state.identity.digest(),
                &state.canonical_bytes()?,
                preparation.chunk_bytes,
                ObjectNamespace::InputValue,
                preparation.inventory_limits,
            )
        })
        .transpose()
        .map_err(protocol)?;
    let entry_tasks = prepare_structural_frequency_fanout(StructuralFrequencyFanoutPreparation {
        local_execution: preparation.local_execution,
        workload: workload_objects.clone(),
        mesh: mesh_objects.clone(),
        prestress_state: prestress_state_objects.clone(),
        modal_state: modal_state_objects,
        artifact_access: preparation.artifact_access.clone(),
        program_revision: preparation.program_revision.clone(),
        chunk_bytes: preparation.chunk_bytes,
        inventory_limits: preparation.inventory_limits,
    })
    .map_err(protocol)?;
    let executed_entries = execute_analysis_task_batch(
        pool,
        entry_tasks
            .iter()
            .map(|task| PreparedAnalysisTask {
                host: &task.host,
                program: &task.program,
                resource_plan: Some(&task.resource_plan),
                prepared_closure: None,
                fallback_resources: policy.resources.clone(),
                dependencies: initial_dependencies.clone(),
            })
            .collect(),
        batch_policy(&policy),
    )
    .await?;
    let mut resource_attempts = Vec::new();
    let mut imported_entries = Vec::with_capacity(entry_tasks.len());
    let mut entry_dependencies = Vec::with_capacity(entry_tasks.len());
    for (task, executed) in entry_tasks.iter().zip(executed_entries) {
        resource_attempts.push(crate::analysis_task_pool::planned_resource_attempt(
            &task.resource_plan,
            &executed,
        )?);
        entry_dependencies.push(task_dependency(&executed));
        imported_entries.push(
            import_structural_frequency_entry(
                pool,
                executed.response,
                task,
                preparation.inventory_limits,
            )
            .map_err(protocol)?,
        );
    }

    let recovery_tasks = entry_tasks
        .iter()
        .zip(&imported_entries)
        .map(|(task, imported)| {
            prepare_structural_frequency_field_recovery(
                StructuralFrequencyFieldRecoveryPreparation {
                    local_execution: preparation.local_execution,
                    source_host: task.host.clone(),
                    workload: workload_objects.clone(),
                    mesh: mesh_objects.clone(),
                    prestress_state: prestress_state_objects.clone(),
                    source_entry: imported.objects.clone(),
                    artifact_access: preparation.artifact_access.clone(),
                    program_revision: preparation.program_revision.clone(),
                    chunk_bytes: preparation.chunk_bytes,
                    inventory_limits: preparation.inventory_limits,
                },
            )
            .map_err(protocol)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let executed_recoveries = execute_analysis_task_batch(
        pool,
        recovery_tasks
            .iter()
            .zip(&entry_dependencies)
            .map(|(task, dependency)| PreparedAnalysisTask {
                host: &task.host,
                program: &task.program,
                resource_plan: Some(&task.resource_plan),
                prepared_closure: None,
                fallback_resources: policy.resources.clone(),
                dependencies: vec![*dependency],
            })
            .collect(),
        batch_policy(&policy),
    )
    .await?;
    let mut completed_entries = Vec::with_capacity(imported_entries.len());
    let mut recovery_dependencies = Vec::with_capacity(recovery_tasks.len());
    for ((entry, task), executed) in imported_entries
        .into_iter()
        .zip(&recovery_tasks)
        .zip(executed_recoveries)
    {
        resource_attempts.push(crate::analysis_task_pool::planned_resource_attempt(
            &task.resource_plan,
            &executed,
        )?);
        recovery_dependencies.push(task_dependency(&executed));
        let recovered_fields = import_structural_frequency_recovered_fields(
            pool,
            executed.response,
            task,
            preparation.inventory_limits,
        )
        .map_err(protocol)?;
        completed_entries.push(CompletedStructuralFrequencyEntry {
            entry,
            recovered_fields,
        });
    }

    let sequence = SequenceIdentity::from_solver_contract(&preparation.workload.model().sequence)
        .map_err(protocol)?;
    let step = preparation
        .workload
        .procedure_step()
        .map_err(protocol)?
        .clone();
    let recovered_fields = completed_entries
        .iter()
        .map(|entry| entry.recovered_fields.clone())
        .collect::<Vec<_>>();
    let observations = execute_structural_frequency_observations(
        pool,
        StructuralFrequencyObservationExecution {
            workload: workload_objects.clone(),
            mesh: mesh_objects.clone(),
            step,
            sequence,
            recoveries: &recovery_tasks,
            recovered_fields: &recovered_fields,
            recovery_dependencies: &recovery_dependencies,
            accelerator_requests: policy.resources.accelerators.clone(),
            observation_limits: preparation.observation_limits,
            program_revision: preparation.program_revision.clone(),
            chunk_bytes: preparation.chunk_bytes,
            inventory_limits: preparation.inventory_limits,
            local_execution: preparation.local_execution,
        },
        &policy,
    )
    .await?;
    resource_attempts.extend(observations.resource_attempts);

    let assembly_request = StructuralFrequencyResultAssemblyRequest::new(
        &completed_entries
            .iter()
            .map(|entry| entry.entry.result.clone())
            .collect::<Vec<_>>(),
    )
    .map_err(protocol)?;
    let assembly_request_objects = prepare_structural_frequency_result_assembly_request_objects(
        &assembly_request,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )
    .map_err(protocol)?;
    let publication_access = preparation.artifact_access.clone();
    let result_task =
        prepare_structural_frequency_result_task(StructuralFrequencyResultPreparation {
            request: assembly_request_objects,
            entries: completed_entries
                .iter()
                .map(|entry| entry.entry.objects.clone())
                .collect(),
            artifact_access: preparation.artifact_access,
            program_revision: preparation.program_revision,
            chunk_bytes: preparation.chunk_bytes,
            inventory_limits: preparation.inventory_limits,
        })
        .map_err(protocol)?;
    let [executed_result] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &result_task.host,
            program: &result_task.program,
            resource_plan: Some(&result_task.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: entry_dependencies,
        }],
        batch_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| {
        NativeExecutionError::Protocol(
            "structural frequency fan-in returned the wrong task inventory".into(),
        )
    })?;
    resource_attempts.push(crate::analysis_task_pool::planned_resource_attempt(
        &result_task.resource_plan,
        &executed_result,
    )?);
    let result = import_structural_frequency_result(
        pool,
        executed_result.response,
        &result_task,
        preparation.inventory_limits,
    )
    .map_err(protocol)?;
    let wall_millis = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let resource_graph =
        PlannedResourceUsageGraph::new(resource_attempts, wall_millis).map_err(protocol)?;
    let resource_usage = prepare_planned_resource_graph_usage(
        &result_task.resource_plan,
        preparation.workload.identity().digest(),
        result.result.manifest().identity().digest(),
        &resource_graph,
        &publication_access,
        preparation.chunk_bytes,
        preparation.inventory_limits,
    )
    .map_err(protocol)?;
    pool.register_result_objects(&resource_usage.objects, &publication_access)?;
    let publication =
        StructuralFrequencyPublicationManifest::new(StructuralFrequencyPublicationInput {
            workload: &preparation.workload,
            result: &result.result,
            receipt: result.numerical_receipt.clone(),
            samples: completed_entries
                .iter()
                .zip(&observations.samples)
                .zip(result.result.manifest().samples())
                .map(
                    |((entry, observations), result)| StructuralFrequencyPublicationSample {
                        result: *result,
                        source_entry: entry.entry.result.manifest().clone(),
                        coordinate: observations.sample,
                        fields: entry.recovered_fields.manifest.clone(),
                        observations: observations
                            .observations
                            .as_ref()
                            .map(|observations| observations.manifest.clone()),
                    },
                )
                .collect(),
            histories: observations
                .histories
                .iter()
                .map(|history| StructuralFrequencyPublicationHistory {
                    observation_id: history.observation_id.clone(),
                    checkpoint: history.checkpoint.checkpoint.manifest().clone(),
                    finalization: history
                        .finalization
                        .as_ref()
                        .map(|finalization| finalization.result.publication.clone()),
                })
                .collect(),
            resource_usage: resource_usage.evidence.clone(),
        })
        .map_err(protocol)?;
    Ok(CompletedStructuralFrequencyStudy {
        result,
        entries: completed_entries,
        observation_samples: observations.samples,
        histories: observations.histories,
        resource_usage,
        publication,
    })
}

pub(crate) fn batch_policy(policy: &StructuralSequenceExecutionPolicy) -> AnalysisTaskBatchPolicy {
    AnalysisTaskBatchPolicy {
        retry: policy.retry,
        deadline_unix_millis: policy.deadline_unix_millis,
        priority: policy.priority,
    }
}

fn task_dependency(task: &ExecutedAnalysisTask) -> AnalysisTaskDependency {
    AnalysisTaskDependency {
        identity: task.identity,
        task_id: task.task_id,
    }
}
