//! Local composition of the same portable whole-study root used by remote execution.

use runmat_analysis_fea::study::{
    compile_analysis_study, plan_structural_study_with_fields, AnalysisStudyCompilationInput,
};
use runmat_analysis_fea::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};

use crate::{
    execute_structural_workloads_locally, NativeExecutionConfig, NativeExecutionError,
    NativeExecutionResult, PreparedStructuralStudyRoot,
};

#[derive(Clone, Default)]
pub struct LocalStructuralStudyObservers {
    pub progress: Option<crate::LocalFeaProgressObserver>,
    pub mesh: Option<crate::LocalFeaMeshObserver>,
}

impl std::fmt::Debug for LocalStructuralStudyObservers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalStructuralStudyObservers")
            .field("progress", &self.progress.is_some())
            .field("mesh", &self.mesh.is_some())
            .finish()
    }
}

/// Execute a portable typed study through geometry, exact meshing, compilation, and solve.
///
/// The input is the same immutable root transported to a remote driver. Local execution only
/// supplies process placement and storage; it does not reconstruct the analysis definition.
pub async fn execute_prepared_structural_study_locally(
    config: NativeExecutionConfig,
    prepared: PreparedStructuralStudyRoot,
    workers: u32,
) -> NativeExecutionResult<runmat_runtime::analysis::StructuralStudyResult> {
    execute_prepared_structural_study_locally_with_observers(
        config,
        prepared,
        workers,
        LocalStructuralStudyObservers::default(),
    )
    .await
}

pub async fn execute_prepared_structural_study_locally_with_observers(
    config: NativeExecutionConfig,
    prepared: PreparedStructuralStudyRoot,
    workers: u32,
    observers: LocalStructuralStudyObservers,
) -> NativeExecutionResult<runmat_runtime::analysis::StructuralStudyResult> {
    match execute_prepared_study_locally_with_observers(config, prepared, workers, observers)
        .await?
    {
        LocalStudyCompletion::Structural(result) => Ok(*result),
        LocalStudyCompletion::Family(_) => Err(protocol(
            "a family study root was executed as a structural study",
        )),
    }
}

/// Local completion of any authored study root.
#[derive(Debug)]
pub enum LocalStudyCompletion {
    Structural(Box<runmat_runtime::analysis::StructuralStudyResult>),
    Family(Box<crate::CompletedFeaStudyResult>),
}

/// Execute any portable typed study root locally: geometry, exact meshing, compilation and the
/// family's governed task graph, all through process-isolated workers.
pub async fn execute_prepared_study_locally_with_observers(
    mut config: NativeExecutionConfig,
    prepared: PreparedStructuralStudyRoot,
    workers: u32,
    observers: LocalStructuralStudyObservers,
) -> NativeExecutionResult<LocalStudyCompletion> {
    const OPERATION: &str = "fea.typed_study";
    emit_local_progress(
        &observers,
        OPERATION,
        FeaProgressPhase::GeometryLoad,
        FeaProgressStatus::Started,
        "Loading and validating study geometry",
        None,
        None,
    );
    let request = prepared.request().clone();
    // The composition store holds this process's own prepared study objects, which the study
    // preparation already admitted; only the platform's single-object ceiling applies here.
    let composition_root = config
        .store_root
        .join("whole-studies")
        .join(uuid::Uuid::new_v4().simple().to_string());
    let mut composition_store = crate::FilesystemObjectStore::open(
        composition_root,
        runmat_execution_artifact::object::PLATFORM_MAXIMUM_OBJECT_BYTES,
    )
    .map_err(protocol)?;
    prepared.write_to(&mut composition_store)?;
    drop(prepared);

    let decoded = crate::decode_structural_study_root(&request, &composition_store)?;
    config.max_workers = workers.max(1);
    // The root's execution policy bounds every analysis worker this placement launches.
    config.authored_worker_memory_bytes = decoded.memory_per_worker_bytes;
    config.worker_cpu_millicores = decoded
        .local_execution
        .checked_cpu_millicores()
        .map_err(|error| crate::NativeExecutionError::Configuration(error.to_string()))?;
    let crate::local_study_geometry::LocalStudyGeometry {
        output,
        meshing,
        analysis_config,
    } = crate::local_study_geometry::execute_local_study_geometry(
        config,
        &decoded.geometry_meshing,
        &decoded.artifact_access,
        &mut composition_store,
        &observers,
    )
    .await?;

    emit_local_progress(
        &observers,
        OPERATION,
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "Assembling typed study operators and inputs",
        None,
        None,
    );
    if crate::StudyFamilyKind::of(&decoded.planning).is_some() {
        return Box::pin(execute_family_locally(
            decoded,
            output,
            meshing,
            analysis_config,
            composition_store,
            workers,
            &observers,
        ))
        .await
        .map(Box::new)
        .map(LocalStudyCompletion::Family);
    }
    let planning = decoded
        .planning
        .structural()
        .ok_or_else(|| protocol("structural study root carries family planning"))?;
    let mut compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: &decoded.study,
        geometry: &output.document,
        regions: &output.regions,
        materials: &decoded.materials,
        mesh: &meshing.artifact,
    })
    .map_err(protocol)?;
    // Geometry has completed compilation; numerical execution retains its bound
    // mesh and property identities, not the original geometry materialization.
    drop(output);
    let resolved_inputs = decoded
        .input_materializations
        .iter()
        .map(|plan| plan.resolve(&meshing.artifact).map_err(protocol))
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let completed_inputs = crate::input_materialization::execute_locally(
        analysis_config.clone(),
        &resolved_inputs,
        &composition_store,
        &decoded.artifact_access,
        workers,
        &meshing.artifact,
    )
    .await?;
    crate::input_materialization::apply_to_model(&mut compiled.model, completed_inputs)?;
    let workloads = plan_structural_study_with_fields(&compiled, planning, &decoded.initial_fields)
        .map_err(protocol)?;
    emit_local_progress(
        &observers,
        OPERATION,
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "Study operators and inputs assembled",
        Some(1),
        Some(1),
    );
    emit_local_progress(
        &observers,
        OPERATION,
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "Executing the analysis sequence",
        None,
        None,
    );
    let numerical = execute_structural_workloads_locally(crate::StructuralWorkloadLocalExecution {
        execution_config: analysis_config,
        program_environment: request.recipe.program_revision.environment(),
        study_digest: decoded.authored_study_digest,
        frozen_project_digest: decoded.frozen_project_digest,
        compiled: Some(&compiled),
        workloads: workloads.clone(),
        mesh: &meshing.artifact,
        workers,
        local_execution: decoded.local_execution,
        admitted_accelerators: Some(&decoded.analysis_accelerator_requests),
    })
    .await?;
    emit_local_progress(
        &observers,
        OPERATION,
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "Analysis sequence completed",
        Some(1),
        Some(1),
    );
    emit_local_progress(
        &observers,
        OPERATION,
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "Recovering fields and assembling study results",
        None,
        None,
    );
    let result = runmat_runtime::analysis::StructuralStudyResult::new(
        compiled,
        meshing.artifact,
        meshing.evidence,
        workloads,
        numerical,
    )
    .map_err(protocol)?;
    let result = LocalStudyCompletion::Structural(Box::new(result));
    emit_local_progress(
        &observers,
        OPERATION,
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "Study results are ready for publication",
        Some(1),
        Some(1),
    );
    Ok(result)
}

async fn execute_family_locally(
    decoded: crate::DecodedStructuralStudyRoot,
    output: runmat_analysis_execution::GeometryMaterializationOutput,
    meshing: runmat_meshing_execution::ExactMeshingDagRunResult,
    mut analysis_config: NativeExecutionConfig,
    composition_store: crate::FilesystemObjectStore,
    workers: u32,
    observers: &LocalStructuralStudyObservers,
) -> NativeExecutionResult<crate::CompletedFeaStudyResult> {
    const OPERATION: &str = "fea.typed_study";
    let resolved_inputs = decoded
        .input_materializations
        .iter()
        .map(|plan| plan.resolve(&meshing.artifact).map_err(protocol))
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let mut completed_inputs = Some(
        crate::input_materialization::execute_locally(
            analysis_config.clone(),
            &resolved_inputs,
            &composition_store,
            &decoded.artifact_access,
            workers,
            &meshing.artifact,
        )
        .await?,
    );
    let planned = crate::study_family::plan_family_study(
        crate::study_family::FamilyStudyContext {
            study: &decoded.study,
            materials: &decoded.materials,
            planning: &decoded.planning,
            document: &output.document,
            regions: &output.regions,
            mesh: &meshing.artifact,
            initial_fields: &decoded.initial_fields,
            frozen_project_digest: decoded.frozen_project_digest,
            artifact_access: &decoded.artifact_access,
            program_revision: &decoded.geometry_meshing.recipe.program_revision,
            local_execution: decoded.local_execution,
        },
        &mut |model| {
            let completed = completed_inputs.take().unwrap_or_default();
            crate::input_materialization::apply_to_model(model, completed)
                .map(drop)
                .map_err(|error| error.to_string())
        },
    )
    .map_err(NativeExecutionError::Protocol)?;
    drop(output);
    emit_local_progress(
        observers,
        OPERATION,
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "Study operators and inputs assembled",
        Some(1),
        Some(1),
    );
    let resources = planned
        .local_resources(
            decoded.local_execution,
            analysis_config.local_memory_capacity_bytes,
        )
        .map_err(NativeExecutionError::Protocol)?;
    analysis_config.max_workers = workers.max(1);
    crate::analysis_task_pool::configure_local_analysis_envelope(&mut analysis_config, &resources);
    let session = crate::NativeProgramSession::new(analysis_config)?;
    // Publication outlives the executing pool. A writable store clone does not retain the
    // session directory; the shared object source carries that lifetime authority.
    let source = session.object_source();
    let mut result_store = session.object_store();
    let mut pool = crate::analysis_task_pool::LocalAnalysisTaskPool::new(session);
    emit_local_progress(
        observers,
        OPERATION,
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "Executing the analysis sequence",
        None,
        None,
    );
    let completed = crate::study_family::execute_family_study(
        &mut pool,
        planned,
        crate::study_family::FamilyStudyExecutionPolicy {
            resources,
            retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .await?;
    let result = crate::typed_fea::local_family_result(
        completed,
        source,
        &mut result_store,
        meshing.artifact,
        meshing.evidence,
    )
    .map_err(NativeExecutionError::Protocol)?;
    emit_local_progress(
        observers,
        OPERATION,
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "Analysis sequence completed",
        Some(1),
        Some(1),
    );
    Ok(result)
}

pub(crate) fn emit_local_progress(
    observers: &LocalStructuralStudyObservers,
    operation: impl Into<String>,
    phase: FeaProgressPhase,
    status: FeaProgressStatus,
    message: impl Into<String>,
    current: Option<u64>,
    total: Option<u64>,
) {
    let event = FeaProgressEvent {
        operation: operation.into(),
        phase,
        status,
        message: message.into(),
        current,
        total,
        fraction: match (current, total) {
            (Some(current), Some(total)) if total > 0 => {
                Some((current as f64 / total as f64).min(1.0))
            }
            _ => None,
        },
        metadata: Default::default(),
    };
    runmat_analysis_fea::emit_fea_progress(event.clone());
    if let Some(observer) = observers.progress.as_ref() {
        observer(event);
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
