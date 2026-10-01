//! Local composition of the same portable whole-study root used by remote execution.

use runmat_analysis_fea::study::{
    compile_analysis_study, plan_structural_study, AnalysisStudyCompilationInput,
};
use runmat_analysis_fea::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};
use runmat_execution_artifact::object::ObjectInventoryLimits;

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
    let inventory = ObjectInventoryLimits::default();
    let composition_root = config
        .store_root
        .join("whole-studies")
        .join(uuid::Uuid::new_v4().simple().to_string());
    let mut composition_store =
        crate::FilesystemObjectStore::open(composition_root, inventory.max_object_bytes)
            .map_err(protocol)?;
    prepared.write_to(&mut composition_store)?;

    let decoded = crate::decode_structural_study_root(&request, &composition_store)?;
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
    let mut compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: &decoded.study,
        geometry: &output.document,
        regions: &output.regions,
        materials: &decoded.materials,
        mesh: &meshing.artifact,
    })
    .map_err(protocol)?;
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
    let workloads = plan_structural_study(&compiled, decoded.planning).map_err(protocol)?;
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
        mesh: meshing.artifact.clone(),
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
