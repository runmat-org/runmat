//! The documented study contract: solving a structural study requires a canonical solver mesh.

use std::collections::BTreeMap;

use super::structural_study::PreparedStudyStep;
use super::{
    operation_error, AnalysisStudySpec, OperationContext, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType, ANALYSIS_RUN_STUDY_OPERATION,
    ANALYSIS_RUN_STUDY_OP_VERSION,
};

/// The one refusal for a structural study that has no canonical solver mesh to solve on.
pub(super) fn structural_study_mesh_required(
    study_id: &str,
    context: &OperationContext,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.PORTABLE_MESH_REQUIRED",
            error_type: OperationErrorType::Input,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        "portable structural study execution requires a canonical solver mesh",
        BTreeMap::from([("study_id".to_string(), study_id.to_string())]),
    )
}

/// Every host prepares a study through `prepare_study_semantics`, so this is where a structural
/// step that has no solver mesh source is refused, before any mesher or step executor runs. A
/// step has a mesh source when it names a solver mesh artifact, when the study supplies one, or
/// when the study carries meshing settings; explicit beam and shell elements carry their own
/// discretization and need no solid mesh.
pub(super) fn require_structural_solver_mesh_source(
    spec: &AnalysisStudySpec,
    steps: &[PreparedStudyStep],
    context: &OperationContext,
) -> Result<(), OperationErrorEnvelope> {
    let study_mesh_source =
        spec.solver_mesh_artifact_path.is_some() || spec.meshing_settings.is_some();
    let explicit_elements = spec
        .model
        .structural
        .as_ref()
        .is_some_and(|structural| !structural.elements.is_empty());
    if study_mesh_source || explicit_elements {
        return Ok(());
    }
    let unmeshed_structural_step = steps.iter().any(|step| {
        let options = &step.authored_execution.solver_options;
        options.is_structural() && options.solver_mesh_artifact_path().is_none()
    });
    if unmeshed_structural_step {
        return Err(structural_study_mesh_required(&spec.study_id, context));
    }
    Ok(())
}
