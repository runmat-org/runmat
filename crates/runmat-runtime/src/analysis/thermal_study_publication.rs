//! Authenticated publication of a completed standalone authored thermal study.
//! Numerical fields, states and observations are retained producer outputs, never re-solved.
use super::{
    AnalysisStudyCompletionIdentity, AnalysisStudyRunData, AnalysisThermalRunOptions,
    ThermalStudyCompletion, ThermalStudyObjectSource,
};
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use runmat_execution::Digest;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact};

mod admission;
pub(in crate::analysis) mod materialization;

/// Authenticate the source-bound definition, numerical result and meshing provenance before
/// admitting it through the same study artifact and cold-query contracts as other physics.
pub fn publish_thermal_study_result(
    completed: &ThermalStudyCompletion,
    source: &ThermalStudyObjectSource,
    mesh: &SolverMeshArtifact,
    meshing_evidence: &MeshingEvidence,
    options: &AnalysisThermalRunOptions,
    study_definition_digest: Digest,
    context: OperationContext,
    publication_identity: AnalysisStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let invalid = |message| publication_error(&context, &completed.compiled.study_id, message);
    admission::validate(
        completed,
        source,
        mesh,
        meshing_evidence,
        options,
        study_definition_digest,
    )
    .map_err(invalid)?;
    let run_id = publication_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Thermal,
        study_definition_digest,
        &completed.workload.step_id,
    );
    // Reconstruct only retained partitions; the shared owner also validates accuracy/resource
    // closure and projects catalog-backed observations without gathering or evaluating them.
    let run = materialization::admit(completed, source, mesh, options, &run_id, &context)
        .map_err(invalid)?;
    let object_artifact_paths = source
        .persist(completed, study_definition_digest)
        .map_err(invalid)?;
    let persisted = super::meshing::persist_mesh_output(&crate::meshing::MeshingExecutionOutput {
        artifact: mesh.clone(),
        evidence: meshing_evidence.clone(),
        region_aliases: Default::default(),
    })
    .map_err(invalid)?;
    let mut resolved_options = options.clone();
    resolved_options.solver_mesh_artifact_path = persisted.artifact_path.clone();
    let prepared = super::structural_study::standalone_thermal_completion(
        context.clone(),
        study_definition_digest,
        completed.clone(),
        source.clone(),
        run,
        resolved_options,
        persisted,
    );
    let mut published = super::complete_study_run(
        prepared,
        Vec::new(),
        Vec::new(),
        completed.frozen_project_digest,
    )?;
    let evidence = ThermalStudyNumericalEvidence::Thermal {
        root: completed.root.clone(),
        catalog: completed.catalog.clone(),
        object_artifact_paths,
    };
    let manifest = super::study_result_manifest::StudyResultManifest::new(
        &completed.compiled,
        completed
            .workload
            .identity()
            .map_err(|error| invalid(error.to_string()))?
            .digest(),
        completed.workload.clone(),
        mesh,
        meshing_evidence,
        evidence,
        published.data.clone(),
    )
    .map_err(invalid)?;
    published.data.result_manifest_artifact_path = Some(manifest.persist().map_err(invalid)?);
    Ok(published)
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ThermalStudyNumericalEvidence {
    Thermal {
        root: runmat_execution::value::ValueRef,
        catalog: runmat_execution::value::PublishedObjectCatalog,
        object_artifact_paths: Vec<String>,
    },
}

fn publication_error(
    context: &OperationContext,
    study_id: &str,
    message: String,
) -> OperationErrorEnvelope {
    crate::operations::operation_error(
        super::ANALYSIS_RUN_STUDY_OPERATION,
        super::ANALYSIS_RUN_STUDY_OP_VERSION,
        context,
        crate::operations::OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.INVALID_THERMAL_RESULT",
            error_type: crate::operations::OperationErrorType::Contract,
            retryable: false,
            severity: crate::operations::OperationErrorSeverity::Fatal,
        },
        message,
        std::collections::BTreeMap::from([("study_id".into(), study_id.into())]),
    )
}
