//! Authenticated whole-study maxwell publication; no electrical proxy or numerical re-solve.
use super::{
    AnalysisMaxwellRunOptions, AnalysisStudyCompletionIdentity, AnalysisStudyRunData,
    MaxwellStudyCompletion, MaxwellStudyObjectSource,
};
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use runmat_execution::Digest;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact};

#[path = "maxwell_study_publication/admission.rs"]
mod admission;
#[path = "maxwell_study_publication/materialization.rs"]
mod materialization;

pub fn publish_maxwell_study_result(
    completed: &MaxwellStudyCompletion,
    source: &MaxwellStudyObjectSource,
    mesh: &SolverMeshArtifact,
    meshing_evidence: &MeshingEvidence,
    options: &AnalysisMaxwellRunOptions,
    definition: Digest,
    context: OperationContext,
    publication_identity: AnalysisStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let invalid = |message| publication_error(&context, &completed.compiled.study_id, message);
    let fields = admission::validate(
        completed,
        source,
        mesh,
        meshing_evidence,
        options,
        definition,
    )
    .map_err(invalid)?;
    let run_id = publication_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Electromagnetic,
        definition,
        &completed.workload.step_id,
    );
    let run = materialization::admit(completed, mesh, options, fields, &run_id, &context)
        .map_err(invalid)?;
    let object_artifact_paths = source.persist(completed, definition).map_err(invalid)?;
    let persisted = super::meshing::persist_mesh_output(&crate::meshing::MeshingExecutionOutput {
        artifact: mesh.clone(),
        evidence: meshing_evidence.clone(),
        region_aliases: Default::default(),
    })
    .map_err(invalid)?;
    let execution = super::AnalysisExecutionStep {
        step_id: completed.workload.step_id.clone(),
        backend: runmat_analysis_fea::ComputeBackend::Cpu,
        requirement: completed.workload.formulation_requirement,
        solver_options: super::AnalysisSolverOptions::MaxwellHarmonic(options.clone()),
    };
    let prepared = super::structural_study::standalone_completion(
        context.clone(),
        &completed.compiled,
        execution,
        persisted,
    )
    .map_err(invalid)?;
    let step = super::structural_study::CompletedStudyStep {
        step_id: completed.workload.step_id.clone(),
        solver_result_identity: completed
            .result
            .identity()
            .map_err(|e| invalid(e.to_string()))?,
        run: super::structural_study::CompletedStudyStepRun::Standard(Box::new(run)),
        resolved_solver_options: super::AnalysisSolverOptions::MaxwellHarmonic(options.clone()),
    };
    let mut published = super::complete_study_run(
        prepared,
        vec![step],
        completed.states.clone(),
        completed.frozen_project_digest,
    )?;
    let manifest = super::study_result_manifest::StudyResultManifest::new(
        &completed.compiled,
        completed
            .workload
            .identity()
            .map_err(|e| invalid(e.to_string()))?
            .digest(),
        completed.workload.clone(),
        mesh,
        meshing_evidence,
        MaxwellNumericalEvidence {
            root: completed.root.clone(),
            catalog: completed.catalog.clone(),
            object_artifact_paths,
        },
        published.data.clone(),
    )
    .map_err(invalid)?;
    published.data.result_manifest_artifact_path = Some(manifest.persist().map_err(invalid)?);
    Ok(published)
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MaxwellNumericalEvidence {
    root: runmat_execution::value::ValueRef,
    catalog: runmat_execution::value::PublishedObjectCatalog,
    object_artifact_paths: Vec<String>,
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
            error_code: "RM.FEA.RUN_STUDY.INVALID_MAXWELL_RESULT",
            error_type: crate::operations::OperationErrorType::Contract,
            retryable: false,
            severity: crate::operations::OperationErrorSeverity::Fatal,
        },
        message,
        std::collections::BTreeMap::from([("study_id".into(), study_id.into())]),
    )
}
