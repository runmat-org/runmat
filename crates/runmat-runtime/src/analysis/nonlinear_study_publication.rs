//! Authenticated original J2 completion publication; runtime executes no physics kernels.
use crate::analysis::{
    AnalysisNonlinearRunOptions, AnalysisSolverOptions, AnalysisStudyCompletionIdentity,
    AnalysisStudyRunData, NonlinearSolidStudyCompletion, NonlinearSolidStudyObjectSource,
};
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use runmat_execution::Digest;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact};
#[path = "nonlinear_study_publication/admission.rs"]
mod admission;
#[path = "nonlinear_study_publication/materialization.rs"]
mod materialization;

pub fn prepare_nonlinear_solid_study_publication(
    completed: &NonlinearSolidStudyCompletion,
    source: &NonlinearSolidStudyObjectSource,
    mesh: &SolverMeshArtifact,
    meshing_evidence: &MeshingEvidence,
    definition: Digest,
    context: OperationContext,
    publication_identity: AnalysisStudyCompletionIdentity,
) -> Result<crate::analysis::PreparedStudyPublication, OperationErrorEnvelope> {
    let invalid = |message| publication_error(&context, &completed.compiled.study_id, message);
    let mut preparation =
        crate::analysis::study_publication::preparation::PublicationPreparation::capture_catalog(
            definition,
            &(completed, mesh, meshing_evidence),
            &completed.catalog,
        )
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;

    {
        let _source_phase = preparation.staging.begin_source_phase().map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
        let workspace = preparation
            .staging
            .source_workspace_with_preparation(&preparation.retained_input, completed)
            .map_err(|error| {
                crate::analysis::prepared_study_publication::publication_failure(
                    error, &context, invalid,
                )
            })?;
        admission::validate(
            completed,
            source,
            mesh,
            meshing_evidence,
            definition,
            workspace,
        )
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error.context("nonlinear input authentication"),
                &context,
                invalid,
            )
        })?;
    }
    let options = admission::options(&completed.workload.controls);
    let step_id = &completed.workload.model.sequence.steps[0].step_id;
    let run_id = publication_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        definition,
        step_id,
    );
    let object_artifact_paths = source
        .persist(
            completed,
            definition,
            &preparation.staging,
            &preparation.retained_input,
        )
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error.context("nonlinear immutable source persistence"),
                &context,
                invalid,
            )
        })?;
    let run = {
        let _source_phase = preparation.staging.begin_source_phase().map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
        materialization::admit(
            completed,
            source,
            mesh,
            &options,
            &run_id,
            &context,
            &mut preparation.staging,
            &preparation.retained_input,
        )
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error.context("nonlinear field materialization"),
                &context,
                invalid,
            )
        })?
    };
    let persisted = preparation
        .persist_mesh(mesh, meshing_evidence)
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
    let execution = super::AnalysisExecutionStep {
        step_id: step_id.clone(),
        backend: runmat_analysis_fea::ComputeBackend::Cpu,
        accelerator_admission: Default::default(),
        solver_options: AnalysisSolverOptions::StructuralNonlinear(options.clone()),
    };
    let study = completed
        .compiled
        .definition()
        .map_err(|error| invalid(error.to_string()))?;
    let prepared = super::structural_study::standalone_completion(
        context.clone(),
        study,
        execution,
        persisted,
    )
    .map_err(invalid)?;
    let step = super::structural_study::CompletedStudyStep {
        step_id: step_id.clone(),
        solver_result_identity: completed
            .terminal_block
            .source_result_identity()
            .map_err(|e| invalid(e.to_string()))?,
        run: super::structural_study::CompletedStudyStepRun::Standard(Box::new(run)),
        resolved_solver_options: AnalysisSolverOptions::StructuralNonlinear(options),
    };
    let mut published =
        crate::analysis::structural_study::publication::complete_study_run_prepared(
            prepared,
            vec![step],
            completed.states.clone(),
            completed.frozen_project_digest,
            preparation,
        )?;
    let manifest = super::study_result_manifest::StudyResultManifest::new(
        study,
        completed
            .workload
            .identity()
            .map_err(|e| invalid(e.to_string()))?
            .digest(),
        &completed.workload,
        mesh,
        meshing_evidence,
        NonlinearNumericalEvidence {
            root: completed.root.clone(),
            catalog: completed.catalog.clone(),
            object_artifact_paths,
        },
        &mut published,
    )
    .map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(error, &context, invalid)
    })?;
    crate::analysis::PreparedStudyPublication::new(published, manifest).map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(error, &context, invalid)
    })
}

pub fn publish_nonlinear_solid_study_result(
    completed: &NonlinearSolidStudyCompletion,
    source: &NonlinearSolidStudyObjectSource,
    mesh: &SolverMeshArtifact,
    meshing_evidence: &MeshingEvidence,
    definition: Digest,
    context: OperationContext,
    identity: AnalysisStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    prepare_nonlinear_solid_study_publication(
        completed,
        source,
        mesh,
        meshing_evidence,
        definition,
        context.clone(),
        identity,
    )?
    .publish_without_function_execution()
    .map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(
            error,
            &context,
            |message| publication_error(&context, &completed.compiled.study_id, message),
        )
    })
}
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NonlinearNumericalEvidence {
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
            error_code: "RM.FEA.RUN_STUDY.INVALID_NONLINEAR_RESULT",
            error_type: crate::operations::OperationErrorType::Contract,
            retryable: false,
            severity: crate::operations::OperationErrorSeverity::Fatal,
        },
        message,
        std::collections::BTreeMap::from([("study_id".into(), study_id.into())]),
    )
}
