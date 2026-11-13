//! Authenticated publication of a completed standalone authored thermal study.
//! Numerical fields, states and observations are retained producer outputs, never re-solved.
use super::{
    AnalysisStudyCompletionIdentity, AnalysisStudyRunData, AnalysisThermalRunOptions,
    ThermalStudyCompletion, ThermalStudyObjectSource,
};
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use runmat_execution::Digest;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact};

#[path = "thermal_study_publication/admission.rs"]
mod admission;
#[path = "thermal_study_publication/materialization.rs"]
pub(in crate::analysis) mod materialization;
#[path = "thermal_study_publication/preparation.rs"]
mod preparation;

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
    let publication_context = context.clone();
    let study_id = completed.compiled.study_id.clone();
    prepare_thermal_study_publication(
        completed,
        source,
        mesh,
        meshing_evidence,
        options,
        study_definition_digest,
        context,
        publication_identity,
    )?
    .publish_without_function_execution()
    .map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(
            error,
            &publication_context,
            |message| publication_error(&publication_context, &study_id, message),
        )
    })
}

pub fn prepare_thermal_study_publication(
    completed: &ThermalStudyCompletion,
    source: &ThermalStudyObjectSource,
    mesh: &SolverMeshArtifact,
    meshing_evidence: &MeshingEvidence,
    options: &AnalysisThermalRunOptions,
    study_definition_digest: Digest,
    context: OperationContext,
    publication_identity: AnalysisStudyCompletionIdentity,
) -> Result<crate::analysis::PreparedStudyPublication, OperationErrorEnvelope> {
    let invalid = |message| publication_error(&context, &completed.compiled.study_id, message);
    let (mut preparation, completion_copies) = preparation::capture(
        study_definition_digest,
        completed,
        mesh,
        meshing_evidence,
        options,
    )
    .map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(error, &context, invalid)
    })?;
    source
        .admit_publication(completed, &mut preparation.staging)
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
        source
            .authenticate_admitted(
                completed,
                preparation
                    .staging
                    .source_replay_allowance(&completed.catalog.objects)
                    .map_err(|error| {
                        crate::analysis::prepared_study_publication::publication_failure(
                            error, &context, invalid,
                        )
                    })?,
            )
            .map_err(|error| {
                crate::analysis::prepared_study_publication::publication_failure(
                    error, &context, invalid,
                )
            })?;
        admission::validate(
            completed,
            mesh,
            meshing_evidence,
            options,
            study_definition_digest,
        )
        .map_err(invalid)?;
    }
    let run_id = publication_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Thermal,
        study_definition_digest,
        &completed.workload.step_id,
    );
    // Persist borrowed geometry before constructing retained field histories, so
    // mesh encoding and complete public-run materialization do not coexist.
    let persisted = preparation
        .persist_mesh(mesh, meshing_evidence)
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
    // Reconstruct only retained partitions; the shared owner also validates accuracy/resource
    // closure and projects catalog-backed observations without gathering or evaluating them.
    let object_artifact_paths = source
        .persist(completed, study_definition_digest, &preparation.staging)
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
    let run = {
        let _source_phase = preparation.staging.begin_source_phase().map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
        materialization::admit(completed, source, mesh, options, &run_id, &context)
            .map_err(invalid)?
    };
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
    let mut published =
        crate::analysis::structural_study::publication::complete_study_run_prepared(
            prepared,
            Vec::new(),
            Vec::new(),
            completed.frozen_project_digest,
            preparation,
        )?;
    // The completed family call has dropped its owned completion/model/options.
    published
        .staging
        .release_owned_copies(completion_copies)
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
    let evidence = ThermalStudyNumericalEvidence::Thermal {
        root: completed.root.clone(),
        catalog: completed.catalog.clone(),
        object_artifact_paths,
    };
    let manifest = super::study_result_manifest::StudyResultManifest::new(
        completed
            .compiled
            .definition()
            .map_err(|error| invalid(error.to_string()))?,
        completed
            .workload
            .identity()
            .map_err(|error| invalid(error.to_string()))?
            .digest(),
        &completed.workload,
        mesh,
        meshing_evidence,
        evidence,
        &mut published,
    )
    .map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(error, &context, invalid)
    })?;
    crate::analysis::PreparedStudyPublication::new(published, manifest).map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(error, &context, invalid)
    })
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
