//! Admit the complete portable structural result without exposing any step.
use super::*;

pub(in crate::analysis) fn prepare_structural_study_intent_completion(
    intent: PreparedStructuralStudyIntent,
    meshing_output: crate::meshing::MeshingExecutionOutput,
    result: StructuralStudyNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    intent.validate().map_err(|message| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            message,
        )
    })?;
    meshing_output
        .evidence
        .validate(&meshing_output.artifact)
        .map_err(|error| {
            intent_completion_error(
                &intent,
                "RM.FEA.RUN_STUDY.INVALID_REMOTE_MESH_EVIDENCE",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                error.to_string(),
            )
        })?;
    let expected_source =
        runmat_meshing_core::StableDigest::from_bytes(*intent.geometry_source_digest.bytes());
    if meshing_output.artifact.geometry.source_digest != expected_source
        || meshing_output.artifact.geometry.geometry_revision
            != u64::from(intent.model.geometry_revision)
        || Some(&meshing_output.artifact.resolved_request) != intent.meshing_request.as_ref()
    {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.REMOTE_MESH_IDENTITY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "remote mesh source, geometry revision, or meshing request differs from the prepared study intent",
        ));
    }

    let study_model = intent.model.clone();
    let persisted_mesh = preparation
        .persist_mesh(&meshing_output.artifact, &meshing_output.evidence)
        .map_err(|error| {
            intent_completion_error(
                &intent,
                "RM.FEA.RUN_STUDY.MESH_ARTIFACT_STORE_FAILED",
                OperationErrorType::Internal,
                OperationErrorSeverity::Error,
                error.to_string(),
            )
        })?;
    match (intent.workloads.clone(), result) {
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads,
            },
            StructuralStudyNumericalResult::ThermalPrestress {
                thermal,
                thermal_source,
                structural,
            },
        ) => thermal_completion::complete(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            workloads,
            *thermal,
            thermal_source,
            *structural,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence {
                workloads: authored_workloads,
            },
            StructuralStudyNumericalResult::LinearSequence(result),
        ) => complete_linear_sequence(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            authored_workloads,
            result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralTransient {
                workload,
                ..
            },
            StructuralStudyNumericalResult::StructuralTransient(result),
        ) => complete_direct_transient(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            *workload,
            *result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralFrequency {
                workload,
            },
            StructuralStudyNumericalResult::StructuralFrequency(result),
        ) => complete_structural_frequency(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            *workload,
            *result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralModal { workload },
            StructuralStudyNumericalResult::StructuralModal(result),
        ) => complete_structural_modal(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            *workload,
            *result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads,
            },
            StructuralStudyNumericalResult::StructuralModalTransient(result),
        ) => complete_structural_modal_transient(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            workloads,
            *result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads,
            },
            StructuralStudyNumericalResult::StructuralModalFrequency(result),
        ) => complete_structural_modal_frequency(
            intent,
            study_model,
            ModalFrequencyMeshes {
                persisted: persisted_mesh,
                solver: meshing_output.artifact,
            },
            workloads,
            *result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads,
            },
            StructuralStudyNumericalResult::StructuralModalResponseSpectrum(result),
        ) => complete_structural_response_spectrum(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            workloads,
            *result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads,
            },
            StructuralStudyNumericalResult::StructuralRandomResponse(result),
        ) => complete_structural_random_response(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            workloads,
            *result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads,
            },
            StructuralStudyNumericalResult::StructuralPrestressedModal(result),
        ) => complete_structural_prestressed_modal(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            workloads,
            *result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads,
            },
            StructuralStudyNumericalResult::StructuralPrestressedFrequency(result),
        ) => complete_structural_prestressed_frequency(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            workloads,
            *result,
            completion_identity,
            preparation,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads,
            },
            StructuralStudyNumericalResult::StructuralBuckling(result),
        ) => complete_structural_buckling(
            intent,
            study_model,
            persisted_mesh,
            meshing_output.artifact,
            workloads,
            *result,
            completion_identity,
            preparation,
        ),
        _ => Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "remote structural-study result kind differs from its prepared workload kind",
        )),
    }
}
