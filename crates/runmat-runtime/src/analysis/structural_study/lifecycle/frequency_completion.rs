//! Admission and complete publication of original harmonic structural results.
use super::*;
use crate::analysis::frequency_result::admit_structural_frequency_result;

pub(super) fn complete_structural_frequency(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    authored_workload: runmat_analysis_fea::execution::StructuralFrequencyWorkload,
    numerical: crate::analysis::StructuralStudyFrequencyNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "structural-frequency completion requires exactly one authored procedure step",
        ));
    };
    let step = step.clone();
    let assigned_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &step.procedure_step.step_id,
    );
    complete_frequency_study(
        PreparedStudyCompletion {
            thermal_prefix: None,
            context: intent.context,
            study_id: intent.study_id,
            study_fingerprint: study_fingerprint_from_digest(intent.authored_study_digest),
            authored_study_digest: intent.authored_study_digest,
            model: study_model,
            steps: vec![crate::analysis::structural_study::PreparedStudyStep {
                authored_execution: step.authored_execution,
                procedure_step: step.procedure_step,
            }],
            operation_sequence: intent.operation_sequence,
            solver_mesh_artifact_path: persisted_mesh.artifact_path,
            meshing_evidence_artifact_path: persisted_mesh.evidence_path,
        },
        solver_mesh,
        authored_workload,
        numerical,
        assigned_run_id,
        preparation,
    )
}

pub(in crate::analysis) fn complete_frequency_study(
    mut prepared: PreparedStudyCompletion,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workload: runmat_analysis_fea::execution::StructuralFrequencyWorkload,
    numerical: crate::analysis::StructuralStudyFrequencyNumericalResult,
    assigned_run_id: String,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [step] = prepared.steps.as_slice() else {
        return Err(frequency_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "structural-frequency completion requires exactly one authored procedure step",
        ));
    };
    if numerical.step_id != step.procedure_step.step_id
        || numerical.step_id != *authored_workload.step_id()
    {
        return Err(frequency_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "structural-frequency result names another procedure step",
        ));
    }
    authored_workload
        .replace_model(prepared.model.clone())
        .map_err(|error| {
            frequency_completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.FREQUENCY_WORKLOAD_INVALID",
                error.to_string(),
            )
        })?;
    if numerical.workload != authored_workload {
        return Err(frequency_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.FREQUENCY_WORKLOAD_MISMATCH",
            "structural-frequency result differs from the resolved authored workload",
        ));
    }
    let AnalysisSolverOptions::StructuralFrequency(mut options) =
        step.authored_execution.solver_options.clone()
    else {
        return Err(frequency_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "structural-frequency workload has non-frequency solver options",
        ));
    };
    options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    let step_id = step.procedure_step.step_id.clone();
    prepared.steps[0].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralFrequency(options.clone());
    let solver_result_identity = numerical.result.manifest().identity();
    let frozen_project_digest = numerical.frozen_project_digest;
    let sequence_identity = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &prepared.model.sequence,
    )
    .map_err(|error| {
        frequency_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence_identity,
        &prepared.steps[0].procedure_step,
    )
    .map_err(|error| {
        frequency_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let states = prepared.steps[0]
        .procedure_step
        .state_outputs
        .iter()
        .map(|declaration| {
            runmat_analysis_fea::execution::CanonicalSemanticState::new(
                runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                    producer,
                    declaration: declaration.clone(),
                    topology_digest,
                    source_result: solver_result_identity,
                    payload: runmat_analysis_fea::execution::CanonicalSemanticStatePayload::StructuralFrequency {
                        publication: Box::new(numerical.publication.clone()),
                    },
                },
            )
            .map_err(|error| {
                frequency_completion_error(
                    &prepared,
                    "RM.FEA.RUN_STUDY.STATE_INVALID",
                    error.to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let run =
        admit_structural_frequency_result(&numerical, &solver_mesh, &options, assigned_run_id)
            .map_err(|error| {
                frequency_completion_error(
                    &prepared,
                    "RM.FEA.RUN_STUDY.FREQUENCY_RESULT_ADMISSION_FAILED",
                    error,
                )
            })?;
    complete_study_run_prepared(
        prepared,
        vec![CompletedStudyStep {
            step_id,
            solver_result_identity,
            run: CompletedStudyStepRun::StructuralFrequency(Box::new(run)),
            resolved_solver_options: AnalysisSolverOptions::StructuralFrequency(options),
        }],
        states,
        frozen_project_digest,
        preparation,
    )
}

fn frequency_completion_error(
    prepared: &PreparedStudyCompletion,
    error_code: &'static str,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &prepared.context,
        OperationErrorSpec {
            error_code,
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        message,
        BTreeMap::from([("study_id".to_string(), prepared.study_id.clone())]),
    )
}
