//! Admit a modal-extraction step and the modal-reduced frequency sweep (`modal_harmonic`) that
//! consumes its published basis, then publish both runs and their states.
use super::modal_completion::{modal_basis_states_for_completion, prepared_completion};
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn complete_structural_modal_frequency(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyModalFrequencyNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [modal_step, frequency_step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "modal-frequency completion requires exactly two authored procedure steps",
        ));
    };
    let modal_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &modal_step.procedure_step.step_id,
    );
    let frequency_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &frequency_step.procedure_step.step_id,
    );
    complete_modal_frequency_study(
        prepared_completion(intent, study_model, persisted_mesh),
        solver_mesh,
        authored_workloads,
        numerical,
        modal_run_id,
        frequency_run_id,
        preparation,
    )
}

fn complete_modal_frequency_study(
    mut prepared: PreparedStudyCompletion,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyModalFrequencyNumericalResult,
    modal_run_id: String,
    frequency_run_id: String,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [modal_step, frequency_step] = prepared.steps.as_slice() else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-frequency completion requires exactly two authored procedure steps",
        ));
    };
    let modal_step = modal_step.clone();
    let frequency_step = frequency_step.clone();
    if numerical.modal_step_id != modal_step.procedure_step.step_id
        || numerical.frequency_step_id != frequency_step.procedure_step.step_id
        || numerical.frequency.step_id != numerical.frequency_step_id
        || numerical.frequency.frozen_project_digest != numerical.frozen_project_digest
    {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-frequency result step order differs from the authored sequence",
        ));
    }
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
        workload: modal_workload,
    }) = authored_workloads.remove(&numerical.modal_step_id)
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-frequency workload is missing its modal extraction step",
        ));
    };
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalFrequency { template }) =
        authored_workloads.remove(&numerical.frequency_step_id)
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-frequency workload is missing its unresolved frequency step",
        ));
    };
    if !authored_workloads.is_empty() {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-frequency workload contains an unauthored procedure step",
        ));
    }
    let AnalysisSolverOptions::StructuralModes(mut modal_options) =
        modal_step.authored_execution.solver_options.clone()
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "modal-frequency producer has non-modal solver options",
        ));
    };
    let AnalysisSolverOptions::StructuralFrequency(mut frequency_options) =
        frequency_step.authored_execution.solver_options.clone()
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "modal-frequency consumer has non-frequency solver options",
        ));
    };

    // The modal producer: recover and admit its run, and check the published basis state.
    let mut modal_workload = *modal_workload;
    modal_workload
        .replace_model(prepared.model.clone())
        .map_err(|error| {
            completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.MODAL_WORKLOAD_INVALID",
                error.to_string(),
            )
        })?;
    let recovered_modal =
        runmat_analysis_fea::pipeline::modal::recover_completed_structural_modal_run(
            &modal_workload,
            &solver_mesh,
            &numerical.modal_result,
            &numerical.modal_numerical_receipt,
        )
        .map_err(|error| {
            completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.MODAL_RESULT_ADMISSION_FAILED",
                error.to_string(),
            )
        })?;
    modal_options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    let modal_run = analysis_admit_modal_run_with_id(
        recovered_modal,
        modal_workload.backend(),
        modal_options.clone(),
        prepared.context.clone(),
        Some(&solver_mesh),
        Some(&modal_run_id),
    )?;
    let modal_result_identity = numerical.modal_numerical_receipt.source_result();
    let mut states = modal_basis_states_for_completion(
        &prepared,
        &solver_mesh,
        &modal_workload,
        modal_result_identity,
        numerical.modal_result.basis(),
    )?;
    if states.as_slice() != [numerical.basis_state.clone()] {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            "published modal basis state differs from the admitted producer result",
        ));
    }

    // The frequency consumer: bind the authored template to the basis it consumed, then admit
    // the published sweep against that basis.
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let mut template = *template;
    template.model = prepared.model.clone();
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &prepared.model,
        &template.step_id,
        &numerical.basis_state,
        runmat_analysis_core::StateUse::ModalBasis,
        topology_digest,
    )
    .map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let frequency_workload = template.bind(transition.source_state).map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.FREQUENCY_WORKLOAD_INVALID",
            error.to_string(),
        )
    })?;
    if numerical.frequency.workload != frequency_workload {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.FREQUENCY_WORKLOAD_MISMATCH",
            "modal-frequency result differs from its resolved authored workload",
        ));
    }
    frequency_options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    let frequency_result_identity = numerical.frequency.result.manifest().identity();
    let frequency_run = crate::analysis::frequency_result::admit_modal_structural_frequency_result(
        &numerical.frequency,
        &solver_mesh,
        &frequency_options,
        frequency_run_id,
        &numerical.basis_state,
    )
    .map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.FREQUENCY_RESULT_ADMISSION_FAILED",
            error,
        )
    })?;

    let sequence = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &prepared.model.sequence,
    )
    .map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let frequency_producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence,
        &frequency_step.procedure_step,
    )
    .map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    for declaration in &frequency_step.procedure_step.state_outputs {
        states.push(
            runmat_analysis_fea::execution::CanonicalSemanticState::new(
                runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                    producer: frequency_producer,
                    declaration: declaration.clone(),
                    topology_digest,
                    source_result: frequency_result_identity,
                    payload:
                        runmat_analysis_fea::execution::CanonicalSemanticStatePayload::StructuralFrequency {
                            publication: Box::new(numerical.frequency.publication.clone()),
                        },
                },
            )
            .map_err(|error| {
                completion_error(
                    &prepared,
                    "RM.FEA.RUN_STUDY.STATE_INVALID",
                    error.to_string(),
                )
            })?,
        );
    }

    prepared.steps[0].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralModes(modal_options.clone());
    prepared.steps[1].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralFrequency(frequency_options.clone());
    complete_study_run_prepared(
        prepared,
        vec![
            CompletedStudyStep {
                step_id: numerical.modal_step_id,
                solver_result_identity: modal_result_identity,
                run: CompletedStudyStepRun::Standard(Box::new(modal_run)),
                resolved_solver_options: AnalysisSolverOptions::StructuralModes(modal_options),
            },
            CompletedStudyStep {
                step_id: numerical.frequency_step_id,
                solver_result_identity: frequency_result_identity,
                run: CompletedStudyStepRun::StructuralFrequency(Box::new(frequency_run)),
                resolved_solver_options: AnalysisSolverOptions::StructuralFrequency(
                    frequency_options,
                ),
            },
        ],
        states,
        numerical.frozen_project_digest,
        preparation,
    )
}
