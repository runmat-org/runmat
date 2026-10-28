use super::*;

pub(in crate::analysis) fn complete_modal_study(
    mut prepared: PreparedStudyCompletion,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workload: runmat_analysis_fea::execution::StructuralModalWorkload,
    numerical: StructuralStudyModalNumericalResult,
    assigned_run_id: String,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [step] = prepared.steps.as_slice() else {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "structural modal completion requires exactly one authored procedure step",
        ));
    };
    let step = step.clone();
    if numerical.step_id != step.procedure_step.step_id
        || numerical.step_id != *authored_workload.step_id()
    {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "structural modal result names another procedure step",
        ));
    }
    let AnalysisSolverOptions::StructuralModes(mut options) =
        step.authored_execution.solver_options.clone()
    else {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "structural modal workload has non-modal solver options",
        ));
    };
    authored_workload
        .replace_model(prepared.model.clone())
        .map_err(|error| {
            modal_completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.MODAL_WORKLOAD_INVALID",
                error.to_string(),
            )
        })?;
    let recovered = runmat_analysis_fea::pipeline::modal::recover_completed_structural_modal_run(
        &authored_workload,
        &solver_mesh,
        &numerical.result,
        &numerical.numerical_receipt,
    )
    .map_err(|error| {
        modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.MODAL_RESULT_ADMISSION_FAILED",
            error.to_string(),
        )
    })?;
    options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    let run = analysis_admit_modal_run_with_id(
        recovered,
        authored_workload.backend(),
        options.clone(),
        prepared.context.clone(),
        Some(&solver_mesh),
        Some(&assigned_run_id),
    )?;
    let solver_result_identity = numerical.numerical_receipt.source_result();
    let states = modal_basis_states_for_completion(
        &prepared,
        &solver_mesh,
        &authored_workload,
        solver_result_identity,
        numerical.result.basis(),
    )?;
    prepared.steps[0].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralModes(options.clone());
    complete_study_run_prepared(
        prepared,
        vec![CompletedStudyStep {
            step_id: numerical.step_id,
            solver_result_identity,
            run: CompletedStudyStepRun::Standard(Box::new(run)),
            resolved_solver_options: AnalysisSolverOptions::StructuralModes(options),
        }],
        states,
        numerical.frozen_project_digest,
        preparation,
    )
}

pub(super) fn modal_basis_states_for_completion(
    prepared: &PreparedStudyCompletion,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    workload: &runmat_analysis_fea::execution::StructuralModalWorkload,
    source_result: runmat_analysis_fea::execution::SolverResultIdentity,
    basis: &runmat_analysis_fea::ModalBasisArtifact,
) -> Result<Vec<runmat_analysis_fea::execution::CanonicalSemanticState>, OperationErrorEnvelope> {
    let step = workload.procedure_step().map_err(|error| {
        modal_completion_error(
            prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let declaration = workload.modal_basis_output().map_err(|error| {
        modal_completion_error(
            prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let sequence = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &prepared.model.sequence,
    )
    .map_err(|error| {
        modal_completion_error(
            prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let producer =
        runmat_analysis_fea::execution::StepIdentity::from_solver_contract(sequence, step)
            .map_err(|error| {
                modal_completion_error(
                    prepared,
                    "RM.FEA.RUN_STUDY.STATE_INVALID",
                    error.to_string(),
                )
            })?;
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let state = runmat_analysis_fea::execution::CanonicalSemanticState::new(
        runmat_analysis_fea::execution::CanonicalSemanticStateInput {
            producer,
            declaration: declaration.clone(),
            topology_digest,
            source_result,
            payload: runmat_analysis_fea::execution::CanonicalSemanticStatePayload::ModalBasis {
                basis: Box::new(basis.clone()),
            },
        },
    )
    .map_err(|error| {
        modal_completion_error(
            prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    Ok(vec![state])
}

pub(in crate::analysis) fn complete_modal_transient_study(
    mut prepared: PreparedStudyCompletion,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: StructuralStudyModalTransientNumericalResult,
    modal_run_id: String,
    transient_run_id: String,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [modal_step, transient_step] = prepared.steps.as_slice() else {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-transient completion requires exactly two authored procedure steps",
        ));
    };
    let modal_step = modal_step.clone();
    let transient_step = transient_step.clone();
    if numerical.modal_step_id != modal_step.procedure_step.step_id
        || numerical.transient_step_id != transient_step.procedure_step.step_id
    {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-transient result step order differs from the authored sequence",
        ));
    }
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
        workload: modal_workload,
    }) = authored_workloads.remove(&numerical.modal_step_id)
    else {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-transient workload is missing its modal extraction step",
        ));
    };
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::ModalTransient {
        template,
        ..
    }) = authored_workloads.remove(&numerical.transient_step_id)
    else {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-transient workload is missing its transient step",
        ));
    };
    if !authored_workloads.is_empty() {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "modal-transient workload contains an unauthored procedure step",
        ));
    }
    let AnalysisSolverOptions::StructuralModes(mut modal_options) =
        modal_step.authored_execution.solver_options.clone()
    else {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "modal-transient producer has non-modal solver options",
        ));
    };
    let AnalysisSolverOptions::StructuralTransient(mut transient_options) =
        transient_step.authored_execution.solver_options.clone()
    else {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "modal-transient consumer has non-transient solver options",
        ));
    };
    let mut modal_workload = *modal_workload;
    modal_workload
        .replace_model(prepared.model.clone())
        .map_err(|error| {
            modal_completion_error(
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
            modal_completion_error(
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
    let expected_basis_states = modal_basis_states_for_completion(
        &prepared,
        &solver_mesh,
        &modal_workload,
        modal_result_identity,
        numerical.modal_result.basis(),
    )?;
    if expected_basis_states.as_slice() != [numerical.basis_state.clone()] {
        return Err(modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            "published modal basis state differs from the admitted producer result",
        ));
    }

    let mut transient_template = *template;
    transient_template.model = prepared.model.clone();
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &prepared.model,
        &transient_template.step_id,
        &numerical.basis_state,
        runmat_analysis_core::StateUse::ModalBasis,
        topology_digest,
    )
    .map_err(|error| {
        modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let transient_workload = transient_template.bind(transition).map_err(|error| {
        modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.TRANSIENT_WORKLOAD_INVALID",
            error.to_string(),
        )
    })?;
    let source_producer = numerical.transient_result.manifest().scope.step;
    let transient_result_identity = numerical.transient_result.manifest().identity;
    let recovered_transient =
        runmat_analysis_fea::pipeline::transient::recover_completed_transient_run(
            runmat_analysis_fea::pipeline::transient::CompletedStructuralTransientRunInput {
                workload: &transient_workload,
                solver_mesh: &solver_mesh,
                modal_state: Some(&numerical.basis_state),
                result: numerical.transient_result,
                numerical_receipt: &numerical.transient_numerical_receipt,
            },
        )
        .map_err(|error| {
            modal_completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.TRANSIENT_RESULT_ADMISSION_FAILED",
                error.to_string(),
            )
        })?;
    let original_fields =
        runmat_analysis_fea::pipeline::transient::original_transient_field_source(
            &prepared.model,
            &solver_mesh,
            source_producer,
            transient_result_identity,
            &recovered_transient,
        )
        .map_err(|error| {
            modal_completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.ORIGINAL_FIELDS_INVALID",
                error.to_string(),
            )
        })?;
    transient_options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    let transient_run = analysis_admit_transient_run_with_id(
        transient_workload.backend,
        transient_options.clone(),
        &prepared.context,
        Some(solver_mesh.clone()),
        recovered_transient,
        Some(&transient_run_id),
    )?;
    let mut states = expected_basis_states;
    let sequence = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &prepared.model.sequence,
    )
    .map_err(|error| {
        modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let transient_producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence,
        &transient_step.procedure_step,
    )
    .map_err(|error| {
        modal_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let transient_fields = crate::analysis::collect_semantic_state_fields(&transient_run.data);
    states.extend(
        transient_step
            .procedure_step
            .state_outputs
            .iter()
            .map(|declaration| {
                runmat_analysis_fea::execution::CanonicalSemanticState::new(
                    runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                        producer: transient_producer,
                        declaration: declaration.clone(),
                        topology_digest,
                        source_result: transient_result_identity,
                        payload:
                            runmat_analysis_fea::execution::CanonicalSemanticStatePayload::Fields {
                                fields: transient_fields.clone(),
                            },
                    },
                )
                .map_err(|error| {
                    modal_completion_error(
                        &prepared,
                        "RM.FEA.RUN_STUDY.STATE_INVALID",
                        error.to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    prepared.steps[0].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralModes(modal_options.clone());
    prepared.steps[1].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralTransient(transient_options.clone());
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
                step_id: numerical.transient_step_id,
                solver_result_identity: transient_result_identity,
                run: CompletedStudyStepRun::OriginalFields {
                    run: Box::new(transient_run),
                    source: Box::new(original_fields),
                },
                resolved_solver_options: AnalysisSolverOptions::StructuralTransient(
                    transient_options,
                ),
            },
        ],
        states,
        numerical.frozen_project_digest,
        preparation,
    )
}

fn modal_completion_error(
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

pub(super) fn prepared_completion(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
) -> PreparedStudyCompletion {
    PreparedStudyCompletion {
        thermal_prefix: None,
        context: intent.context,
        study_id: intent.study_id,
        study_fingerprint: study_fingerprint_from_digest(intent.authored_study_digest),
        authored_study_digest: intent.authored_study_digest,
        model: study_model,
        steps: intent
            .steps
            .into_iter()
            .map(|step| super::super::PreparedStudyStep {
                authored_execution: step.authored_execution,
                procedure_step: step.procedure_step,
            })
            .collect(),
        operation_sequence: intent.operation_sequence,
        solver_mesh_artifact_path: persisted_mesh.artifact_path,
        meshing_evidence_artifact_path: persisted_mesh.evidence_path,
    }
}
