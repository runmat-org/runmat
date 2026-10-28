use super::*;

pub(in crate::analysis) fn complete_prestressed_modal_study(
    mut prepared: PreparedStudyCompletion,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyPrestressedModalNumericalResult,
    preload_run_id: String,
    modal_run_id: String,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [preload_step, modal_step] = prepared.steps.as_slice() else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-modal completion requires one preload and one modal step",
        ));
    };
    let preload_step = preload_step.clone();
    let modal_step = modal_step.clone();
    if numerical.preload_step_id != preload_step.procedure_step.step_id
        || numerical.modal_step_id != modal_step.procedure_step.step_id
    {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-modal numerical step order differs from the authored sequence",
        ));
    }
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
        workload: preload_workload,
    }) = authored_workloads.remove(&numerical.preload_step_id)
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-modal workload is missing its preload",
        ));
    };
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal { template }) =
        authored_workloads.remove(&numerical.modal_step_id)
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-modal workload is missing its unresolved modal step",
        ));
    };
    if !authored_workloads.is_empty() {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-modal workload contains an unauthored step",
        ));
    }
    let AnalysisSolverOptions::StructuralLinear(mut preload_options) =
        preload_step.authored_execution.solver_options.clone()
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "prestressed-modal preload has non-linear solver options",
        ));
    };
    let AnalysisSolverOptions::StructuralModes(mut modal_options) =
        modal_step.authored_execution.solver_options.clone()
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "prestressed-modal consumer has non-modal solver options",
        ));
    };

    let mut preload_workload = *preload_workload;
    preload_workload.model = prepared.model.clone();
    preload_workload
        .validate_sequence_template()
        .map_err(|error| {
            completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.PRELOAD_WORKLOAD_INVALID",
                error.to_string(),
            )
        })?;
    preload_options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    let preload_run = analysis_admit_linear_static_run_with_id(
        PreparedStructuralLinearRun {
            workload: preload_workload,
            solver_mesh: Some(solver_mesh.clone()),
            options: preload_options.clone(),
            context: prepared.context.clone(),
        },
        numerical.preload_result.run,
        Some(&preload_run_id),
        Default::default(),
    )?;
    numerical.prestress_state.validate().map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.PRESTRESS_STATE_INVALID",
            error.to_string(),
        )
    })?;
    if numerical.prestress_state.source_result != numerical.preload_solver_result_identity {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.PRESTRESS_STATE_INVALID",
            "prestress state does not identify the admitted preload result",
        ));
    }

    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let mut template = *template;
    template.model = prepared.model.clone();
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &prepared.model,
        &template.step_id,
        &numerical.prestress_state,
        runmat_analysis_core::StateUse::Prestress,
        topology_digest,
    )
    .map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.PRESTRESS_STATE_INVALID",
            error.to_string(),
        )
    })?;
    let workload = template.bind(transition).map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.MODAL_WORKLOAD_INVALID",
            error.to_string(),
        )
    })?;
    let recovered = runmat_analysis_fea::recover_completed_structural_prestressed_modal_run(
        &workload,
        &solver_mesh,
        &numerical.prestress_state,
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
        recovered,
        workload.backend(),
        modal_options.clone(),
        prepared.context.clone(),
        Some(&solver_mesh),
        Some(&modal_run_id),
    )?;

    let preload_fields = crate::analysis::collect_semantic_state_fields(&preload_run.data);
    let mut states = Vec::with_capacity(
        preload_step.procedure_step.state_outputs.len()
            + modal_step.procedure_step.state_outputs.len(),
    );
    for declaration in &preload_step.procedure_step.state_outputs {
        if declaration.state_id == numerical.prestress_state.state_id {
            states.push(numerical.prestress_state.clone());
        } else {
            states.push(
                runmat_analysis_fea::execution::CanonicalSemanticState::new(
                    runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                        producer: numerical.prestress_state.producer,
                        declaration: declaration.clone(),
                        topology_digest,
                        source_result: numerical.preload_solver_result_identity,
                        payload:
                            runmat_analysis_fea::execution::CanonicalSemanticStatePayload::Fields {
                                fields: preload_fields.clone(),
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
    }
    let modal_result_identity = numerical.modal_numerical_receipt.source_result();
    states.extend(super::modal_completion::modal_basis_states_for_completion(
        &prepared,
        &solver_mesh,
        &workload,
        modal_result_identity,
        numerical.modal_result.basis(),
    )?);

    prepared.steps[0].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralLinear(preload_options.clone());
    prepared.steps[1].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralModes(modal_options.clone());
    complete_study_run_prepared(
        prepared,
        vec![
            CompletedStudyStep {
                step_id: numerical.preload_step_id,
                solver_result_identity: numerical.preload_solver_result_identity,
                run: CompletedStudyStepRun::Standard(Box::new(preload_run)),
                resolved_solver_options: AnalysisSolverOptions::StructuralLinear(preload_options),
            },
            CompletedStudyStep {
                step_id: numerical.modal_step_id,
                solver_result_identity: modal_result_identity,
                run: CompletedStudyStepRun::Standard(Box::new(modal_run)),
                resolved_solver_options: AnalysisSolverOptions::StructuralModes(modal_options),
            },
        ],
        states,
        numerical.frozen_project_digest,
        preparation,
    )
}

pub(super) fn complete_structural_prestressed_modal(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyPrestressedModalNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [preload_step, modal_step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "prestressed-modal completion requires exactly two authored procedure steps",
        ));
    };
    let preload_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &preload_step.procedure_step.step_id,
    );
    let modal_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &modal_step.procedure_step.step_id,
    );
    let prepared = PreparedStudyCompletion {
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
    };
    complete_prestressed_modal_study(
        prepared,
        solver_mesh,
        authored_workloads,
        numerical,
        preload_run_id,
        modal_run_id,
        preparation,
    )
}
