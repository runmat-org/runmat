use super::*;

pub(in crate::analysis) fn complete_prestressed_frequency_study(
    mut prepared: PreparedStudyCompletion,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyPrestressedFrequencyNumericalResult,
    preload_run_id: String,
    modal_run_id: Option<String>,
    frequency_run_id: String,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let (preload_step, modal_step, frequency_step) = match (
        &numerical.reduction,
        prepared.steps.as_slice(),
    ) {
        (
            crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::Direct,
            [preload, frequency],
        ) => (preload.clone(), None, frequency.clone()),
        (
            crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
                reduction,
            ),
            [preload, modal, frequency],
        ) if modal.procedure_step.step_id == reduction.modal_step_id => {
            (preload.clone(), Some(modal.clone()), frequency.clone())
        }
        _ => {
            return Err(completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                "prestressed-frequency completion requires preload→frequency or preload→modal→frequency steps matching its reduction",
            ))
        }
    };
    if numerical.preload_step_id != preload_step.procedure_step.step_id
        || numerical.frequency_step_id != frequency_step.procedure_step.step_id
        || numerical.frequency.step_id != numerical.frequency_step_id
        || numerical.frequency.frozen_project_digest != numerical.frozen_project_digest
    {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-frequency numerical result differs from the authored sequence",
        ));
    }
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
        workload: preload_workload,
    }) = authored_workloads.remove(&numerical.preload_step_id)
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-frequency workload is missing its preload",
        ));
    };
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency {
        template,
    }) = authored_workloads.remove(&numerical.frequency_step_id)
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-frequency workload is missing its unresolved frequency step",
        ));
    };
    let modal_template = match (&numerical.reduction, &modal_step) {
        (crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::Direct, None) => None,
        (
            crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
                reduction,
            ),
            Some(_),
        ) => match authored_workloads.remove(&reduction.modal_step_id) {
            Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal {
                template,
            }) => Some(template),
            _ => {
                return Err(completion_error(
                    &prepared,
                    "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                    "modal-reduced prestressed-frequency workload is missing its modal producer",
                ))
            }
        },
        _ => {
            return Err(completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                "prestressed-frequency reduction differs from its authored modal step",
            ))
        }
    };
    if !authored_workloads.is_empty() {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-frequency workload contains an unauthored step",
        ));
    }
    let AnalysisSolverOptions::StructuralLinear(mut preload_options) =
        preload_step.authored_execution.solver_options.clone()
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "prestressed-frequency preload has non-linear solver options",
        ));
    };
    let AnalysisSolverOptions::StructuralFrequency(mut frequency_options) =
        frequency_step.authored_execution.solver_options.clone()
    else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "prestressed-frequency consumer has non-frequency solver options",
        ));
    };
    let mut modal_options = match modal_step.as_ref() {
        Some(step) => {
            let AnalysisSolverOptions::StructuralModes(options) =
                step.authored_execution.solver_options.clone()
            else {
                return Err(completion_error(
                    &prepared,
                    "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                    "prestressed-frequency modal producer has non-modal solver options",
                ));
            };
            Some(options)
        }
        None => None,
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
    let (modal_run, modal_result_identity, basis_state_identity) = match &numerical.reduction {
        crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::Direct => {
            if modal_template.is_some() || modal_options.is_some() || modal_run_id.is_some() {
                return Err(completion_error(
                    &prepared,
                    "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                    "direct prestressed-frequency completion received modal inputs",
                ));
            }
            (None, None, None)
        }
        crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
            reduction,
        ) => {
            let admitted = super::prestressed_frequency_modal_reduction::admit_prestressed_modal_reduction(
                super::prestressed_frequency_modal_reduction::PrestressedModalReductionAdmission {
                    prepared: &prepared,
                    solver_mesh: &solver_mesh,
                    prestress_state: &numerical.prestress_state,
                    frequency_template: &template,
                    modal_template: *modal_template.ok_or_else(|| completion_error(
                        &prepared,
                        "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                        "modal-reduced prestressed-frequency completion is missing its modal template",
                    ))?,
                    modal_result: &reduction.modal_result,
                    modal_receipt: &reduction.modal_numerical_receipt,
                    basis_state: &reduction.basis_state,
                    modal_options: modal_options.as_mut().ok_or_else(|| completion_error(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        "modal-reduced prestressed-frequency completion is missing modal options",
                    ))?,
                    run_id: modal_run_id.as_deref().ok_or_else(|| completion_error(
                        &prepared,
                        "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                        "modal-reduced prestressed-frequency completion is missing its run identity",
                    ))?,
                    topology_digest,
                },
            )?;
            (
                Some(admitted.run),
                Some(admitted.result_identity),
                Some(admitted.basis_state_identity),
            )
        }
    };
    let frequency_workload = template
        .bind(transition, basis_state_identity)
        .map_err(|error| {
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
            "prestressed-frequency result differs from its resolved authored workload",
        ));
    }
    frequency_options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    let frequency_result_identity = numerical.frequency.result.manifest().identity();
    let frequency_run = match &numerical.reduction {
        crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::Direct => {
            crate::analysis::frequency_result::admit_prestressed_structural_frequency_result(
                &numerical.frequency,
                &solver_mesh,
                &frequency_options,
                frequency_run_id,
                &numerical.prestress_state,
            )
        }
        crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
            reduction,
        ) => {
            crate::analysis::frequency_result::admit_prestressed_modal_structural_frequency_result(
                &numerical.frequency,
                &solver_mesh,
                &frequency_options,
                frequency_run_id,
                &numerical.prestress_state,
                &reduction.basis_state,
            )
        }
    }
    .map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.FREQUENCY_RESULT_ADMISSION_FAILED",
            error,
        )
    })?;

    let preload_fields = crate::analysis::collect_semantic_state_fields(&preload_run.data);
    let states = super::prestressed_frequency_states::prestressed_frequency_states(
        super::prestressed_frequency_states::PrestressedFrequencyStatePublication {
            prepared: &prepared,
            preload_step: &preload_step,
            modal_step: modal_step.as_ref(),
            frequency_step: &frequency_step,
            prestress_state: &numerical.prestress_state,
            preload_result_identity: numerical.preload_solver_result_identity,
            reduction: &numerical.reduction,
            frequency_publication: &numerical.frequency.publication,
            preload_fields: &preload_fields,
            topology_digest,
            frequency_result_identity,
        },
    )?;

    prepared.steps[0].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralLinear(preload_options.clone());
    let frequency_index = if let Some(modal_options) = modal_options.as_ref() {
        prepared.steps[1].authored_execution.solver_options =
            AnalysisSolverOptions::StructuralModes(modal_options.clone());
        2
    } else {
        1
    };
    prepared.steps[frequency_index]
        .authored_execution
        .solver_options = AnalysisSolverOptions::StructuralFrequency(frequency_options.clone());
    let mut completed_steps = vec![CompletedStudyStep {
        step_id: numerical.preload_step_id,
        solver_result_identity: numerical.preload_solver_result_identity,
        run: CompletedStudyStepRun::Standard(Box::new(preload_run)),
        resolved_solver_options: AnalysisSolverOptions::StructuralLinear(preload_options),
    }];
    if let (Some(modal_step), Some(modal_run), Some(modal_result_identity), Some(modal_options)) =
        (modal_step, modal_run, modal_result_identity, modal_options)
    {
        completed_steps.push(CompletedStudyStep {
            step_id: modal_step.procedure_step.step_id,
            solver_result_identity: modal_result_identity,
            run: CompletedStudyStepRun::Standard(Box::new(modal_run)),
            resolved_solver_options: AnalysisSolverOptions::StructuralModes(modal_options),
        });
    }
    completed_steps.push(CompletedStudyStep {
        step_id: numerical.frequency_step_id,
        solver_result_identity: frequency_result_identity,
        run: CompletedStudyStepRun::StructuralFrequency(Box::new(frequency_run)),
        resolved_solver_options: AnalysisSolverOptions::StructuralFrequency(frequency_options),
    });
    complete_study_run_prepared(
        prepared,
        completed_steps,
        states,
        numerical.frozen_project_digest,
        preparation,
    )
}

pub(super) fn complete_structural_prestressed_frequency(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyPrestressedFrequencyNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let (preload_step, modal_step, frequency_step) = match (
        &numerical.reduction,
        intent.steps.as_slice(),
    ) {
        (
            crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::Direct,
            [preload, frequency],
        ) => (preload, None, frequency),
        (
            crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
                reduction,
            ),
            [preload, modal, frequency],
        ) if modal.procedure_step.step_id == reduction.modal_step_id => {
            (preload, Some(modal), frequency)
        }
        _ => {
            return Err(intent_completion_error(
                &intent,
                "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "prestressed-frequency completion requires an authored sequence matching its direct or modal-reduced result",
            ))
        }
    };
    let preload_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &preload_step.procedure_step.step_id,
    );
    let frequency_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &frequency_step.procedure_step.step_id,
    );
    let modal_run_id = modal_step.map(|step| {
        completion_identity.local_run_id(
            runmat_analysis_core::PhysicsFamily::Structural,
            intent.authored_study_digest,
            &step.procedure_step.step_id,
        )
    });
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
    complete_prestressed_frequency_study(
        prepared,
        solver_mesh,
        authored_workloads,
        numerical,
        preload_run_id,
        modal_run_id,
        frequency_run_id,
        preparation,
    )
}
