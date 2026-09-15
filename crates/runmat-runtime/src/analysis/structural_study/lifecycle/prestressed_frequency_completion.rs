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
    frequency_run_id: String,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let [preload_step, frequency_step] = prepared.steps.as_slice() else {
        return Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "prestressed-frequency completion requires one preload and one frequency step",
        ));
    };
    let preload_step = preload_step.clone();
    let frequency_step = frequency_step.clone();
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
        numerical.preload_result,
        Some(&preload_run_id),
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
    let frequency_workload = template.bind(transition, None).map_err(|error| {
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
    let frequency_run =
        crate::analysis::frequency_result::admit_prestressed_structural_frequency_result(
            &numerical.frequency,
            &solver_mesh,
            &frequency_options,
            frequency_run_id,
            &numerical.prestress_state,
        )
        .map_err(|error| {
            completion_error(
                &prepared,
                "RM.FEA.RUN_STUDY.FREQUENCY_RESULT_ADMISSION_FAILED",
                error,
            )
        })?;

    let preload_fields = crate::analysis::collect_semantic_state_fields(&preload_run.data);
    let mut states = Vec::with_capacity(
        preload_step.procedure_step.state_outputs.len()
            + frequency_step.procedure_step.state_outputs.len(),
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
    let sequence_identity = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &prepared.model.sequence,
    )
    .map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence_identity,
        &frequency_step.procedure_step,
    )
    .map_err(|error| {
        completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    states.extend(
        frequency_step
            .procedure_step
            .state_outputs
            .iter()
            .map(|declaration| {
                runmat_analysis_fea::execution::CanonicalSemanticState::new(
                    runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                        producer,
                        declaration: declaration.clone(),
                        topology_digest,
                        source_result: frequency_result_identity,
                        payload: runmat_analysis_fea::execution::CanonicalSemanticStatePayload::StructuralFrequency {
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
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    );

    prepared.steps[0].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralLinear(preload_options.clone());
    prepared.steps[1].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralFrequency(frequency_options.clone());
    complete_study_run(
        prepared,
        vec![
            CompletedStudyStep {
                step_id: numerical.preload_step_id,
                solver_result_identity: numerical.preload_solver_result_identity,
                run: CompletedStudyStepRun::Standard(Box::new(preload_run)),
                resolved_solver_options: AnalysisSolverOptions::StructuralLinear(preload_options),
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
    )
}

pub(super) fn complete_structural_prestressed_frequency(
    intent: PreparedStructuralStudyIntent,
    resolved_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyPrestressedFrequencyNumericalResult,
    completion_identity: StructuralStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let [preload_step, frequency_step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "prestressed-frequency completion requires exactly two authored procedure steps",
        ));
    };
    let preload_run_id = completion_identity.local_run_id(
        intent.authored_study_digest,
        &preload_step.procedure_step.step_id,
    );
    let frequency_run_id = completion_identity.local_run_id(
        intent.authored_study_digest,
        &frequency_step.procedure_step.step_id,
    );
    let prepared = PreparedStudyCompletion {
        context: intent.context,
        study_id: intent.study_id,
        model_template: intent.model_template,
        study_fingerprint: study_fingerprint_from_digest(intent.authored_study_digest),
        authored_study_digest: intent.authored_study_digest,
        model: resolved_model,
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
        frequency_run_id,
    )
}
