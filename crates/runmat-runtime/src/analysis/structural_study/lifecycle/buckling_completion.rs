use super::*;

pub(in crate::analysis) fn complete_buckling_study(
    mut prepared: PreparedStudyCompletion,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyBucklingNumericalResult,
    preload_run_id: String,
    buckling_run_id: String,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [preload_step, buckling_step] = prepared.steps.as_slice() else {
        return Err(buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "buckling completion requires exactly one preload and one buckling step",
        ));
    };
    let preload_step = preload_step.clone();
    let buckling_step = buckling_step.clone();
    if numerical.preload_step_id != preload_step.procedure_step.step_id
        || numerical.buckling_step_id != buckling_step.procedure_step.step_id
    {
        return Err(buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "buckling numerical result step order differs from the authored sequence",
        ));
    }
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
        workload: preload_workload,
    }) = authored_workloads.remove(&numerical.preload_step_id)
    else {
        return Err(buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "buckling workload is missing its structural preload",
        ));
    };
    let Some(runmat_analysis_fea::execution::StructuralStepWorkload::Buckling { template }) =
        authored_workloads.remove(&numerical.buckling_step_id)
    else {
        return Err(buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "buckling workload is missing its unresolved buckling step",
        ));
    };
    if !authored_workloads.is_empty() {
        return Err(buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "buckling workload contains an unauthored step",
        ));
    }
    let AnalysisSolverOptions::StructuralLinear(mut preload_options) =
        preload_step.authored_execution.solver_options.clone()
    else {
        return Err(buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "buckling preload has non-linear solver options",
        ));
    };
    let AnalysisSolverOptions::StructuralBuckling(mut buckling_options) =
        buckling_step.authored_execution.solver_options.clone()
    else {
        return Err(buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "buckling consumer has non-buckling solver options",
        ));
    };

    let mut preload_workload = *preload_workload;
    preload_workload.model = prepared.model.clone();
    preload_workload
        .validate_sequence_template()
        .map_err(|error| {
            buckling_completion_error(
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
    let preload_fields = crate::analysis::collect_semantic_state_fields(&preload_run.data);
    numerical.prestress_state.validate().map_err(|error| {
        buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.PRESTRESS_STATE_INVALID",
            error.to_string(),
        )
    })?;
    if numerical.prestress_state.source_result != numerical.preload_solver_result_identity {
        return Err(buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.PRESTRESS_STATE_INVALID",
            "prestress state does not identify the admitted preload result",
        ));
    }

    let mut template = *template;
    template.model = prepared.model.clone();
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &prepared.model,
        &template.step_id,
        &numerical.prestress_state,
        runmat_analysis_core::StateUse::Prestress,
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes()),
    )
    .map_err(|error| {
        buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.PRESTRESS_STATE_INVALID",
            error.to_string(),
        )
    })?;
    let workload = template.bind(transition).map_err(|error| {
        buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.BUCKLING_WORKLOAD_INVALID",
            error.to_string(),
        )
    })?;
    let recovered = runmat_analysis_fea::recover_completed_structural_buckling_run(
        &workload,
        &solver_mesh,
        &numerical.prestress_state,
        &numerical.buckling_result,
        &numerical.buckling_numerical_receipt,
    )
    .map_err(|error| {
        buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.BUCKLING_RESULT_ADMISSION_FAILED",
            error.to_string(),
        )
    })?;
    buckling_options.solver_mesh_artifact_path = prepared.solver_mesh_artifact_path.clone();
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let buckling_run = analysis_admit_buckling_run_with_id(
        recovered,
        buckling_options.clone(),
        prepared.context.clone(),
        solver_mesh,
        &buckling_run_id,
    )?;
    let buckling_fields = crate::analysis::collect_semantic_state_fields(&buckling_run.data);
    let sequence = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &prepared.model.sequence,
    )
    .map_err(|error| {
        buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let buckling_producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence,
        &buckling_step.procedure_step,
    )
    .map_err(|error| {
        buckling_completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let buckling_result_identity = numerical.buckling_numerical_receipt.source_result();
    let mut states = Vec::with_capacity(
        preload_step.procedure_step.state_outputs.len()
            + buckling_step.procedure_step.state_outputs.len(),
    );
    for declaration in &preload_step.procedure_step.state_outputs {
        if declaration.state_id == numerical.prestress_state.state_id {
            states.push(numerical.prestress_state.clone());
            continue;
        }
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
                buckling_completion_error(
                    &prepared,
                    "RM.FEA.RUN_STUDY.STATE_INVALID",
                    error.to_string(),
                )
            })?,
        );
    }
    for declaration in &buckling_step.procedure_step.state_outputs {
        states.push(
            runmat_analysis_fea::execution::CanonicalSemanticState::new(
                runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                    producer: buckling_producer,
                    declaration: declaration.clone(),
                    topology_digest,
                    source_result: buckling_result_identity,
                    payload:
                        runmat_analysis_fea::execution::CanonicalSemanticStatePayload::Fields {
                            fields: buckling_fields.clone(),
                        },
                },
            )
            .map_err(|error| {
                buckling_completion_error(
                    &prepared,
                    "RM.FEA.RUN_STUDY.STATE_INVALID",
                    error.to_string(),
                )
            })?,
        );
    }

    prepared.steps[0].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralLinear(preload_options.clone());
    prepared.steps[1].authored_execution.solver_options =
        AnalysisSolverOptions::StructuralBuckling(buckling_options.clone());
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
                step_id: numerical.buckling_step_id,
                solver_result_identity: buckling_result_identity,
                run: CompletedStudyStepRun::Standard(Box::new(buckling_run)),
                resolved_solver_options: AnalysisSolverOptions::StructuralBuckling(
                    buckling_options,
                ),
            },
        ],
        states,
        numerical.frozen_project_digest,
        preparation,
    )
}

pub(super) fn complete_structural_buckling(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: crate::analysis::StructuralStudyBucklingNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [preload_step, buckling_step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "buckling completion requires exactly one preload and one buckling step",
        ));
    };
    let preload_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &preload_step.procedure_step.step_id,
    );
    let buckling_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &buckling_step.procedure_step.step_id,
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
    complete_buckling_study(
        prepared,
        solver_mesh,
        authored_workloads,
        numerical,
        preload_run_id,
        buckling_run_id,
        preparation,
    )
}

fn buckling_completion_error(
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
