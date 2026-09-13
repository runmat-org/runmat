use super::*;

/// Resolves and validates structural study semantics without running meshing or loading a solver
/// mesh artifact. The returned intent is path-free and can be checkpointed by a remote study
/// coordinator before geometry admission begins.
pub fn analysis_prepare_structural_study_intent_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<PreparedStructuralStudyIntent, OperationErrorEnvelope> {
    let study = prepare_study_semantics(spec, context)?;
    let mut steps = Vec::with_capacity(study.steps.len());
    let mut linear_workloads = BTreeMap::new();
    let mut transient_workload = None;
    let mut transient_block_plan = None;
    for step in &study.steps {
        let mut authored_execution = step.authored_execution.clone();
        match step.authored_execution.solver_options.clone() {
            AnalysisSolverOptions::StructuralLinear(mut options) => {
                if transient_workload.is_some() {
                    return Err(portable_study_kind_error(&study, step));
                }
                options.solver_mesh_artifact_path = None;
                let workload = prepare_linear_static_workload_template(
                    &study.model,
                    &step.procedure_step.step_id,
                    step.authored_execution.backend,
                    step.authored_execution.requirement,
                    &options,
                )?;
                authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralLinear(options);
                linear_workloads.insert(step.procedure_step.step_id.clone(), workload);
            }
            AnalysisSolverOptions::StructuralTransient(mut options)
                if study.steps.len() == 1 && linear_workloads.is_empty() =>
            {
                options.solver_mesh_artifact_path = None;
                let interval_count = options.time_step_policy.interval_count();
                transient_block_plan = Some(
                    runmat_analysis_fea::execution::StructuralTransientBlockPlan::new(
                        options.maximum_intervals_per_block,
                    )
                    .map_err(|error| portable_study_contract_error(&study, error.to_string()))?,
                );
                transient_workload = Some(
                    runmat_analysis_fea::execution::StructuralTransientWorkload {
                        schema_version: 1,
                        model: study.model.clone(),
                        step_id: step.procedure_step.step_id.clone(),
                        backend: step.authored_execution.backend,
                        formulation_requirement: step.authored_execution.requirement,
                        time_step_policy:
                            runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy::Fixed {
                                interval_count,
                            },
                        residual_target: options.residual_target,
                    },
                );
                authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralTransient(options);
            }
            _ => return Err(portable_study_kind_error(&study, step)),
        }
        steps.push(PreparedStructuralStudyIntentStep {
            authored_execution,
            procedure_step: step.procedure_step.clone(),
        });
    }
    let workloads = match (transient_workload, transient_block_plan) {
        (Some(workload), Some(block_plan)) => {
            runmat_analysis_fea::execution::StructuralStudyWorkloads::DirectTransient {
                workload: Box::new(workload),
                block_plan,
            }
        }
        (None, None) => runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence {
            workloads: linear_workloads,
        },
        _ => {
            return Err(portable_study_contract_error(
                &study,
                "structural transient workload and block plan are inconsistent".into(),
            ))
        }
    };
    let geometry_source_digest =
        crate::meshing::parse_source_sha256(&study.spec.geometry.source.sha256)
            .map(runmat_execution::Digest::from_bytes)
            .map_err(|message| {
                operation_error(
                    ANALYSIS_RUN_STUDY_OPERATION,
                    ANALYSIS_RUN_STUDY_OP_VERSION,
                    &study.context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                        error_type: OperationErrorType::Input,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    message,
                    BTreeMap::from([("study_id".to_string(), study.spec.study_id.clone())]),
                )
            })?;
    let mut intent = PreparedStructuralStudyIntent {
        study_id: study.spec.study_id.clone(),
        model_template: study.spec.create_model_intent.template,
        geometry_source_digest,
        geometry_units: study.spec.geometry.units,
        meshing_settings: study.spec.meshing_settings.clone(),
        meshing_request: super::super::resolve_intent_meshing_request(
            study.spec.geometry.units,
            study.spec.meshing_settings.clone(),
        )
        .map_err(|message| {
            operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &study.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                message,
                BTreeMap::from([("study_id".to_string(), study.spec.study_id.clone())]),
            )
        })?,
        meshing_contacts: meshing::meshing_contact_definitions_for_model(&study.model).collect(),
        context: study.context,
        authored_study_digest: analysis_study_content_digest(spec),
        intent_digest: runmat_execution::Digest::from_bytes([0_u8; 32]),
        model: study.model,
        steps,
        workloads,
        operation_sequence: study.operation_sequence,
    };
    intent.intent_digest = intent.semantic_digest().map_err(|message| {
        operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &intent.context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            message,
            BTreeMap::from([("study_id".to_string(), intent.study_id.clone())]),
        )
    })?;
    intent.validate().map_err(|message| {
        operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &intent.context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            message,
            BTreeMap::from([("study_id".to_string(), intent.study_id.clone())]),
        )
    })?;
    Ok(intent)
}

fn portable_study_kind_error(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &study.context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.PORTABLE_FAMILY_UNSUPPORTED",
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        "portable structural studies require either a structural-linear sequence or one direct structural-transient step",
        BTreeMap::from([
            ("study_id".to_string(), study.spec.study_id.clone()),
            (
                "step_id".to_string(),
                step.procedure_step.step_id.as_str().to_string(),
            ),
        ]),
    )
}

fn portable_study_contract_error(
    study: &super::super::PreparedStudySequence,
    message: String,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &study.context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        message,
        BTreeMap::from([("study_id".to_string(), study.spec.study_id.clone())]),
    )
}
