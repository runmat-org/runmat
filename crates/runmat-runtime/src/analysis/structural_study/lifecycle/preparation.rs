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
    let mut frequency_workload = None;
    let mut heterogeneous_workloads = BTreeMap::new();
    let is_buckling_sequence = study.steps.len() == 2
        && study.steps.iter().any(|step| {
            matches!(
                step.authored_execution.solver_options,
                AnalysisSolverOptions::StructuralBuckling(_)
            )
        });
    let is_prestressed_modal_sequence = study.steps.len() == 2
        && study.steps.iter().any(|step| {
            step.procedure_step.procedure
                == runmat_analysis_core::AnalysisProcedure::PrestressedModalExtraction
        });
    let is_prestress_sequence = is_buckling_sequence || is_prestressed_modal_sequence;
    for step in &study.steps {
        let mut authored_execution = step.authored_execution.clone();
        match step.authored_execution.solver_options.clone() {
            AnalysisSolverOptions::StructuralLinear(mut options) => {
                if transient_workload.is_some() || frequency_workload.is_some() {
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
                if is_prestress_sequence {
                    heterogeneous_workloads.insert(
                        step.procedure_step.step_id.clone(),
                        runmat_analysis_fea::execution::StructuralStepWorkload::Linear {
                            workload: Box::new(workload),
                        },
                    );
                } else {
                    linear_workloads.insert(step.procedure_step.step_id.clone(), workload);
                }
            }
            AnalysisSolverOptions::StructuralTransient(mut options)
                if study.steps.len() == 1
                    && step.procedure_step.solution_space
                        == runmat_analysis_core::SolutionSpace::Direct
                    && linear_workloads.is_empty()
                    && frequency_workload.is_none()
                    && heterogeneous_workloads.is_empty() =>
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
                        solver:
                            runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct,
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
            AnalysisSolverOptions::StructuralModes(mut options)
                if is_prestressed_modal_sequence
                    && linear_workloads.is_empty()
                    && transient_workload.is_none()
                    && frequency_workload.is_none()
                    && heterogeneous_workloads.len() == 1 =>
            {
                options.solver_mesh_artifact_path = None;
                let template =
                    prepare_structural_prestressed_modal_template(&study, step, &options)?;
                heterogeneous_workloads.insert(
                    step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal {
                        template: Box::new(template),
                    },
                );
                authored_execution.solver_options = AnalysisSolverOptions::StructuralModes(options);
            }
            AnalysisSolverOptions::StructuralModes(mut options)
                if linear_workloads.is_empty()
                    && transient_workload.is_none()
                    && frequency_workload.is_none() =>
            {
                options.solver_mesh_artifact_path = None;
                let workload = prepare_structural_modal_workload(&study, step, &options)?;
                heterogeneous_workloads.insert(
                    step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
                        workload: Box::new(workload),
                    },
                );
                authored_execution.solver_options = AnalysisSolverOptions::StructuralModes(options);
            }
            AnalysisSolverOptions::StructuralTransient(mut options)
                if study.steps.len() == 2
                    && step.procedure_step.solution_space
                        == runmat_analysis_core::SolutionSpace::ModalReduced
                    && linear_workloads.is_empty()
                    && transient_workload.is_none()
                    && frequency_workload.is_none()
                    && heterogeneous_workloads.len() == 1 =>
            {
                options.solver_mesh_artifact_path = None;
                let block_plan = runmat_analysis_fea::execution::StructuralTransientBlockPlan::new(
                    options.maximum_intervals_per_block,
                )
                .map_err(|error| portable_study_contract_error(&study, error.to_string()))?;
                let template = prepare_structural_modal_transient_template(&study, step, &options)?;
                heterogeneous_workloads.insert(
                    step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::ModalTransient {
                        template: Box::new(template),
                        block_plan,
                    },
                );
                authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralTransient(options);
            }
            AnalysisSolverOptions::StructuralFrequency(mut options)
                if study.steps.len() == 1
                    && linear_workloads.is_empty()
                    && transient_workload.is_none() =>
            {
                options.solver_mesh_artifact_path = None;
                frequency_workload = Some(prepare_structural_frequency_workload(
                    &study, step, &options,
                )?);
                authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralFrequency(options);
            }
            AnalysisSolverOptions::StructuralBuckling(mut options)
                if is_buckling_sequence
                    && linear_workloads.is_empty()
                    && transient_workload.is_none()
                    && frequency_workload.is_none()
                    && heterogeneous_workloads.len() == 1 =>
            {
                options.solver_mesh_artifact_path = None;
                let template = prepare_structural_buckling_template(&study, step, &options)?;
                heterogeneous_workloads.insert(
                    step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::Buckling {
                        template: Box::new(template),
                    },
                );
                authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralBuckling(options);
            }
            _ => return Err(portable_study_kind_error(&study, step)),
        }
        steps.push(PreparedStructuralStudyIntentStep {
            authored_execution,
            procedure_step: step.procedure_step.clone(),
        });
    }
    let workloads = match (
        transient_workload,
        transient_block_plan,
        frequency_workload,
        heterogeneous_workloads.len(),
    ) {
        (Some(workload), Some(block_plan), None, 0) => {
            runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralTransient {
                workload: Box::new(workload),
                block_plan,
            }
        }
        (None, None, Some(workload), 0) => {
            runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralFrequency {
                workload: Box::new(workload),
            }
        }
        (None, None, None, 0) => {
            runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence {
                workloads: linear_workloads,
            }
        }
        (None, None, None, 1) => {
            let (_, workload) = heterogeneous_workloads.into_iter().next().ok_or_else(|| {
                portable_study_contract_error(&study, "structural modal workload is absent".into())
            })?;
            let runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
                workload,
            } = workload
            else {
                return Err(portable_study_contract_error(
                    &study,
                    "standalone heterogeneous workload is not modal extraction".into(),
                ));
            };
            runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralModal { workload }
        }
        (None, None, None, 2) => {
            runmat_analysis_fea::execution::StructuralStudyWorkloads::HeterogeneousSequence {
                workloads: heterogeneous_workloads,
            }
        }
        _ => {
            return Err(portable_study_contract_error(
                &study,
                "structural study workload kind is internally inconsistent".into(),
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

pub(in crate::analysis) fn prepare_structural_buckling_template(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
    options: &crate::analysis::AnalysisBucklingRunOptions,
) -> Result<runmat_analysis_fea::execution::StructuralBucklingTemplate, OperationErrorEnvelope> {
    options
        .solver
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    let template = runmat_analysis_fea::execution::StructuralBucklingTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.procedure_step.step_id.clone(),
        backend: step.authored_execution.backend,
        formulation_requirement: step.authored_execution.requirement,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        solver: options.solver,
    };
    template
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    Ok(template)
}

pub(in crate::analysis) fn prepare_structural_modal_workload(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
    options: &crate::analysis::AnalysisModalRunOptions,
) -> Result<runmat_analysis_fea::execution::StructuralModalWorkload, OperationErrorEnvelope> {
    runmat_analysis_fea::execution::StructuralModalWorkload::new(
        runmat_analysis_fea::execution::StructuralModalWorkloadInput {
            model: study.model.clone(),
            step_id: step.procedure_step.step_id.clone(),
            backend: step.authored_execution.backend,
            formulation_requirement: step.authored_execution.requirement,
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
            truncation: options.truncation,
            convergence: options.convergence,
            prestress_transition: None,
        },
    )
    .map_err(|error| portable_study_contract_error(study, error.to_string()))
}

pub(in crate::analysis) fn prepare_structural_prestressed_modal_template(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
    options: &crate::analysis::AnalysisModalRunOptions,
) -> Result<
    runmat_analysis_fea::execution::StructuralPrestressedModalTemplate,
    OperationErrorEnvelope,
> {
    let template = runmat_analysis_fea::execution::StructuralPrestressedModalTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.procedure_step.step_id.clone(),
        backend: step.authored_execution.backend,
        formulation_requirement: step.authored_execution.requirement,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: options.truncation,
        convergence: options.convergence,
    };
    template
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    Ok(template)
}

pub(in crate::analysis) fn prepare_structural_modal_transient_template(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
    options: &crate::analysis::AnalysisTransientRunOptions,
) -> Result<runmat_analysis_fea::execution::StructuralModalTransientTemplate, OperationErrorEnvelope>
{
    let template = runmat_analysis_fea::execution::StructuralModalTransientTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.procedure_step.step_id.clone(),
        backend: step.authored_execution.backend,
        formulation_requirement: step.authored_execution.requirement,
        time_step_policy: runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: options.time_step_policy.interval_count(),
        },
        residual_target: options.residual_target,
    };
    template
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    Ok(template)
}

pub(in crate::analysis) fn prepare_structural_frequency_workload(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
    options: &crate::analysis::AnalysisStructuralFrequencyRunOptions,
) -> Result<runmat_analysis_fea::execution::StructuralFrequencyWorkload, OperationErrorEnvelope> {
    options
        .direct_solver
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    let runmat_analysis_core::IndependentDomain::FrequencyHertz(frequencies) =
        &step.procedure_step.independent_domain
    else {
        return Err(portable_study_contract_error(
            study,
            "structural harmonic execution requires a typed frequency domain".into(),
        ));
    };
    let (&reference_frequency_hz, requested_frequencies_hz) =
        frequencies.split_first().ok_or_else(|| {
            portable_study_contract_error(
                study,
                "structural harmonic execution requires at least one frequency".into(),
            )
        })?;
    let plan = runmat_analysis_fea::frequency::FrequencyResponsePlan::new(
        reference_frequency_hz,
        requested_frequencies_hz,
        &[],
    )
    .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    runmat_analysis_fea::execution::StructuralFrequencyWorkload::new(
        runmat_analysis_fea::execution::StructuralFrequencyWorkloadInput {
            model: study.model.clone(),
            step_id: step.procedure_step.step_id.clone(),
            backend: step.authored_execution.backend,
            formulation_requirement: step.authored_execution.requirement,
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
            plan,
            solver: runmat_analysis_fea::execution::StructuralFrequencySolverPlan::Direct {
                policy: options.direct_solver,
            },
            prestress_transition: None,
        },
    )
    .map_err(|error| portable_study_contract_error(study, error.to_string()))
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
        "portable structural studies require a structural-linear sequence, direct transient or harmonic response, modal extraction, or an authored modal-to-transient sequence",
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
