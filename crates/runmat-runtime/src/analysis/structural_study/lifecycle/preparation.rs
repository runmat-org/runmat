use super::*;
use crate::analysis::meshing;
use crate::analysis::storage;
mod errors;
mod transient;
use errors::{portable_study_contract_error, portable_study_kind_error};
pub(in crate::analysis) use transient::prepare_structural_modal_transient_template;

pub(in crate::analysis) fn complete_response_spectrum_modal_options(
    model: &runmat_analysis_core::AnalysisModel,
    response_step: &runmat_analysis_core::ProcedureStep,
    options: &mut crate::analysis::AnalysisModalRunOptions,
) -> Result<(), String> {
    let runmat_analysis_core::IndependentDomain::ResponseSpectra { spectrum_ids } =
        &response_step.independent_domain
    else {
        return Err("modal response-spectrum step has no response-spectrum domain".into());
    };
    for spectrum_id in spectrum_ids {
        let spectrum = model
            .response_spectra
            .iter()
            .find(|candidate| candidate.spectrum_id == *spectrum_id)
            .ok_or_else(|| {
                "modal response-spectrum step references an absent spectrum".to_string()
            })?;
        if spectrum.truncation.correction
            == runmat_analysis_core::ModalTruncationCorrection::ResidualMode
            && !options
                .residual_vector_directions
                .contains(&spectrum.direction)
        {
            options
                .residual_vector_directions
                .push(spectrum.direction.clone());
        }
    }
    Ok(())
}

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
    let is_prestressed_modal_sequence = matches!(study.steps.len(), 2 | 3)
        && study.steps.iter().any(|step| {
            step.procedure_step.procedure
                == runmat_analysis_core::AnalysisProcedure::PrestressedModalExtraction
        });
    let is_prestressed_frequency_sequence = matches!(study.steps.len(), 2 | 3)
        && study.steps.iter().any(|step| {
            step.procedure_step.procedure
                == runmat_analysis_core::AnalysisProcedure::PrestressedHarmonicResponse
        });
    let response_spectrum_step = study.steps.iter().find(|step| {
        step.procedure_step.procedure
            == runmat_analysis_core::AnalysisProcedure::ModalResponseSpectrum
    });
    let is_modal_response_spectrum_sequence = study.steps.len() == 2
        && response_spectrum_step.is_some()
        && study.steps.iter().any(|step| {
            step.procedure_step.procedure
                == runmat_analysis_core::AnalysisProcedure::ModalExtraction
        });
    let is_random_response_sequence = study.steps.len() == 2
        && study.steps.iter().any(|step| {
            step.procedure_step.procedure == runmat_analysis_core::AnalysisProcedure::RandomResponse
        })
        && study.steps.iter().any(|step| {
            step.procedure_step.procedure
                == runmat_analysis_core::AnalysisProcedure::ModalExtraction
        });
    let is_prestress_sequence =
        is_buckling_sequence || is_prestressed_modal_sequence || is_prestressed_frequency_sequence;
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
                        schema_version: 2,
                        initial_fields: storage::resolve_initial_fields(
                            &study.model,
                            &step.procedure_step,
                            step.procedure_step
                                .accuracy
                                .resources
                                .resident_byte_budget(),
                        )
                        .map_err(|error| {
                            portable_study_contract_error(&study, error.to_string())
                        })?,
                        model: study.model.clone(),
                        step_id: step.procedure_step.step_id.clone(),
                        backend: step.authored_execution.backend,
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
                if is_modal_response_spectrum_sequence {
                    let response_step = response_spectrum_step.ok_or_else(|| {
                        portable_study_contract_error(
                            &study,
                            "modal response-spectrum consumer is absent".into(),
                        )
                    })?;
                    complete_response_spectrum_modal_options(
                        &study.model,
                        &response_step.procedure_step,
                        &mut options,
                    )
                    .map_err(|error| portable_study_contract_error(&study, error))?;
                }
                let workload = prepare_structural_modal_workload(&study, step, &options)?;
                heterogeneous_workloads.insert(
                    step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
                        workload: Box::new(workload),
                    },
                );
                authored_execution.solver_options = AnalysisSolverOptions::StructuralModes(options);
            }
            AnalysisSolverOptions::StructuralResponseSpectrum(mut options)
                if is_modal_response_spectrum_sequence
                    && linear_workloads.is_empty()
                    && transient_workload.is_none()
                    && frequency_workload.is_none()
                    && heterogeneous_workloads.len() == 1 =>
            {
                options.solver_mesh_artifact_path = None;
                let template =
                    prepare_structural_response_spectrum_template(&study, step, &options)?;
                heterogeneous_workloads.insert(
                    step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::ModalResponseSpectrum {
                        template: Box::new(template),
                    },
                );
                authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralResponseSpectrum(options);
            }
            AnalysisSolverOptions::StructuralRandomResponse(mut options)
                if is_random_response_sequence
                    && linear_workloads.is_empty()
                    && transient_workload.is_none()
                    && frequency_workload.is_none()
                    && heterogeneous_workloads.len() == 1 =>
            {
                options.solver_mesh_artifact_path = None;
                let template = prepare_structural_random_response_template(&study, step, &options)?;
                heterogeneous_workloads.insert(
                    step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::RandomResponse {
                        template: Box::new(template),
                    },
                );
                authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralRandomResponse(options);
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
                if is_prestressed_frequency_sequence
                    && linear_workloads.is_empty()
                    && transient_workload.is_none()
                    && frequency_workload.is_none()
                    && ((step.procedure_step.solution_space
                        == runmat_analysis_core::SolutionSpace::Direct
                        && heterogeneous_workloads.len() == 1)
                        || (step.procedure_step.solution_space
                            == runmat_analysis_core::SolutionSpace::ModalReduced
                            && heterogeneous_workloads.len() == 2)) =>
            {
                options.solver_mesh_artifact_path = None;
                let template =
                    prepare_structural_prestressed_frequency_template(&study, step, &options)?;
                heterogeneous_workloads.insert(
                    step.procedure_step.step_id.clone(),
                    runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency {
                        template: Box::new(template),
                    },
                );
                authored_execution.solver_options =
                    AnalysisSolverOptions::StructuralFrequency(options);
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
        (None, None, None, 2 | 3) => {
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
        geometry_source_digest,
        geometry_unit_evidence: study.spec.geometry.source.unit_evidence.clone(),
        meshing_request: super::super::resolve_intent_meshing_request(
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
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
            truncation: options.truncation,
            convergence: options.convergence,
            residual_vector_directions: options.residual_vector_directions.clone(),
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
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: options.truncation,
        convergence: options.convergence,
        residual_vector_directions: options.residual_vector_directions.clone(),
    };
    template
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    Ok(template)
}

pub(in crate::analysis) fn prepare_structural_response_spectrum_template(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
    options: &crate::analysis::AnalysisResponseSpectrumRunOptions,
) -> Result<
    runmat_analysis_fea::execution::StructuralModalResponseSpectrumTemplate,
    OperationErrorEnvelope,
> {
    if options.precision_mode != crate::analysis::PrecisionMode::Fp64
        || options.solver_mesh_artifact_path.is_some()
    {
        return Err(portable_study_contract_error(
            study,
            "modal response spectrum currently requires fp64 and a path-free prepared workload"
                .into(),
        ));
    }
    let template = runmat_analysis_fea::execution::StructuralModalResponseSpectrumTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.procedure_step.step_id.clone(),
        backend: step.authored_execution.backend,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
    };
    template
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    Ok(template)
}

pub(in crate::analysis) fn prepare_structural_random_response_template(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
    options: &crate::analysis::AnalysisRandomResponseRunOptions,
) -> Result<runmat_analysis_fea::execution::StructuralRandomResponseTemplate, OperationErrorEnvelope>
{
    if options.precision_mode != crate::analysis::PrecisionMode::Fp64
        || options.maximum_intervals_per_task == 0
        || options.solver_mesh_artifact_path.is_some()
    {
        return Err(portable_study_contract_error(
            study,
            "random response requires fp64, nonzero interval tasks, and a path-free prepared workload"
                .into(),
        ));
    }
    let template = runmat_analysis_fea::execution::StructuralRandomResponseTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.procedure_step.step_id.clone(),
        backend: step.authored_execution.backend,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        integration_policy: options.integration,
        maximum_intervals_per_task: options.maximum_intervals_per_task,
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
    let crate::analysis::AnalysisStructuralFrequencySolver::Direct { policy } = options.solver
    else {
        return Err(portable_study_contract_error(
            study,
            "standalone structural frequency execution requires a direct solver".into(),
        ));
    };
    policy
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    let plan = prepare_structural_frequency_plan(study, step)?;
    runmat_analysis_fea::execution::StructuralFrequencyWorkload::new(
        runmat_analysis_fea::execution::StructuralFrequencyWorkloadInput {
            model: study.model.clone(),
            step_id: step.procedure_step.step_id.clone(),
            backend: step.authored_execution.backend,
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
            plan,
            solver: runmat_analysis_fea::execution::StructuralFrequencySolverPlan::Direct {
                policy,
            },
            prestress_transition: None,
        },
    )
    .map_err(|error| portable_study_contract_error(study, error.to_string()))
}

pub(in crate::analysis) fn prepare_structural_prestressed_frequency_template(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
    options: &crate::analysis::AnalysisStructuralFrequencyRunOptions,
) -> Result<
    runmat_analysis_fea::execution::StructuralPrestressedFrequencyTemplate,
    OperationErrorEnvelope,
> {
    let solver = match (step.procedure_step.solution_space, options.solver) {
        (
            runmat_analysis_core::SolutionSpace::Direct,
            crate::analysis::AnalysisStructuralFrequencySolver::Direct { policy },
        ) => {
            policy
                .validate()
                .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
            runmat_analysis_fea::execution::StructuralPrestressedFrequencySolverTemplate::Direct {
                policy,
            }
        }
        (
            runmat_analysis_core::SolutionSpace::ModalReduced,
            crate::analysis::AnalysisStructuralFrequencySolver::ModalReduced,
        ) => {
            runmat_analysis_fea::execution::StructuralPrestressedFrequencySolverTemplate::ModalReduced
        }
        _ => {
            return Err(portable_study_contract_error(
                study,
                "structural frequency solver method differs from the authored solution space"
                    .into(),
            ))
        }
    };
    let template = runmat_analysis_fea::execution::StructuralPrestressedFrequencyTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.procedure_step.step_id.clone(),
        backend: step.authored_execution.backend,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        plan: prepare_structural_frequency_plan(study, step)?,
        solver,
    };
    template
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    Ok(template)
}

fn prepare_structural_frequency_plan(
    study: &super::super::PreparedStudySequence,
    step: &super::super::PreparedStudyStep,
) -> Result<runmat_analysis_fea::frequency::FrequencyResponsePlan, OperationErrorEnvelope> {
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
    runmat_analysis_fea::frequency::FrequencyResponsePlan::new(
        reference_frequency_hz,
        requested_frequencies_hz,
        &[],
    )
    .map_err(|error| portable_study_contract_error(study, error.to_string()))
}
