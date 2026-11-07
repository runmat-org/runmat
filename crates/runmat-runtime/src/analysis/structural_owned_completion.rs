//! Complete owned structural outcomes through the captured publication authority.
use super::*;

pub fn analysis_complete_structural_study_op(
    mut prepared: PreparedStructuralStudy,
    outcome: StructuralStudyExecutionOutcome,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let outcome = match outcome {
        Ok(result) => result,
        Err(StructuralStudyExecutionFailure::Failed(message)) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.PORTABLE_EXECUTION_FAILED",
                OperationErrorType::Backend,
                OperationErrorSeverity::Error,
                message,
            ));
        }
        Err(StructuralStudyExecutionFailure::Cancelled) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.PORTABLE_EXECUTION_CANCELLED",
                OperationErrorType::Cancelled,
                OperationErrorSeverity::Error,
                "portable structural study execution was cancelled",
            ));
        }
        Err(StructuralStudyExecutionFailure::Indeterminate(message)) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.PORTABLE_EXECUTION_INDETERMINATE",
                OperationErrorType::Internal,
                OperationErrorSeverity::Fatal,
                message,
            ));
        }
    };
    let mut preparation =
        crate::analysis::study_publication::preparation::PublicationPreparation::capture(
            analysis_study_content_digest(&prepared.study.spec),
            &(&prepared.study.model, &prepared.solver_mesh, &outcome),
        )
        .map_err(|error| {
            structural_study::publication::completion_storage_error(
                &prepared.study.context,
                "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
                error,
            )
        })?;
    preparation
        .admit_structural_sources(&outcome)
        .map_err(|error| {
            structural_study::publication::completion_storage_error(
                &prepared.study.context,
                "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
                error,
            )
        })?;
    let result = match outcome {
        StructuralStudyNumericalResult::ThermalPrestress { .. } => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "thermal prestress requires typed compiled-study publication",
            ))
        }
        StructuralStudyNumericalResult::StructuralModalFrequency(_) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "modal-frequency studies complete through typed compiled-study publication",
            ))
        }
        StructuralStudyNumericalResult::LinearSequence(result) => result,
        StructuralStudyNumericalResult::StructuralTransient(_) => {
            return Err(structural_study_completion_failure(
                &prepared,
                "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "linear structural-study completion received a direct-transient result",
            ))
        }
        StructuralStudyNumericalResult::StructuralModal(result) => {
            let workload = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralModal {
                    workload,
                } => workload.as_ref().clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "structural modal result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_modal_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workload,
                *result,
                storage::next_run_id(),
                preparation,
            )?
            .publish_legacy();
        }
        StructuralStudyNumericalResult::StructuralModalTransient(result) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralModalTransient {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "modal-transient result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_modal_transient_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
                preparation,
            )?
            .publish_legacy();
        }
        StructuralStudyNumericalResult::StructuralModalResponseSpectrum(result) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralModalResponseSpectrum {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "modal response-spectrum result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_response_spectrum_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
                preparation,
            )?
            .publish_legacy();
        }
        StructuralStudyNumericalResult::StructuralRandomResponse(result) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralRandomResponse {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "random-response result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_random_response_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
                preparation,
            )?
            .publish_legacy();
        }
        StructuralStudyNumericalResult::StructuralBuckling(result) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralBuckling {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "structural buckling result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_buckling_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
                preparation,
            )?
            .publish_legacy();
        }
        StructuralStudyNumericalResult::StructuralPrestressedModal(result) => {
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralPrestressedModal {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "prestressed-modal result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_prestressed_modal_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                storage::next_run_id(),
                preparation,
            )?
            .publish_legacy();
        }
        StructuralStudyNumericalResult::StructuralPrestressedFrequency(result) => {
            let modal_run_id = matches!(
                &result.reduction,
                StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(_)
            )
            .then(storage::next_run_id);
            let workloads = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralPrestressedFrequency {
                    workloads,
                } => workloads.clone(),
                _ => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "prestressed-frequency result differs from the prepared study kind",
                    ));
                }
            };
            return structural_study::complete_prestressed_frequency_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workloads,
                *result,
                storage::next_run_id(),
                modal_run_id,
                storage::next_run_id(),
                preparation,
            )?
            .publish_legacy();
        }
        StructuralStudyNumericalResult::StructuralFrequency(result) => {
            let workload = match &prepared.execution {
                structural_study::PreparedStructuralStudyExecution::StructuralFrequency {
                    workload,
                } => workload.as_ref().clone(),
                structural_study::PreparedStructuralStudyExecution::LinearSequence { .. } => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "linear structural-study completion received a structural-frequency result",
                    ));
                }
                structural_study::PreparedStructuralStudyExecution::StructuralModal { .. }
                | structural_study::PreparedStructuralStudyExecution::StructuralModalTransient {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralModalResponseSpectrum {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralRandomResponse {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralBuckling {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralPrestressedModal {
                    ..
                }
                | structural_study::PreparedStructuralStudyExecution::StructuralPrestressedFrequency {
                    ..
                } => {
                    return Err(structural_study_completion_failure(
                        &prepared,
                        "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                        OperationErrorType::Contract,
                        OperationErrorSeverity::Fatal,
                        "modal structural-study completion received a structural-frequency result",
                    ));
                }
            };
            return structural_study::complete_frequency_study(
                study_completion_from_prepared(prepared.study),
                prepared.solver_mesh,
                workload,
                *result,
                storage::next_run_id(),
                preparation,
            )?
            .publish_legacy();
        }
    };
    let structural_study::PreparedStructuralStudyExecution::LinearSequence { linears } =
        &mut prepared.execution
    else {
        return Err(structural_study_completion_failure(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "structural-frequency study completion received a linear sequence result",
        ));
    };
    let mut completed = Vec::with_capacity(result.steps.len());
    let effective_memory_ceiling_bytes = result.effective_memory_ceiling_bytes;
    for step in result.steps {
        let linear = linears.remove(&step.step_id).ok_or_else(|| {
            operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &prepared.study.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                    error_type: OperationErrorType::Contract,
                    retryable: false,
                    severity: OperationErrorSeverity::Fatal,
                },
                "sequence execution returned an absent or duplicate procedure step",
                BTreeMap::from([("step_id".to_string(), step.step_id.as_str().to_string())]),
            )
        })?;
        let resolved_solver_options =
            AnalysisSolverOptions::StructuralLinear(linear.options.clone());
        step.numerical_receipt
            .validate_against(step.solver_result_identity, &step.result)
            .map_err(|error| study_sequence_contract_error(&prepared.study.context, error))?;
        let original_fields =
            runmat_analysis_fea::pipeline::linear_static::original_static_displacement_source(
                &linear.workload,
                linear.solver_mesh.as_ref(),
                step.solver_result_identity,
                &step.result.run,
            )
            .map_err(|error| study_sequence_contract_error(&prepared.study.context, error))?;
        let run = analysis_admit_linear_static_run_with_id(
            linear,
            step.result.run,
            None,
            crate::analysis::StaticSolvePlacement {
                effective_memory_ceiling_bytes,
                static_solve_path: step.static_solve_path,
            },
        )?;
        completed.push(CompletedStudyStep {
            step_id: step.step_id,
            solver_result_identity: step.solver_result_identity,
            run: structural_study::publication::original_fields::standard(run, original_fields),
            resolved_solver_options,
        });
    }
    if !linears.is_empty() {
        return Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &prepared.study.context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "sequence execution omitted one or more prepared procedure steps",
            BTreeMap::new(),
        ));
    }
    structural_study::publication::complete_study_run_prepared(
        study_completion_from_prepared(prepared.study),
        completed,
        result.states,
        result.frozen_project_digest,
        preparation,
    )?
    .publish_legacy()
}

fn structural_study_completion_failure(
    prepared: &PreparedStructuralStudy,
    error_code: &'static str,
    error_type: OperationErrorType,
    severity: OperationErrorSeverity,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &prepared.study.context,
        OperationErrorSpec {
            error_code,
            error_type,
            retryable: error_type == OperationErrorType::Internal,
            severity,
        },
        message,
        BTreeMap::from([(
            "analysis_model_id".to_string(),
            prepared.study.model.model_id.0.clone(),
        )]),
    )
}
