use runmat_analysis_core::{AnalysisModel, AnalysisSequence, ProcedureStep, RestartPolicy};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::contracts::{AnalysisExecutionStep, AnalysisRunResult, AnalysisSolverOptions};
use super::{
    analysis_run_acoustic_with_options_op, analysis_run_cfd_with_options_op,
    analysis_run_cht_with_options_op, analysis_run_electromagnetic_with_options_op,
    analysis_run_fsi_with_options_op, analysis_run_linear_static_with_executor,
    analysis_run_modal_with_options_op, analysis_run_nonlinear_with_options_op,
    analysis_run_thermal_with_options_op, analysis_run_transient_with_options_op,
    StructuralLinearExecutor, ANALYSIS_RUN_STUDY_OPERATION, ANALYSIS_RUN_STUDY_OP_VERSION,
};

pub(super) fn execute_study_step(
    model: &AnalysisModel,
    procedure_step: &ProcedureStep,
    execution: &AnalysisExecutionStep,
    solver_mesh_artifact_path: Option<&str>,
    context: &OperationContext,
    structural_executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<(OperationEnvelope<AnalysisRunResult>, AnalysisSolverOptions), OperationErrorEnvelope> {
    let model = solver_model(model, procedure_step);
    match &execution.solver_options {
        AnalysisSolverOptions::ElectricalStatic(_) | AnalysisSolverOptions::Magnetostatic(_) => Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.STATIC_EM_TYPED_EXECUTION_REQUIRED",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "static electromagnetic studies require their governed typed execution path",
            std::collections::BTreeMap::new(),
        )),
        AnalysisSolverOptions::StructuralLinear(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_linear_static_with_executor(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
                structural_executor,
            )?;
            Ok((run, AnalysisSolverOptions::StructuralLinear(options)))
        }
        AnalysisSolverOptions::StructuralFrequency(_) => Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.FREQUENCY_REQUIRES_PLACEMENT",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "structural frequency studies execute through the portable placement boundary",
            std::collections::BTreeMap::from([(
                "step_id".to_string(),
                procedure_step.step_id.as_str().to_string(),
            )]),
        )),
        AnalysisSolverOptions::StructuralResponseSpectrum(_) => Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.RESPONSE_SPECTRUM_REQUIRES_SEQUENCE",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "modal response spectrum executes with its modal producer through the portable study boundary",
            std::collections::BTreeMap::from([(
                "step_id".to_string(),
                procedure_step.step_id.as_str().to_string(),
            )]),
        )),
        AnalysisSolverOptions::StructuralRandomResponse(_) => Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.RANDOM_RESPONSE_REQUIRES_SEQUENCE",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "random response executes with its modal producer through the portable study boundary",
            std::collections::BTreeMap::from([(
                "step_id".to_string(),
                procedure_step.step_id.as_str().to_string(),
            )]),
        )),
        AnalysisSolverOptions::StructuralBuckling(_) => Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.BUCKLING_REQUIRES_SEQUENCE",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "structural buckling executes with its preload through the portable study boundary",
            std::collections::BTreeMap::from([(
                "step_id".to_string(),
                procedure_step.step_id.as_str().to_string(),
            )]),
        )),
        AnalysisSolverOptions::StructuralModes(authored_options) => {
            if procedure_step.procedure
                == runmat_analysis_core::AnalysisProcedure::PrestressedModalExtraction
            {
                return Err(operation_error(
                    ANALYSIS_RUN_STUDY_OPERATION,
                    ANALYSIS_RUN_STUDY_OP_VERSION,
                    context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_STUDY.PRESTRESSED_MODAL_REQUIRES_SEQUENCE",
                        error_type: OperationErrorType::Contract,
                        retryable: false,
                        severity: OperationErrorSeverity::Fatal,
                    },
                    "prestressed modal extraction executes with its preload through the portable study boundary",
                    std::collections::BTreeMap::from([(
                        "step_id".to_string(),
                        procedure_step.step_id.as_str().to_string(),
                    )]),
                ));
            }
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_modal_with_options_op(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::StructuralModes(options)))
        }
        AnalysisSolverOptions::AcousticHelmholtz(authored_options) => {
            let options = authored_options.clone();
            let run = analysis_run_acoustic_with_options_op(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::AcousticHelmholtz(options)))
        }
        AnalysisSolverOptions::ThermalDiffusion(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_thermal_with_options_op(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::ThermalDiffusion(options)))
        }
        AnalysisSolverOptions::StructuralTransient(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_transient_with_options_op(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::StructuralTransient(options)))
        }
        AnalysisSolverOptions::IncompressibleFlow(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_cfd_with_options_op(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::IncompressibleFlow(options)))
        }
        AnalysisSolverOptions::ConjugateHeatTransfer(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_cht_with_options_op(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::ConjugateHeatTransfer(options)))
        }
        AnalysisSolverOptions::FluidStructureInteraction(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_fsi_with_options_op(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
            )?;
            Ok((
                run,
                AnalysisSolverOptions::FluidStructureInteraction(options),
            ))
        }
        AnalysisSolverOptions::StructuralNonlinear(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_nonlinear_with_options_op(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::StructuralNonlinear(options)))
        }
        AnalysisSolverOptions::ElectromagneticPotential(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_electromagnetic_with_options_op(
                &model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
            )?;
            Ok((
                run,
                AnalysisSolverOptions::ElectromagneticPotential(options),
            ))
        }
    }
}

/// Retain only complete finite host fields when publishing an in-process study state.
pub(super) fn collect_semantic_state_fields(
    run: &AnalysisRunResult,
) -> Vec<runmat_analysis_core::AnalysisField> {
    super::collect_analysis_result_fields(run)
        .into_iter()
        .filter(|field| match &field.values {
            runmat_analysis_core::AnalysisFieldValues::HostF64(values) => {
                !values.is_empty()
                    && values.iter().all(|value| value.is_finite())
                    && field
                        .shape
                        .iter()
                        .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
                        == Some(values.len())
            }
            runmat_analysis_core::AnalysisFieldValues::DeviceRef(_) => false,
        })
        .collect()
}

/// Bind an in-process study state to its admitted mesh, or its explicit synthetic model.
pub(super) fn in_process_study_topology_digest(
    prepared: &super::PreparedStudySequence,
) -> Result<runmat_execution::Digest, OperationErrorEnvelope> {
    if let Some(path) = prepared.solver_mesh_artifact_path.as_deref() {
        let mesh =
            super::solver_mesh_artifact::load_solver_mesh_artifact(std::path::Path::new(path))
                .map_err(|error| super::study_sequence_contract_error(&prepared.context, error))?;
        return Ok(runmat_execution::Digest::from_bytes(
            *mesh.canonical_digest.bytes(),
        ));
    }
    let bytes = serde_json::to_vec(&prepared.model)
        .map_err(|error| super::study_sequence_contract_error(&prepared.context, error))?;
    Ok(runmat_execution::Digest::sha256(bytes))
}

fn solver_model(model: &AnalysisModel, authored_step: &ProcedureStep) -> AnalysisModel {
    let mut step = authored_step.clone();
    step.state_inputs.clear();
    step.restart = RestartPolicy::Fresh;
    let mut projected = model.clone();
    projected.sequence = AnalysisSequence::from_steps(
        format!("{}.solver-projection", model.sequence.sequence_id),
        vec![step],
    );
    projected
}

fn attach_solver_mesh_artifact_path(target: &mut Option<String>, resolved: Option<&str>) {
    if target.is_none() {
        *target = resolved.map(str::to_string);
    }
}
