use runmat_analysis_core::{AnalysisModel, AnalysisSequence, ProcedureStep, RestartPolicy};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::contracts::{AnalysisExecutionStep, AnalysisRunResult, AnalysisSolverOptions};
use super::{
    execute_acoustic_with_options, execute_cfd_with_options, execute_modal_with_options,
    execute_nonlinear_with_options, execute_transient_with_options,
    linear_static_execution::execute_linear_static_with_capacity,
    thermal_execution::execute_thermal_with_options, StructuralLinearExecutor,
    ANALYSIS_RUN_STUDY_OPERATION, ANALYSIS_RUN_STUDY_OP_VERSION,
};

/// Runs one authored step with its accelerator admission policy in scope, so every family solve
/// it starts admits placement through the shared `accelerator_admission` decision.
pub(super) fn execute_study_step(
    model: &AnalysisModel,
    procedure_step: &ProcedureStep,
    execution: &AnalysisExecutionStep,
    solver_mesh_artifact_path: Option<&str>,
    context: &OperationContext,
    structural_executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<(OperationEnvelope<AnalysisRunResult>, AnalysisSolverOptions), OperationErrorEnvelope> {
    execute_study_step_with_policy(
        model,
        procedure_step,
        execution,
        solver_mesh_artifact_path,
        context,
        structural_executor,
        false,
    )
}

pub(super) fn execute_study_step_with_policy(
    model: &AnalysisModel,
    procedure_step: &ProcedureStep,
    execution: &AnalysisExecutionStep,
    solver_mesh_artifact_path: Option<&str>,
    context: &OperationContext,
    structural_executor: Option<&dyn StructuralLinearExecutor>,
    require_accelerator: bool,
) -> Result<(OperationEnvelope<AnalysisRunResult>, AnalysisSolverOptions), OperationErrorEnvelope> {
    use futures::FutureExt;
    execute_study_step_with_capacity(
        model,
        procedure_step,
        execution,
        solver_mesh_artifact_path,
        context,
        structural_executor,
        require_accelerator,
        None,
    )
    .now_or_never()
    .expect("the synchronous study host does not await capacity")
}

pub(super) async fn execute_study_step_with_capacity(
    model: &AnalysisModel,
    procedure_step: &ProcedureStep,
    execution: &AnalysisExecutionStep,
    solver_mesh_artifact_path: Option<&str>,
    context: &OperationContext,
    structural_executor: Option<&dyn StructuralLinearExecutor>,
    require_accelerator: bool,
    capacity: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner>,
) -> Result<(OperationEnvelope<AnalysisRunResult>, AnalysisSolverOptions), OperationErrorEnvelope> {
    let policy = if require_accelerator
        || crate::context::current_acceleration_policy()
            .is_some_and(|policy| !policy.allow_host_fallback)
    {
        runmat_analysis_fea::accelerator_admission::AcceleratorAdmissionPolicy::Required
    } else {
        execution.accelerator_admission
    };
    let future = execute_admitted_study_step(
        model,
        procedure_step,
        execution,
        solver_mesh_artifact_path,
        context,
        structural_executor,
        capacity,
    );
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|cx| {
        runmat_analysis_fea::accelerator_admission::with_accelerator_admission_policy(
            policy,
            || std::future::Future::poll(future.as_mut(), cx),
        )
    })
    .await
}

async fn execute_admitted_study_step(
    model: &AnalysisModel,
    procedure_step: &ProcedureStep,
    execution: &AnalysisExecutionStep,
    solver_mesh_artifact_path: Option<&str>,
    context: &OperationContext,
    structural_executor: Option<&dyn StructuralLinearExecutor>,
    capacity: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner>,
) -> Result<(OperationEnvelope<AnalysisRunResult>, AnalysisSolverOptions), OperationErrorEnvelope> {
    let model = solver_model(model, procedure_step);
    match &execution.solver_options {
        AnalysisSolverOptions::AcousticResponse(_) => Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.ACOUSTIC_TYPED_EXECUTION_REQUIRED",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "acoustic cavity-mode and transient studies run through their governed typed execution path",
            std::collections::BTreeMap::new(),
        )),
        AnalysisSolverOptions::ElectricalStatic(_) | AnalysisSolverOptions::Magnetostatic(_) | AnalysisSolverOptions::MaxwellHarmonic(_) | AnalysisSolverOptions::LineParameters(_) => Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.STATIC_EM_TYPED_EXECUTION_REQUIRED",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "electromagnetic studies require their governed typed execution path",
            std::collections::BTreeMap::new(),
        )),
        AnalysisSolverOptions::StructuralLinear(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = execute_linear_static_with_capacity(
                &model,
                execution.backend,
                options.clone(),
                context.clone(),
                structural_executor,
                capacity,
            ).await?;
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
            let run = execute_modal_with_options(
                &model,
                execution.backend,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::StructuralModes(options)))
        }
        AnalysisSolverOptions::AcousticHelmholtz(authored_options) => {
            let options = authored_options.clone();
            let run = execute_acoustic_with_options(
                &model,
                execution.backend,
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
            let run = execute_thermal_with_options(
                &model,
                execution.backend,
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
            let run = execute_transient_with_options(
                &model,
                execution.backend,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::StructuralTransient(options)))
        }
        AnalysisSolverOptions::IncompressibleFlow(authored_options) => {
            let options = authored_options.clone();
            let run = execute_cfd_with_options(
                &model,
                execution.backend,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::IncompressibleFlow(options)))
        }
        AnalysisSolverOptions::StructuralNonlinear(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = execute_nonlinear_with_options(
                &model,
                execution.backend,
                options.clone(),
                context.clone(),
            )?;
            Ok((run, AnalysisSolverOptions::StructuralNonlinear(options)))
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
