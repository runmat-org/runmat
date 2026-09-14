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
        AnalysisSolverOptions::StructuralModes(authored_options) => {
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
