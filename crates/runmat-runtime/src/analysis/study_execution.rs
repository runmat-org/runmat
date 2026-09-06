use runmat_analysis_core::AnalysisModel;

use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};

use super::contracts::{AnalysisExecutionStep, AnalysisRunResult, AnalysisSolverOptions};
use super::{
    analysis_run_acoustic_with_options_op, analysis_run_cfd_with_options_op,
    analysis_run_cht_with_options_op, analysis_run_electromagnetic_with_options_op,
    analysis_run_fsi_with_options_op, analysis_run_linear_static_with_executor,
    analysis_run_modal_with_options_op, analysis_run_nonlinear_with_options_op,
    analysis_run_thermal_with_options_op, analysis_run_transient_with_options_op,
    StructuralLinearExecutor,
};

pub(super) fn execute_study_step(
    model: &AnalysisModel,
    execution: &AnalysisExecutionStep,
    solver_mesh_artifact_path: Option<&str>,
    context: &OperationContext,
    structural_executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<(OperationEnvelope<AnalysisRunResult>, AnalysisSolverOptions), OperationErrorEnvelope> {
    match &execution.solver_options {
        AnalysisSolverOptions::StructuralLinear(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_linear_static_with_executor(
                model,
                execution.backend,
                execution.requirement,
                options.clone(),
                context.clone(),
                structural_executor,
            )?;
            Ok((run, AnalysisSolverOptions::StructuralLinear(options)))
        }
        AnalysisSolverOptions::StructuralModes(authored_options) => {
            let mut options = authored_options.clone();
            attach_solver_mesh_artifact_path(
                &mut options.solver_mesh_artifact_path,
                solver_mesh_artifact_path,
            );
            let run = analysis_run_modal_with_options_op(
                model,
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
                model,
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
                model,
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
                model,
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
                model,
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
                model,
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
                model,
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
                model,
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
                model,
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

fn attach_solver_mesh_artifact_path(target: &mut Option<String>, resolved: Option<&str>) {
    if target.is_none() {
        *target = resolved.map(str::to_string);
    }
}
