use runmat_analysis_core::{
    validate_model, AnalysisModel, CoupledPhysicsFamily, PhysicsFamily, ProcedureStep,
    RestartPolicy, SolutionSpace,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    contracts::{
        ComputeBackend, FeaRunError, FeaThermalRunResult, ThermalProcedureControl,
        ThermalSolveOptions,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
};

mod physical;

pub fn run_thermal_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    mut options: ThermalSolveOptions,
) -> Result<FeaThermalRunResult, FeaRunError> {
    validate_run(model, &options)?;
    let mut steps = model.sequence.steps.iter().filter(|step| {
        step.procedure == options.procedure.procedure()
            && matches!(
                step.family,
                PhysicsFamily::Thermal
                    | PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer)
            )
    });
    let step = steps.next().filter(|_| steps.next().is_none()).ok_or_else(|| {
        FeaRunError::InvalidModel("thermal solve requires exactly one matching thermal procedure step; use an explicit thermal workload for multiple matching steps".into())
    })?;
    let mesh = options.solver_mesh.take().ok_or_else(|| {
        FeaRunError::InvalidModel("thermal solve requires a canonical solver mesh".into())
    })?;
    execute_validated_model(model, backend, options, &mesh, step)
}

pub(crate) fn run_selected(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ThermalSolveOptions,
    mesh: &SolverMeshArtifact,
    step: &ProcedureStep,
) -> Result<FeaThermalRunResult, FeaRunError> {
    validate_run(model, &options)?;
    execute_validated_model(model, backend, options, mesh, step)
}

fn validate_run(model: &AnalysisModel, options: &ThermalSolveOptions) -> Result<(), FeaRunError> {
    validate_controls(options.procedure, options.residual_target)?;
    super::admission::require_nonstructural(
        PhysicsFamily::Thermal,
        options.procedure.procedure(),
        options.formulation_requirement,
    )?;
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating thermal FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_thermal")?;
    validate_model(model).map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    super::reject_moment_loads_for_nonstructural_pipeline(model, "thermal")?;
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "thermal model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_thermal")?;
    Ok(())
}

fn execute_validated_model(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ThermalSolveOptions,
    mesh: &SolverMeshArtifact,
    step: &ProcedureStep,
) -> Result<FeaThermalRunResult, FeaRunError> {
    validate_step(step, options.procedure)?;
    mesh.validate_canonical()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    physical::run(model, backend, options, mesh, step)
}

pub(crate) fn validate_controls(
    procedure: ThermalProcedureControl,
    residual_target: f64,
) -> Result<(), FeaRunError> {
    if let ThermalProcedureControl::BackwardEuler { snapshot_count } = procedure {
        if snapshot_count < 2 {
            return Err(FeaRunError::InvalidModel(
                "transient thermal solve requires at least two snapshots".into(),
            ));
        }
    }
    if !residual_target.is_finite() || residual_target <= 0.0 {
        return Err(FeaRunError::InvalidModel(
            "thermal residual target must be finite and positive".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_step(
    step: &ProcedureStep,
    procedure: ThermalProcedureControl,
) -> Result<(), FeaRunError> {
    if step.procedure != procedure.procedure() || step.solution_space != SolutionSpace::Direct {
        return Err(FeaRunError::InvalidModel(
            "thermal controls require a matching direct procedure step".into(),
        ));
    }
    if step.restart != RestartPolicy::Fresh || !step.state_inputs.is_empty() {
        return Err(FeaRunError::InvalidModel("thermal execution does not yet bind upstream state or restart checkpoints; a fresh step without state inputs is required".into()));
    }
    Ok(())
}
