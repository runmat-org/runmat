use runmat_analysis_core::{validate_model, AnalysisModel, PhysicsFamily};

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
    options: ThermalSolveOptions,
) -> Result<FeaThermalRunResult, FeaRunError> {
    validate_controls(&options)?;
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

    let mesh = options.solver_mesh.clone().ok_or_else(|| {
        FeaRunError::InvalidModel("thermal solve requires a canonical solver mesh".into())
    })?;
    physical::run(model, backend, options, &mesh)
}

fn validate_controls(options: &ThermalSolveOptions) -> Result<(), FeaRunError> {
    if let ThermalProcedureControl::BackwardEuler {
        step_count,
        time_step_s,
    } = options.procedure
    {
        if step_count < 2 {
            return Err(FeaRunError::InvalidModel(
                "transient thermal solve requires at least two snapshots".into(),
            ));
        }
        if !time_step_s.is_finite() || time_step_s <= 0.0 {
            return Err(FeaRunError::InvalidModel(
                "thermal time step must be finite and positive".into(),
            ));
        }
    }
    if !options.residual_target.is_finite() || options.residual_target <= 0.0 {
        return Err(FeaRunError::InvalidModel(
            "thermal residual target must be finite and positive".into(),
        ));
    }
    Ok(())
}
