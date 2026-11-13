use runmat_analysis_core::{
    validate_model, AnalysisModel, CoupledPhysicsFamily, PhysicsFamily, ProcedureStep,
    RestartPolicy, SolutionSpace,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    contracts::accelerator_admission::{admit_accelerator, AcceleratorFamily},
    contracts::{
        ComputeBackend, FeaRunError, FeaThermalRunResult, ThermalProcedureControl,
        ThermalSolveOptions,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
};

mod inline_history;
mod physical;
mod prepared;
pub(crate) use prepared::ThermalObservationExecution;
pub(crate) use prepared::ThermalPreparationAdmission;
pub use prepared::{PreparedThermalPrimarySolve, PreparedThermalSolve};

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

pub(crate) fn prepare_selected<'a>(
    model: &'a AnalysisModel,
    backend: ComputeBackend,
    options: ThermalSolveOptions,
    mesh: &'a SolverMeshArtifact,
    step: &'a ProcedureStep,
) -> Result<PreparedThermalSolve<'a>, FeaRunError> {
    validate_run(model, &options)?;
    prepare_validated_model(
        model,
        backend,
        options,
        mesh,
        step,
        ThermalPreparationAdmission {
            observations: ThermalObservationExecution::Inline,
            maximum_resident_bytes: step.accuracy.resources.resident_byte_budget(),
        },
    )
}

pub(crate) fn prepare_selected_primary<'a>(
    model: &'a AnalysisModel,
    backend: ComputeBackend,
    options: ThermalSolveOptions,
    mesh: &'a SolverMeshArtifact,
    step: &'a ProcedureStep,
    maximum_resident_bytes: u64,
) -> Result<PreparedThermalPrimarySolve<'a>, FeaRunError> {
    validate_run(model, &options)?;
    Ok(PreparedThermalPrimarySolve::new(prepare_validated_model(
        model,
        backend,
        options,
        mesh,
        step,
        ThermalPreparationAdmission {
            observations: ThermalObservationExecution::Deferred,
            maximum_resident_bytes,
        },
    )?))
}

fn validate_run(model: &AnalysisModel, options: &ThermalSolveOptions) -> Result<(), FeaRunError> {
    validate_controls(options.procedure, options.residual_target)?;
    super::admission::require_nonstructural(PhysicsFamily::Thermal, options.procedure.procedure())?;
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
    prepare_validated_model(
        model,
        backend,
        options,
        mesh,
        step,
        ThermalPreparationAdmission {
            observations: ThermalObservationExecution::Inline,
            maximum_resident_bytes: step.accuracy.resources.resident_byte_budget(),
        },
    )?
    .execute()
}

fn prepare_validated_model<'a>(
    model: &'a AnalysisModel,
    backend: ComputeBackend,
    options: ThermalSolveOptions,
    mesh: &'a SolverMeshArtifact,
    step: &'a ProcedureStep,
    preparation_admission: ThermalPreparationAdmission,
) -> Result<PreparedThermalSolve<'a>, FeaRunError> {
    let maximum_resident_bytes = preparation_admission
        .maximum_resident_bytes
        .min(step.accuracy.resources.resident_byte_budget());
    validate_step(step, options.procedure)?;
    let radiation = model.boundaries.iter().any(|boundary| {
        matches!(
            boundary.operator,
            runmat_analysis_core::BoundaryOperator::ThermalRadiation { .. }
        )
    });
    // Strict placement fails here, before any assembly or solve, whenever admission can
    // already tell the device will not run (radiation, or no provider on this worker).
    let accelerator_family = if radiation {
        AcceleratorFamily::ThermalRadiation
    } else {
        AcceleratorFamily::Thermal
    };
    let admission = admit_accelerator(accelerator_family, backend)?;
    crate::planning::resource::require_initial_field_ceiling(
        &options.initial_fields,
        crate::execution::THERMAL_WORKLOAD_LIMITS,
        maximum_resident_bytes,
    )?;
    crate::physics::thermal::initial_state::fields::validate_selected(
        model,
        step,
        mesh,
        &options.initial_fields,
    )?;
    let temporal = match (options.procedure, radiation) {
        (ThermalProcedureControl::Steady, false) => {
            runmat_analysis_core::TemporalImplementation::ThermalSteadyLinearSolve
        }
        (ThermalProcedureControl::Steady, true) => {
            runmat_analysis_core::TemporalImplementation::ThermalSteadyDampedNewton
        }
        (ThermalProcedureControl::BackwardEuler { .. }, false) => {
            runmat_analysis_core::TemporalImplementation::ThermalBackwardEuler
        }
        (ThermalProcedureControl::BackwardEuler { .. }, true) => {
            runmat_analysis_core::TemporalImplementation::ThermalBackwardEulerDampedNewton
        }
    };
    let identity = crate::admission::admit_nonstructural_with_temporal(
        PhysicsFamily::Thermal,
        options.procedure.procedure(),
        Some(mesh),
        runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
        runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
        Some(temporal),
    )?;
    let schedule = crate::execution::ThermalSolveSchedule::derive_admitted(
        model,
        step,
        options.procedure,
        maximum_resident_bytes,
    )?;
    let facts =
        crate::planning::resource::ThermalPreassemblyResourceFacts::derive_with_observations(
            &options.initial_fields,
            model,
            step,
            mesh,
            &schedule,
            &identity,
            ThermalPreparationAdmission {
                maximum_resident_bytes,
                ..preparation_admission
            },
        )?;
    facts.require_fits(maximum_resident_bytes)?;
    Ok(PreparedThermalSolve {
        model,
        backend: admission.effective,
        accelerator: (accelerator_family, admission),
        options,
        mesh,
        step,
        schedule,
        facts,
        maximum_resident_bytes,
    })
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
    if step.restart != RestartPolicy::Fresh
        || !crate::study::thermal_step_initial_selection_supported(step)
    {
        return Err(FeaRunError::InvalidModel("thermal execution requires a fresh step with only authored physical thermal initial selection, not upstream state or restart checkpoints".into()));
    }
    Ok(())
}
