//! Shared admission and physical input realization for full and blocked J2 execution.
use super::*;

pub(super) struct PreparedNonlinear<'a> {
    pub execution_step: &'a runmat_analysis_core::ProcedureStep,
    pub summary: crate::assembly::AssemblySummary,
    pub identity: crate::admission::AnalysisRunIdentity,
    pub resource_plan: crate::planning::resource::FeaResourcePlan,
    pub increment_schedule: LoadIncrementSchedule,
    pub load_factor_start: f64,
    pub load_factor_end: f64,
}

pub(super) fn prepare<'a>(
    model: &'a AnalysisModel,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
) -> Result<PreparedNonlinear<'a>, FeaRunError> {
    prepare_with_retained(model, backend, options, 0)
}

pub(super) fn prepare_with_retained<'a>(
    model: &'a AnalysisModel,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
    additional_retained_bytes: u64,
) -> Result<PreparedNonlinear<'a>, FeaRunError> {
    prepare_with_output_limit(
        model,
        backend,
        options,
        options.max_accepted_increments,
        additional_retained_bytes,
    )
}

pub(super) fn prepare_with_output_limit<'a>(
    model: &'a AnalysisModel,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
    output_limit: usize,
    additional_retained_bytes: u64,
) -> Result<PreparedNonlinear<'a>, FeaRunError> {
    prepare_for(
        model,
        backend,
        options,
        output_limit,
        additional_retained_bytes,
        false,
    )
}

pub(super) fn prepare_recovery<'a>(
    model: &'a AnalysisModel,
    options: &NonlinearSolveOptions,
    output_limit: usize,
    additional_retained_bytes: u64,
) -> Result<PreparedNonlinear<'a>, FeaRunError> {
    prepare_for(
        model,
        ComputeBackend::Cpu,
        options,
        output_limit,
        additional_retained_bytes,
        true,
    )
}

fn prepare_for<'a>(
    model: &'a AnalysisModel,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
    output_limit: usize,
    additional_retained_bytes: u64,
    recovery: bool,
) -> Result<PreparedNonlinear<'a>, FeaRunError> {
    let execution_step =
        super::super::admission::require_structural(model, AnalysisProcedure::Nonlinear)?;
    validate_nonlinear_event_requests(execution_step)?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating nonlinear FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    if options.solver_mesh.is_none() {
        return Err(FeaRunError::InvalidModel(
            "nonlinear structural execution requires an authoritative canonical solid mesh".into(),
        ));
    }
    if backend != ComputeBackend::Cpu {
        return Err(FeaRunError::InvalidModel(
            "canonical nonlinear solids run on the CPU backend".into(),
        ));
    }
    if options.contact_enforcement.is_some() && model.interfaces.is_empty() {
        return Err(FeaRunError::InvalidModel(
            "contact enforcement controls require an authored contact interface".into(),
        ));
    }
    let required_history_samples = required_load_factor_history_samples(execution_step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let IndependentDomain::LoadFactor {
        start: load_factor_start,
        end: load_factor_end,
    } = execution_step.independent_domain
    else {
        return Err(FeaRunError::InvalidModel(
            "nonlinear structural execution requires a load-factor domain".into(),
        ));
    };
    let increment_schedule = LoadIncrementSchedule::canonical(
        &[],
        &required_history_samples,
        load_factor_start,
        load_factor_end,
    )
    .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    // Admit the full point-history/output envelope before excitation realization,
    // initial operator assembly or physical quadrature-history allocation.
    let resource_plan = if recovery {
        preflight::prepare_recovery(model, execution_step, backend, options, output_limit)?
    } else if output_limit == options.max_accepted_increments {
        preflight::prepare(model, execution_step, backend, options)?
    } else {
        preflight::prepare_with_output_limit(model, execution_step, backend, options, output_limit)?
    };
    let required = resource_plan
        .request
        .memory_bytes
        .checked_add(additional_retained_bytes)
        .ok_or(crate::planning::resource::ResourcePlanningError::Overflow(
            "original nonlinear preparation plus retained context",
        ))?;
    if required > execution_step.accuracy.resources.resident_byte_budget() {
        return Err(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "original nonlinear preparation plus retained context",
                required,
                maximum: execution_step.accuracy.resources.resident_byte_budget(),
            }
            .into(),
        );
    }
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "nonlinear model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    let excitations =
        realize_step_excitations(model, execution_step, ExcitationSamplePoint::Static)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling nonlinear system",
        Some(1),
        Some(5),
    );
    // Fixed-frame loads form the dead-load vector. Follower loads depend on the deformed
    // boundary; the nonlinear solid integrates them at every iterate and owns their source.
    let follower_present = excitations
        .iter()
        .any(|load| load.policy.kinematics == runmat_analysis_core::ExcitationKinematics::Follower);
    let dead_loads = excitations.filtered(|load| {
        load.policy.kinematics == runmat_analysis_core::ExcitationKinematics::FixedFrame
    });
    let mut summary = assemble_structural_system_borrowed(
        model,
        &dead_loads,
        options.solver_mesh.as_ref(),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .map_err(|err| match err {
        LinearAssemblyError::CouplingRealization(_) => FeaRunError::InvalidModel(err.to_string()),
        _ => FeaRunError::Assembly(err.to_string()),
    })?;
    if follower_present
        && !summary.structural_sources.contains(
            &runmat_analysis_core::SourceImplementation::CanonicalFollowerBoundaryIntegration,
        )
    {
        summary
            .structural_sources
            .push(runmat_analysis_core::SourceImplementation::CanonicalFollowerBoundaryIntegration);
    }
    let identity = super::super::admission::admit_structural(&summary, execution_step)?;
    super::super::validate_rotational_dof_targets(model, &summary)?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "nonlinear system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;

    Ok(PreparedNonlinear {
        execution_step,
        summary,
        identity,
        resource_plan,
        increment_schedule,
        load_factor_start,
        load_factor_end,
    })
}
