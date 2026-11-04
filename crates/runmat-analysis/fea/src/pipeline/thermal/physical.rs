use runmat_analysis_core::{AnalysisModel, IndependentDomain, SignalSampleApproach};

use crate::{
    contracts::{
        fea_thermal_energy_storage_rate_field_id, fea_thermal_heat_flux_field_id,
        fea_thermal_temperature_field_id, fea_thermal_temperature_gradient_field_id,
        ComputeBackend, ExecutionBackend, FeaRunError, FeaRunResult, FeaThermalRunResult,
        ThermalEnergyBalance, ThermalProcedureControl, ThermalSolveOptions,
    },
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    physics::{
        scalar_constraints::ScalarNodeRelation,
        thermal::{assembly::assemble as assemble_thermal_operator, initial_state},
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::{
        backend::{
            build_backend, kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend,
        },
        linear::{solve_operator_system_scoped, LinearSolveResult, ScopedLinearSolveSettings},
        preconditioner::SpdPreconditionerKind,
    },
};

type ThermalDeparture =
    runmat_types::physical::Quantity<runmat_types::physical::kind::TemperatureDifference, Vec<f64>>;
type ThermalSolveResult = LinearSolveResult<ThermalDeparture>;
type ThermalSystem =
    crate::algebra::PhysicalScalarSystem<runmat_types::physical::kind::TemperatureDifference>;
mod backend_evidence;
mod residual;
use residual::linear_residual;
mod diagnostics;
mod nonlinear;
mod snapshots;
mod transient;
#[cfg(test)]
mod relation_tests;
#[cfg(test)]
mod interface_tests;
use diagnostics::{diagnostics, ThermalDiagnosticsInput};
use snapshots::{scalar_snapshots, vector_snapshots};
use transient::{solve_transient, TransientSolveInput};

pub(super) fn run(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ThermalSolveOptions,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    thermal_step: &runmat_analysis_core::ProcedureStep,
    schedule: &crate::execution::ThermalSolveSchedule,
    facts: &crate::planning::resource::ThermalPreassemblyResourceFacts,
) -> Result<FeaThermalRunResult, FeaRunError> {
    let source_workload = crate::execution::ThermalWorkload::identity_for(
        model,
        &thermal_step.step_id,
        backend,
        options.procedure,
        options.residual_target,
        &options.initial_fields,
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let operators = &facts.operators;
    let numerical_intent = crate::execution::ThermalWorkload::numerical_intent_identity_for(
        model,
        &thermal_step.step_id,
        options.procedure,
        options.residual_target,
        &options.initial_fields,
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling physical thermal system",
        Some(1),
        Some(5),
    );
    let domain = model.thermal.as_ref().ok_or_else(|| {
        FeaRunError::InvalidModel("thermal solve requires model.thermal".to_string())
    })?;
    let identity = facts.admitted_identity.clone();

    let time_origin_s = schedule.points_s()[0];
    let active_model = super::super::activation::model_with_active_constraints(model, thermal_step);
    let initial_sample = sample_point(thermal_step, time_origin_s, SignalSampleApproach::At)?;
    let initial_excitations =
        realize_excitations_for_step(&active_model, thermal_step, initial_sample)?;
    let initial_assembly = assemble_thermal_operator(
        &active_model,
        mesh,
        &facts.properties,
        operators.support(),
        &initial_excitations,
        initial_sample,
    )
    .map_err(thermal_assembly_failure)?;
    drop(initial_excitations);
    operators.verify_matrix(&initial_assembly.conductivity)?;
    operators.verify_matrix(&initial_assembly.capacity)?;
    let initial = initial_state::realize(
        &active_model,
        &initial_assembly.projection,
        thermal_step.step_id.as_str(),
        domain.reference_temperature.canonical_value(),
        initial_assembly.prescribed_temperature_k.canonical_values(),
        &options.initial_fields,
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    facts.metadata.verify_initial_state(&initial.evidence)?;

    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "physical thermal system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_thermal")?;

    let requested_solver_backend = match backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => LinearAlgebraBackendKind::Accelerator,
    };
    let algebra = build_backend(requested_solver_backend);
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        match options.procedure {
            ThermalProcedureControl::Steady => "solving physical steady thermal system",
            ThermalProcedureControl::BackwardEuler { .. } => "solving physical thermal time steps",
        },
        Some(2),
        Some(5),
    );
    let outcome = match options.procedure {
        ThermalProcedureControl::Steady => solve_steady(SteadySolveInput {
            assembly: &initial_assembly,
            model: &active_model,
            sample: initial_sample,
            initial_temperature_k: initial.temperature_k.canonical_values(),
            time_origin_s,
            requested_solver_backend,
            algebra: algebra.as_ref(),
            facts,
            residual_target: options.residual_target,
        })?,
        ThermalProcedureControl::BackwardEuler { .. } => solve_transient(TransientSolveInput {
            properties: &facts.properties,
            model: &active_model,
            mesh,
            step: thermal_step,
            initial_assembly: &initial_assembly,
            initial_temperature_k: initial.temperature_k.canonical_values(),
            schedule,
            facts,
            requested_solver_backend,
            algebra: algebra.as_ref(),
            residual_target: options.residual_target,
        })?,
    };
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "physical thermal solve complete",
        Some(3),
        Some(5),
    );

    let diagnostics = diagnostics(ThermalDiagnosticsInput {
        metadata: &facts.metadata,
        model,
        initial: &initial.evidence,
        assembly: &initial_assembly,
        last_solve: &outcome.last_solve,
        temperature_snapshots: &outcome.recovered.temperatures,
        residual_norms: &outcome.residual_norms,
        energy_balances: &outcome.energy_balances,
        residual_target: options.residual_target,
    })?;
    let device_apply_ratio = if outcome.device_apply_attempt_count == 0 {
        0.0
    } else {
        outcome.device_apply_count as f64 / outcome.device_apply_attempt_count as f64
    };
    let time_points_s = outcome.time_points_s;
    // Projection nodes are the mesh nodes, then the second-side nodes of conductance interfaces.
    let dofs = &initial_assembly.dofs;
    let mut interface_temperatures = Vec::new();
    let mut nodal_temperatures = Vec::with_capacity(outcome.recovered.temperatures.len());
    for values in outcome.recovered.temperatures {
        let (nodal, second) = dofs
            .split(&values)
            .map_err(|error| FeaRunError::Solve(error))?;
        nodal_temperatures.push(nodal);
        if !dofs.is_nodal() {
            interface_temperatures.push(second.into_iter().map(|(_, value)| value).collect());
        }
    }
    let temperature_snapshots =
        scalar_snapshots(nodal_temperatures, fea_thermal_temperature_field_id);
    let interface_temperature_snapshots = scalar_snapshots(
        interface_temperatures,
        crate::contracts::fea_thermal_interface_temperature_field_id,
    );
    let interface_node_ids = dofs
        .second()
        .iter()
        .map(|&(node, _)| initial_assembly.projection.nodes[node].node_id)
        .collect();
    let temperature_gradient_snapshots = vector_snapshots(
        outcome.recovered.gradients,
        fea_thermal_temperature_gradient_field_id,
    );
    let heat_flux_snapshots =
        vector_snapshots(outcome.recovered.heat_flux, fea_thermal_heat_flux_field_id);
    let energy_storage_rate_snapshots = scalar_snapshots(
        outcome.recovered.storage_rates,
        fea_thermal_energy_storage_rate_field_id,
    );
    facts.sources.verify(&outcome.source_evidence)?;
    let run = FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: ExecutionBackend::from_solver_backend(outcome.last_solve.solver_backend),
        solver_backend: outcome.last_solve.solver_backend,
        solver_device_apply_k_ratio: device_apply_ratio,
        solver_method: outcome.last_solve.solver_method,
        preconditioner: outcome.last_solve.preconditioner,
        solver_host_sync_count: outcome.host_sync_count,
        accelerator_evidence: outcome.accelerator_evidence,
        diagnostics,
        spatial_projection_evidence: outcome.source_evidence,
        structural_static_convergence: None,
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: Vec::new(),
    };

    run.validate_solver_formulation()?;
    let sequence = crate::execution::SequenceIdentity::from_solver_contract(&model.sequence)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let producer = crate::execution::StepIdentity::from_solver_contract(sequence, thermal_step)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let result = FeaThermalRunResult {
        source_workload,
        numerical_intent,
        run,
        producer,
        topology_digest: runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
        ordered_node_ids: mesh
            .topology
            .nodes
            .iter()
            .map(|node| node.node_id)
            .collect(),
        ordered_element_ids: mesh
            .topology
            .volume_elements
            .iter()
            .map(|element| element.element_id)
            .collect(),
        time_points_s,
        temperature_snapshots,
        interface_node_ids,
        interface_temperature_snapshots,
        temperature_gradient_snapshots,
        heat_flux_snapshots,
        energy_storage_rate_snapshots,
        residual_norms: outcome.residual_norms,
        residual_target: options.residual_target,
        energy_balances: outcome.energy_balances,
        nonlinear_evidence: outcome.nonlinear_evidence,
        reference_temperature_k: runmat_types::physical::Temperature::new(
            domain.reference_temperature.canonical_value(),
        ),
        initial_state: initial.evidence,
    };
    result
        .validate_complete()
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA physical thermal run complete",
        Some(5),
        Some(5),
    );
    Ok(result)
}

fn thermal_assembly_failure(
    error: crate::physics::thermal::assembly::ThermalAssemblyError,
) -> FeaRunError {
    match error {
        crate::physics::thermal::assembly::ThermalAssemblyError::Accounting(cause) => {
            FeaRunError::Usage(cause)
        }
        other => FeaRunError::Assembly(other.to_string()),
    }
}
struct ProcedureOutcome {
    time_points_s: Vec<f64>,
    recovered: RecoveredSnapshots,
    residual_norms: Vec<f64>,
    energy_balances: Vec<ThermalEnergyBalance>,
    nonlinear_evidence: Vec<crate::contracts::ThermalNonlinearSampleEvidence>,
    source_evidence: Vec<crate::excitation::SpatialProjectionEvidence>,
    last_solve: LinearSolveResult<()>,
    host_sync_count: u32,
    device_apply_count: u32,
    device_apply_attempt_count: u32,
    accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
}

struct SteadySolveInput<'a> {
    assembly: &'a crate::physics::thermal::assembly::ThermalOperatorAssembly,
    model: &'a AnalysisModel,
    sample: ExcitationSamplePoint<'a>,
    initial_temperature_k: &'a [f64],
    time_origin_s: f64,
    requested_solver_backend: LinearAlgebraBackendKind,
    algebra: &'a dyn LinearAlgebraBackend,
    facts: &'a crate::planning::resource::ThermalPreassemblyResourceFacts,
    residual_target: f64,
}

fn solve_steady(input: SteadySolveInput<'_>) -> Result<ProcedureOutcome, FeaRunError> {
    if nonlinear::has_radiation(input.model) {
        return nonlinear::solve_steady(input);
    }
    let SteadySolveInput {
        assembly,
        model,
        sample,
        initial_temperature_k,
        time_origin_s,
        requested_solver_backend,
        algebra,
        facts,
        residual_target,
    } = input;
    let reference_temperature_k = assembly
        .steady_reference_temperatures(model, sample, initial_temperature_k[0])
        .map_err(thermal_assembly_failure)?;
    let operator = assembly
        .steady_increment_operator(model, sample, &reference_temperature_k)
        .map_err(thermal_assembly_failure)?;
    // A homogeneous anchored increment system has the exact zero solution. Starting it at
    // zero avoids introducing a spurious heat flow by iterating an unrelated warm guess.
    // This chooses the solver's initial state, never rewrites a computed solution.
    let has_forcing = operator
        .rhs_field()
        .map_err(|error| FeaRunError::Solve(error.into()))?
        .with_canonical_storage(|values| values.iter().any(|&value| value != 0.0));
    let initial_increment = has_forcing.then(|| {
        initial_temperature_k
            .iter()
            .zip(&reference_temperature_k)
            .map(|(value, reference)| value - reference)
            .collect::<Vec<_>>()
    });
    let relations = crate::physics::scalar_constraints::about_reference(
        &assembly.relations,
        &reference_temperature_k,
    );
    let mut solve = solve_checked(
        &operator,
        requested_solver_backend,
        algebra,
        initial_increment
            .as_deref()
            .map(runmat_types::physical::Quantity::new),
        &facts.operators,
        residual_target,
        None,
        &relations,
    )?;
    drop(initial_increment);
    // Prescribed coordinates are known exactly; avoid losing them in a shift/unshift
    // round trip when their magnitude differs greatly from the reference.
    let absolute = |increment: &[f64]| {
        increment
            .iter()
            .zip(assembly.prescribed_temperature_k.canonical_values())
            .zip(&reference_temperature_k)
            .map(|((value, prescribed), reference)| prescribed.unwrap_or(value + reference))
            .collect::<Vec<_>>()
    };
    let recover_balance = |temperature: &[f64]| {
        assembly
            .energy_balance(model, sample, temperature, temperature, 1.0, 0.0)
            .map_err(thermal_assembly_failure)
    };
    let mut temperature = absolute(solve.solution.canonical_values());
    let mut energy_balance = recover_balance(&temperature)?;
    let mut accelerator_evidence = Vec::new();
    if energy_balance.relative_residual > residual_target {
        // The same single bounded refinement as each transient step: the linear stopping
        // criterion alone does not bound the global signed power residual.
        let target =
            energy_refinement_target(&operator, algebra, &solve, &energy_balance, residual_target);
        let mut refined = solve_refinement(
            &operator,
            requested_solver_backend,
            algebra,
            &solve.solution,
            &facts.operators,
            target,
            None,
            &relations,
        )?;
        backend_evidence::retain(
            &mut accelerator_evidence,
            &mut solve.accelerator_evidence,
            &facts.metadata.backend_evidence,
        )?;
        refined.host_sync_count = refined
            .host_sync_count
            .saturating_add(solve.host_sync_count);
        refined.device_apply_k_count = refined
            .device_apply_k_count
            .saturating_add(solve.device_apply_k_count);
        refined.device_apply_k_attempt_count = refined
            .device_apply_k_attempt_count
            .saturating_add(solve.device_apply_k_attempt_count);
        temperature = absolute(refined.solution.canonical_values());
        energy_balance = recover_balance(&temperature)?;
        solve = refined;
    }
    // Numerical preparation is complete. Release the constraint mask and zero damping
    // carrier before allocating the complete recovered fields.
    drop(operator);
    drop(reference_temperature_k);
    let mut solve = solve.try_map_solution(|_departure| {
        runmat_types::physical::Quantity::<runmat_types::physical::kind::Temperature, _>::try_new(
            temperature,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    })?;
    let mut recovered = RecoveredSnapshots::with_capacity(1);
    push_recovered_snapshot(
        assembly,
        solve.solution.canonical_values(),
        None,
        1.0,
        0.0,
        &mut recovered,
    )?;
    backend_evidence::retain(
        &mut accelerator_evidence,
        &mut solve.accelerator_evidence,
        &facts.metadata.backend_evidence,
    )?;
    Ok(ProcedureOutcome {
        time_points_s: vec![time_origin_s],
        recovered,
        residual_norms: vec![linear_residual(&solve)?],
        energy_balances: vec![energy_balance],
        nonlinear_evidence: Vec::new(),
        source_evidence: assembly.source_evidence.clone(),
        host_sync_count: solve.host_sync_count,
        device_apply_count: solve.device_apply_k_count,
        device_apply_attempt_count: solve.device_apply_k_attempt_count,
        accelerator_evidence,
        last_solve: solve.map_solution(|_state| ()),
    })
}

/// Solver tolerance for one bounded refinement after the recovered global balance missed the
/// retained target. |sum r_i| <= sqrt(N) ||r||_2, so the measured physical power scale is
/// translated to the shared solver's RHS-relative criterion, reserving half the power budget for
/// assembly/recovery roundoff. Shared preparation owns prescribed-temperature lifting; this
/// matches PCG's normalization (initial residual when the lifted RHS is exactly zero). Retained
/// physical acceptance stays authoritative and is never rewritten.
fn energy_refinement_target(
    operator: &ThermalSystem,
    algebra: &dyn LinearAlgebraBackend,
    solve: &ThermalSolveResult,
    balance: &ThermalEnergyBalance,
    residual_target: f64,
) -> f64 {
    let lifted_rhs = operator.linear_rhs();
    let lifted_norm = algebra.dot(&lifted_rhs, &lifted_rhs).sqrt();
    drop(lifted_rhs);
    let reference_norm = if lifted_norm > 0.0 {
        lifted_norm
    } else {
        solve.residual_norm
    };
    (0.5 * residual_target * balance.scale.maximum().canonical_value()
        / ((operator.dof_count() as f64).sqrt() * reference_norm))
        .min(residual_target)
}

/// The bounded energy refinement starting from an accepted solve. Its linear tolerance can lie
/// below what floating point reaches on an ill-conditioned operator; conjugate gradients only
/// lower the energy-norm error from the accepted start, so the last iterate is kept and the
/// recovered energy balance, not the refinement's own stopping test, decides acceptance.
pub(super) fn solve_refinement(
    operator: &ThermalSystem,
    requested_solver_backend: LinearAlgebraBackendKind,
    algebra: &dyn LinearAlgebraBackend,
    initial_guess: &ThermalDeparture,
    facts: &crate::planning::resource::ThermalOperatorResourceFacts,
    residual_target: f64,
    preparation: Option<&mut crate::algebra::LinearPreparationReuse>,
    relations: &[ScalarNodeRelation],
) -> Result<ThermalSolveResult, FeaRunError> {
    match solve_unchecked(
        operator,
        requested_solver_backend,
        algebra,
        Some(runmat_types::physical::Quantity::new(
            initial_guess.canonical_values().as_slice(),
        )),
        facts,
        residual_target,
        preparation,
        relations,
    )? {
        solve
            if solve
                .solution
                .canonical_values()
                .iter()
                .all(|value| value.is_finite()) =>
        {
            Ok(solve)
        }
        _ => Err(FeaRunError::Assembly(
            "thermal energy refinement produced a non-finite temperature".into(),
        )),
    }
}

pub(super) fn solve_checked(
    operator: &ThermalSystem,
    requested_solver_backend: LinearAlgebraBackendKind,
    algebra: &dyn LinearAlgebraBackend,
    initial_guess: Option<
        runmat_types::physical::Quantity<
            runmat_types::physical::kind::TemperatureDifference,
            &[f64],
        >,
    >,
    facts: &crate::planning::resource::ThermalOperatorResourceFacts,
    residual_target: f64,
    preparation: Option<&mut crate::algebra::LinearPreparationReuse>,
    relations: &[ScalarNodeRelation],
) -> Result<ThermalSolveResult, FeaRunError> {
    let solve = solve_unchecked(
        operator,
        requested_solver_backend,
        algebra,
        initial_guess,
        facts,
        residual_target,
        preparation,
        relations,
    )?;
    if solve.converged {
        Ok(solve)
    } else {
        Err(FeaRunError::Assembly(format!(
            "physical thermal solve did not converge: residual_norm={}",
            solve.residual_norm
        )))
    }
}

fn solve_unchecked(
    operator: &ThermalSystem,
    requested_solver_backend: LinearAlgebraBackendKind,
    algebra: &dyn LinearAlgebraBackend,
    initial_guess: Option<
        runmat_types::physical::Quantity<
            runmat_types::physical::kind::TemperatureDifference,
            &[f64],
        >,
    >,
    facts: &crate::planning::resource::ThermalOperatorResourceFacts,
    residual_target: f64,
    preparation: Option<&mut crate::algebra::LinearPreparationReuse>,
    relations: &[ScalarNodeRelation],
) -> Result<ThermalSolveResult, FeaRunError> {
    for matrix in [
        operator.operators().stiffness(),
        operator.operators().mass(),
        operator.operators().damping(),
    ] {
        facts.verify_matrix(matrix)?;
    }
    let power = runmat_types::physical::DynamicDimension::quantity_type(
        runmat_types::physical::QuantityKind::Power,
    )
    .with_semantic_kind(Some(runmat_types::physical::QuantityKind::Power))
    .expect("thermal equation");
    operator
        .require_equation(power)
        .map_err(|error| FeaRunError::Solve(error.into()))?;
    operator
        .physical_operator()
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    // The authored acceptance target may tighten, but must not relax, the shared numerical
    // stopping policy. Linear residual normalization alone does not bound global energy error.
    let mut settings = ScopedLinearSolveSettings::with_default_tolerance(
        crate::planning::resource::LocalExecutionPlan::serial(),
        Some(facts.solver_admission()),
    );
    settings.relative_tolerance = settings.relative_tolerance.min(residual_target);
    if !relations.is_empty() {
        let related = operator.with_canonical_system(|numerical| {
            solve_related(numerical, settings.relative_tolerance, relations)
        })?;
        return operator
            .admit_solution(related)
            .map_err(|error| FeaRunError::Solve(error.into()));
    }
    // Reused preparation keeps the proving Cholesky factor across solves with an unchanged
    // operator (fixed-step transients, refinement re-solves).
    settings.preparation = preparation;
    let solve = operator
        .with_canonical_system(|numerical| {
            solve_operator_system_scoped(
                numerical,
                SpdPreconditionerKind::Auto,
                requested_solver_backend,
                algebra,
                initial_guess.map(|quantity| quantity.canonical_value()),
                settings,
            )
        })
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    operator
        .admit_solution(solve)
        .map_err(|error| FeaRunError::Solve(error.into()))
}

/// Tie, periodic and jump relations eliminate secondary nodes (Galerkin reduction) and solve
/// the reduced system with the shared sparse direct solver.
fn solve_related(
    operator: &crate::algebra::AssembledRealSystem,
    relative_tolerance: f64,
    relations: &[ScalarNodeRelation],
) -> Result<LinearSolveResult, FeaRunError> {
    let load = operator.load();
    let known = operator
        .constrained_dofs()
        .iter()
        .zip(load)
        .map(|(&constrained, &value)| constrained.then_some(value))
        .collect::<Vec<_>>();
    let reduced = crate::physics::scalar_constraints::solve(
        operator.operators().stiffness(),
        load,
        &known,
        relations,
        relative_tolerance,
        crate::solve::preconditioner::SpdPreconditionerKind::Auto,
    )
    .map_err(|error| match error {
        crate::physics::scalar_constraints::ScalarSolveError::Usage(error) => {
            FeaRunError::Usage(error)
        }
        crate::physics::scalar_constraints::ScalarSolveError::Detail(detail) => {
            FeaRunError::Solve(detail)
        }
    })?;
    Ok(LinearSolveResult {
        spectral_estimate: None,
        iterations: reduced.iterations,
        residual_norm: reduced.relative_residual * rhs_norm(load),
        relative_residual: Some(reduced.relative_residual),
        converged: true,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: reduced.solution,
        solver_method: reduced.method,
        preconditioner: reduced.preconditioner,
        diagnostics: Vec::new(),
    })
}

fn rhs_norm(rhs: &[f64]) -> f64 {
    rhs.iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0)
}

fn realize_excitations_for_step(
    model: &AnalysisModel,
    step: &runmat_analysis_core::ProcedureStep,
    sample: ExcitationSamplePoint<'_>,
) -> Result<crate::excitation::RealizedExcitations, FeaRunError> {
    realize_step_excitations(model, step, sample)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))
}

fn sample_point<'a>(
    step: &'a runmat_analysis_core::ProcedureStep,
    time_s: f64,
    approach: SignalSampleApproach,
) -> Result<ExcitationSamplePoint<'a>, FeaRunError> {
    match &step.independent_domain {
        IndependentDomain::TimeSeconds { .. } => Ok(ExcitationSamplePoint::TimeSeconds {
            value: time_s,
            approach,
        }),
        IndependentDomain::Static => Ok(ExcitationSamplePoint::Static),
        _ => Err(FeaRunError::InvalidModel(
            "thermal solve requires a static or time-domain procedure step".into(),
        )),
    }
}

struct RecoveredSnapshots {
    temperatures: Vec<Vec<f64>>,
    gradients: Vec<Vec<f64>>,
    heat_flux: Vec<Vec<f64>>,
    storage_rates: Vec<Vec<f64>>,
}

impl RecoveredSnapshots {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            temperatures: Vec::with_capacity(capacity),
            gradients: Vec::with_capacity(capacity),
            heat_flux: Vec::with_capacity(capacity),
            storage_rates: Vec::with_capacity(capacity),
        }
    }
}

fn push_recovered_snapshot(
    assembly: &crate::physics::thermal::assembly::ThermalOperatorAssembly,
    temperature_k: &[f64],
    previous_temperature_k: Option<&[f64]>,
    time_step_s: f64,
    reference_temperature_k: f64,
    recovered: &mut RecoveredSnapshots,
) -> Result<(), FeaRunError> {
    recovered.temperatures.push(
        temperature_k
            .iter()
            .enumerate()
            .map(|(node, temperature)| {
                assembly.prescribed_temperature_k.canonical_values()[node]
                    .unwrap_or(temperature + reference_temperature_k)
            })
            .collect(),
    );
    recovered.gradients.push(
        assembly
            .recover_element_temperature_gradients(temperature_k)
            .map_err(thermal_assembly_failure)?
            .into_iter()
            .flatten()
            .collect(),
    );
    recovered.heat_flux.push(
        assembly
            .recover_element_heat_flux(temperature_k)
            .map_err(thermal_assembly_failure)?
            .into_iter()
            .flatten()
            .collect(),
    );
    recovered.storage_rates.push(match previous_temperature_k {
        Some(previous) => assembly
            .recover_element_storage_rate(previous, temperature_k, time_step_s)
            .map_err(thermal_assembly_failure)?,
        None => vec![0.0; assembly.projection.elements.len()],
    });
    Ok(())
}
