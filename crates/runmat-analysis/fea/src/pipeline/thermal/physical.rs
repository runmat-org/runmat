use runmat_analysis_core::{AnalysisModel, IndependentDomain, SignalSampleApproach};

use crate::{
    contracts::{
        fea_thermal_energy_storage_rate_field_id, fea_thermal_heat_flux_field_id,
        fea_thermal_temperature_field_id, fea_thermal_temperature_gradient_field_id,
        ComputeBackend, ExecutionBackend, FeaRunError, FeaRunResult, FeaThermalRunResult,
        ThermalEnergyBalance, ThermalProcedureControl, ThermalSolveOptions,
    },
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    physics::thermal::{assembly::assemble as assemble_thermal_operator, initial_state},
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::{
        backend::{
            build_backend, kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend,
        },
        linear::{solve_operator_system_scoped, LinearSolveResult, ScopedLinearSolveSettings},
        preconditioner::SpdPreconditionerKind,
    },
};

mod backend_evidence;
mod diagnostics;
mod nonlinear;
mod snapshots;
mod transient;
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
        options.formulation_requirement,
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
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    drop(initial_excitations);
    operators.verify_matrix(&initial_assembly.conductivity)?;
    operators.verify_matrix(&initial_assembly.capacity)?;
    let initial = initial_state::realize(
        &active_model,
        &initial_assembly.projection,
        thermal_step.step_id.as_str(),
        domain.reference_temperature.kelvin,
        &initial_assembly.prescribed_temperature_k,
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
            initial_temperature_k: &initial.temperature_k,
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
            initial_temperature_k: &initial.temperature_k,
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
    let temperature_snapshots = scalar_snapshots(
        outcome.recovered.temperatures,
        fea_thermal_temperature_field_id,
    );
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
        temperature_gradient_snapshots,
        heat_flux_snapshots,
        energy_storage_rate_snapshots,
        residual_norms: outcome.residual_norms,
        residual_target: options.residual_target,
        energy_balances: outcome.energy_balances,
        nonlinear_evidence: outcome.nonlinear_evidence,
        reference_temperature_k: domain.reference_temperature.kelvin,
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
struct ProcedureOutcome {
    time_points_s: Vec<f64>,
    recovered: RecoveredSnapshots,
    residual_norms: Vec<f64>,
    energy_balances: Vec<ThermalEnergyBalance>,
    nonlinear_evidence: Vec<crate::contracts::ThermalNonlinearSampleEvidence>,
    source_evidence: Vec<crate::excitation::SpatialProjectionEvidence>,
    last_solve: LinearSolveResult,
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
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let operator = assembly
        .steady_increment_operator(model, sample, &reference_temperature_k)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    // A homogeneous anchored increment system has the exact zero solution. Starting it at
    // zero avoids introducing a spurious heat flow by iterating an unrelated warm guess.
    // This chooses the solver's initial state, never rewrites a computed solution.
    let initial_increment = (!operator.load().iter().all(|&value| value == 0.0)).then(|| {
        initial_temperature_k
            .iter()
            .zip(&reference_temperature_k)
            .map(|(value, reference)| value - reference)
            .collect::<Vec<_>>()
    });
    let rhs_norm = rhs_norm(&operator.linear_rhs());
    let mut solve = solve_checked(
        &operator,
        requested_solver_backend,
        algebra,
        initial_increment.as_deref(),
        &facts.operators,
        residual_target,
    )?;
    for ((temperature, prescribed), reference) in solve
        .solution
        .iter_mut()
        .zip(&assembly.prescribed_temperature_k)
        .zip(&reference_temperature_k)
    {
        // Prescribed coordinates are known exactly; avoid losing them in a shift/unshift
        // round trip when their magnitude differs greatly from the reference.
        *temperature = prescribed.unwrap_or(*temperature + reference);
    }
    let mut recovered = RecoveredSnapshots::with_capacity(1);
    push_recovered_snapshot(assembly, &solve.solution, None, 1.0, 0.0, &mut recovered)?;
    let energy_balance = assembly
        .energy_balance(model, sample, &solve.solution, &solve.solution, 1.0, 0.0)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let mut accelerator_evidence = Vec::new();
    backend_evidence::retain(
        &mut accelerator_evidence,
        &mut solve.accelerator_evidence,
        &facts.metadata.backend_evidence,
    )?;
    Ok(ProcedureOutcome {
        time_points_s: vec![time_origin_s],
        recovered,
        residual_norms: vec![solve.residual_norm / rhs_norm],
        energy_balances: vec![energy_balance],
        nonlinear_evidence: Vec::new(),
        source_evidence: assembly.source_evidence.clone(),
        host_sync_count: solve.host_sync_count,
        device_apply_count: solve.device_apply_k_count,
        device_apply_attempt_count: solve.device_apply_k_attempt_count,
        accelerator_evidence,
        last_solve: solve,
    })
}

fn solve_checked(
    operator: &crate::algebra::AssembledRealSystem,
    requested_solver_backend: LinearAlgebraBackendKind,
    algebra: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    facts: &crate::planning::resource::ThermalOperatorResourceFacts,
    residual_target: f64,
) -> Result<LinearSolveResult, FeaRunError> {
    for matrix in [
        operator.operators().stiffness(),
        operator.operators().mass(),
        operator.operators().damping(),
    ] {
        facts.verify_matrix(matrix)?;
    }
    // The authored acceptance target may tighten, but must not relax, the shared numerical
    // stopping policy. Linear residual normalization alone does not bound global energy error.
    let mut settings = ScopedLinearSolveSettings::with_default_tolerance(
        crate::planning::resource::LocalExecutionPlan::serial(),
        Some(facts.solver_admission()),
    );
    settings.relative_tolerance = settings.relative_tolerance.min(residual_target);
    let solve = solve_operator_system_scoped(
        operator,
        SpdPreconditionerKind::Jacobi,
        requested_solver_backend,
        algebra,
        initial_guess,
        settings,
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    if solve.converged {
        Ok(solve)
    } else {
        Err(FeaRunError::Assembly(format!(
            "physical thermal solve did not converge: residual_norm={}",
            solve.residual_norm
        )))
    }
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
                assembly.prescribed_temperature_k[node]
                    .unwrap_or(temperature + reference_temperature_k)
            })
            .collect(),
    );
    recovered.gradients.push(
        assembly
            .recover_element_temperature_gradients(temperature_k)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?
            .into_iter()
            .flatten()
            .collect(),
    );
    recovered.heat_flux.push(
        assembly
            .recover_element_heat_flux(temperature_k)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?
            .into_iter()
            .flatten()
            .collect(),
    );
    recovered.storage_rates.push(match previous_temperature_k {
        Some(previous) => assembly
            .recover_element_storage_rate(previous, temperature_k, time_step_s)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?,
        None => vec![0.0; assembly.projection.elements.len()],
    });
    Ok(())
}
