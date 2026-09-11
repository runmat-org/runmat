use runmat_analysis_core::{
    AnalysisField, AnalysisModel, CoupledPhysicsFamily, IndependentDomain, PhysicsFamily,
    SignalSampleApproach,
};

use crate::{
    contracts::{
        fea_thermal_energy_storage_rate_field_id, fea_thermal_heat_flux_field_id,
        fea_thermal_temperature_field_id, fea_thermal_temperature_gradient_field_id,
        ComputeBackend, ExecutionBackend, FeaRunError, FeaRunResult, FeaThermalRunResult,
        ThermalProcedureControl, ThermalSolveOptions,
    },
    diagnostics::{
        builders::material_assignment_diagnostics, FeaDiagnostic, FeaDiagnosticSeverity,
    },
    excitation::{realize_step_excitations, time_discontinuities, ExcitationSamplePoint},
    physics::thermal::{
        assembly::{assemble as assemble_thermal_operator, ThermalEnergyBalance},
        constitutive_stats, initial_state,
    },
    pipeline::observation_history::{
        evaluate_field_histories, required_time_history_samples, HistoryFieldSeries,
        HistoryTopology,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::{
        backend::{
            build_backend, kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend,
        },
        linear::{solve_operator_system, LinearSolveResult},
        preconditioner::SpdPreconditionerKind,
    },
};

const THERMAL_TEMPERATURE_HISTORY_SOURCE: &str = "thermal.temperature";
const THERMAL_TEMPERATURE_GRADIENT_HISTORY_SOURCE: &str = "thermal.temperature_gradient";
const THERMAL_HEAT_FLUX_HISTORY_SOURCE: &str = "thermal.heat_flux";

pub(super) fn run(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ThermalSolveOptions,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> Result<FeaThermalRunResult, FeaRunError> {
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
    let procedure = options.procedure.procedure();
    let identity = super::super::admission::admit_nonstructural(
        PhysicsFamily::Thermal,
        procedure,
        Some(mesh),
        runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
        runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
        options.formulation_requirement,
    )?;

    let thermal_step = thermal_step(model, procedure)?;
    let required_history_samples = required_time_history_samples(thermal_step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let time_origin_s = match &thermal_step.independent_domain {
        IndependentDomain::TimeSeconds { start, .. } => *start,
        IndependentDomain::Static => 0.0,
        _ => {
            return Err(FeaRunError::InvalidModel(
                "thermal solve requires a static or time-domain procedure step".into(),
            ));
        }
    };
    let active_model = super::super::activation::model_with_active_constraints(model, thermal_step);
    let initial_sample = sample_point(thermal_step, time_origin_s, SignalSampleApproach::At)?;
    let initial_excitations =
        realize_excitations_for_step(&active_model, thermal_step, initial_sample)?;
    let initial_assembly =
        assemble_thermal_operator(&active_model, mesh, &initial_excitations, initial_sample)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let initial = initial_state::realize(
        &active_model,
        &initial_assembly.projection,
        thermal_step.step_id.as_str(),
        domain.reference_temperature.kelvin,
        &initial_assembly.prescribed_temperature_k,
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;

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
        ThermalProcedureControl::Steady => solve_steady(
            &initial_assembly,
            &initial.temperature_k,
            time_origin_s,
            requested_solver_backend,
            algebra.as_ref(),
        )?,
        ThermalProcedureControl::BackwardEuler { snapshot_count } => {
            let time_grid = crate::excitation::ProcedureTimeGrid::from_sample_count(
                thermal_step,
                snapshot_count,
            )
            .map_err(FeaRunError::InvalidModel)?;
            solve_transient(TransientSolveInput {
                model: &active_model,
                mesh,
                step: thermal_step,
                initial_assembly: &initial_assembly,
                initial_temperature_k: &initial.temperature_k,
                time_grid,
                required_history_samples: &required_history_samples,
                requested_solver_backend,
                algebra: algebra.as_ref(),
            })?
        }
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
        model,
        initial: &initial.evidence,
        assembly: &initial_assembly,
        last_solve: &outcome.last_solve,
        temperature_snapshots: &outcome.recovered.temperatures,
        residual_norms: &outcome.residual_norms,
        energy_balances: &outcome.energy_balances,
        residual_target: options.residual_target,
    });
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
    let mut run = FeaRunResult {
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
        observations: Vec::new(),
        observation_histories: Vec::new(),
        fields: Vec::new(),
    };
    if thermal_step.observations.iter().any(|request| {
        matches!(
            request,
            runmat_analysis_core::ObservationRequest::History(_)
        )
    }) {
        let coordinates = time_points_s
            .iter()
            .copied()
            .map(crate::results::field::RecoveredFieldSample::TimeSeconds)
            .collect::<Vec<_>>();
        run.observation_histories = evaluate_field_histories(
            thermal_step,
            HistoryTopology::Solid(&initial_assembly.projection),
            &coordinates,
            &[
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::Temperature,
                    source_field_id: THERMAL_TEMPERATURE_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Node,
                    entity_domain:
                        crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                    value_shape: runmat_analysis_core::ObservationValueShape::Scalar,
                    snapshots: &temperature_snapshots,
                },
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::TemperatureGradient,
                    source_field_id: THERMAL_TEMPERATURE_GRADIENT_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Element,
                    entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                    value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                    snapshots: &temperature_gradient_snapshots,
                },
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::HeatFlux,
                    source_field_id: THERMAL_HEAT_FLUX_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Element,
                    entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                    value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                    snapshots: &heat_flux_snapshots,
                },
            ],
        )
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    }

    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA physical thermal run complete",
        Some(5),
        Some(5),
    );
    Ok(FeaThermalRunResult {
        run,
        time_points_s,
        temperature_snapshots,
        temperature_gradient_snapshots,
        heat_flux_snapshots,
        energy_storage_rate_snapshots,
        residual_norms: outcome.residual_norms,
        reference_temperature_k: domain.reference_temperature.kelvin,
        initial_state: initial.evidence,
    })
}

struct ProcedureOutcome {
    time_points_s: Vec<f64>,
    recovered: RecoveredSnapshots,
    residual_norms: Vec<f64>,
    energy_balances: Vec<ThermalEnergyBalance>,
    source_evidence: Vec<crate::excitation::SpatialProjectionEvidence>,
    last_solve: LinearSolveResult,
    host_sync_count: u32,
    device_apply_count: u32,
    device_apply_attempt_count: u32,
    accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
}

fn solve_steady(
    assembly: &crate::physics::thermal::assembly::ThermalOperatorAssembly,
    initial_temperature_k: &[f64],
    time_origin_s: f64,
    requested_solver_backend: LinearAlgebraBackendKind,
    algebra: &dyn LinearAlgebraBackend,
) -> Result<ProcedureOutcome, FeaRunError> {
    let operator = assembly
        .steady_operator()
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let rhs_norm = rhs_norm(operator.load());
    let solve = solve_checked(
        &operator,
        requested_solver_backend,
        algebra,
        Some(initial_temperature_k),
    )?;
    let mut recovered = RecoveredSnapshots::with_capacity(1);
    push_recovered_snapshot(assembly, &solve.solution, None, 1.0, &mut recovered)?;
    let energy_balance = assembly.energy_balance(&solve.solution, &solve.solution, 1.0);
    Ok(ProcedureOutcome {
        time_points_s: vec![time_origin_s],
        recovered,
        residual_norms: vec![solve.residual_norm / rhs_norm],
        energy_balances: vec![energy_balance],
        source_evidence: assembly.source_evidence.clone(),
        host_sync_count: solve.host_sync_count,
        device_apply_count: solve.device_apply_k_count,
        device_apply_attempt_count: solve.device_apply_k_attempt_count,
        accelerator_evidence: solve.accelerator_evidence.clone().into_iter().collect(),
        last_solve: solve,
    })
}

struct TransientSolveInput<'a> {
    model: &'a AnalysisModel,
    mesh: &'a runmat_meshing_core::SolverMeshArtifact,
    step: &'a runmat_analysis_core::ProcedureStep,
    initial_assembly: &'a crate::physics::thermal::assembly::ThermalOperatorAssembly,
    initial_temperature_k: &'a [f64],
    time_grid: crate::excitation::ProcedureTimeGrid,
    required_history_samples: &'a [f64],
    requested_solver_backend: LinearAlgebraBackendKind,
    algebra: &'a dyn LinearAlgebraBackend,
}

fn solve_transient(input: TransientSolveInput<'_>) -> Result<ProcedureOutcome, FeaRunError> {
    let events = time_discontinuities(
        input.model,
        input.step,
        input.time_grid.start_s(),
        input.time_grid.end_s(),
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let mut solve_times = input
        .time_grid
        .points_s()
        .iter()
        .copied()
        .chain(events.iter().copied())
        .chain(input.required_history_samples.iter().copied())
        .collect::<Vec<_>>();
    solve_times.sort_by(f64::total_cmp);
    solve_times.dedup_by(|left, right| left.to_bits() == right.to_bits());
    let mut time_points_s = Vec::with_capacity(solve_times.len());
    let mut recovered = RecoveredSnapshots::with_capacity(solve_times.len());
    let mut residual_norms = Vec::with_capacity(solve_times.len() - 1);
    let mut energy_balances = Vec::with_capacity(solve_times.len() - 1);
    let mut source_evidence = input.initial_assembly.source_evidence.clone();
    let mut previous = input.initial_temperature_k.to_vec();
    push_recovered_snapshot(
        input.initial_assembly,
        &previous,
        None,
        input.time_grid.nominal_interval_s(),
        &mut recovered,
    )?;
    time_points_s.push(input.time_grid.start_s());
    let mut last_solve = None;
    let mut host_sync_count = 0_u32;
    let mut device_apply_count = 0_u32;
    let mut device_apply_attempt_count = 0_u32;
    let mut accelerator_evidence = Vec::new();
    for step_index in 1..solve_times.len() {
        check_cancelled("fea.run_thermal")?;
        let previous_time_s = solve_times[step_index - 1];
        let time_s = solve_times[step_index];
        let time_step_s = time_s - previous_time_s;
        emit_phase(
            "fea.run_thermal",
            FeaProgressPhase::Solve,
            FeaProgressStatus::Advanced,
            format!("solving physical thermal step {}", step_index + 1),
            Some(step_index as u64),
            Some(solve_times.len() as u64),
        );
        let approach = if events
            .binary_search_by(|event| event.total_cmp(&time_s))
            .is_ok()
        {
            SignalSampleApproach::LeftLimit
        } else {
            SignalSampleApproach::At
        };
        let sample = sample_point(input.step, time_s, approach)?;
        let excitations = realize_excitations_for_step(input.model, input.step, sample)?;
        let assembly = assemble_thermal_operator(input.model, input.mesh, &excitations, sample)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        source_evidence.extend(assembly.source_evidence.clone());
        let operator = assembly
            .transient_operator(time_step_s, &previous)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        let rhs_norm = rhs_norm(operator.load());
        let solve = solve_checked(
            &operator,
            input.requested_solver_backend,
            input.algebra,
            Some(&previous),
        )?;
        host_sync_count = host_sync_count.saturating_add(solve.host_sync_count);
        device_apply_count = device_apply_count.saturating_add(solve.device_apply_k_count);
        device_apply_attempt_count =
            device_apply_attempt_count.saturating_add(solve.device_apply_k_attempt_count);
        accelerator_evidence.extend(solve.accelerator_evidence.clone());
        residual_norms.push(solve.residual_norm / rhs_norm);
        energy_balances.push(assembly.energy_balance(&previous, &solve.solution, time_step_s));
        push_recovered_snapshot(
            &assembly,
            &solve.solution,
            Some(&previous),
            time_step_s,
            &mut recovered,
        )?;
        previous = solve.solution.clone();
        time_points_s.push(time_s);
        last_solve = Some(solve);
    }
    Ok(ProcedureOutcome {
        time_points_s,
        recovered,
        residual_norms,
        energy_balances,
        source_evidence,
        last_solve: last_solve.expect("validated transient controls perform at least one solve"),
        host_sync_count,
        device_apply_count,
        device_apply_attempt_count,
        accelerator_evidence,
    })
}

fn solve_checked(
    operator: &crate::algebra::AssembledRealSystem,
    requested_solver_backend: LinearAlgebraBackendKind,
    algebra: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> Result<LinearSolveResult, FeaRunError> {
    let solve = solve_operator_system(
        operator,
        SpdPreconditionerKind::Jacobi,
        requested_solver_backend,
        algebra,
        initial_guess,
    );
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

fn thermal_step(
    model: &AnalysisModel,
    procedure: runmat_analysis_core::AnalysisProcedure,
) -> Result<&runmat_analysis_core::ProcedureStep, FeaRunError> {
    let mut steps = model.sequence.steps.iter().filter(|step| {
        step.procedure == procedure
            && matches!(
                step.family,
                PhysicsFamily::Thermal
                    | PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer)
            )
    });
    steps
        .next()
        .filter(|_| steps.next().is_none())
        .ok_or_else(|| {
            FeaRunError::InvalidModel(
                "thermal solve requires exactly one matching thermal procedure step".into(),
            )
        })
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
    recovered: &mut RecoveredSnapshots,
) -> Result<(), FeaRunError> {
    recovered.temperatures.push(temperature_k.to_vec());
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

fn scalar_snapshots(values: Vec<Vec<f64>>, field_id: fn(usize) -> String) -> Vec<AnalysisField> {
    values
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            let len = values.len();
            AnalysisField::host_f64(field_id(index), vec![len], values)
        })
        .collect()
}

fn vector_snapshots(values: Vec<Vec<f64>>, field_id: fn(usize) -> String) -> Vec<AnalysisField> {
    values
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            let count = values.len() / 3;
            AnalysisField::host_f64(field_id(index), vec![count, 3], values)
        })
        .collect()
}

struct ThermalDiagnosticsInput<'a> {
    model: &'a AnalysisModel,
    initial: &'a crate::contracts::ThermalInitialStateEvidence,
    assembly: &'a crate::physics::thermal::assembly::ThermalOperatorAssembly,
    last_solve: &'a LinearSolveResult,
    temperature_snapshots: &'a [Vec<f64>],
    residual_norms: &'a [f64],
    energy_balances: &'a [ThermalEnergyBalance],
    residual_target: f64,
}

fn diagnostics(input: ThermalDiagnosticsInput<'_>) -> Vec<FeaDiagnostic> {
    let ThermalDiagnosticsInput {
        model,
        initial,
        assembly,
        last_solve,
        temperature_snapshots,
        residual_norms,
        energy_balances,
        residual_target,
    } = input;
    let maximum_residual = residual_norms.iter().copied().fold(0.0_f64, f64::max);
    let maximum_energy_residual = energy_balances
        .iter()
        .map(|balance| balance.relative_residual)
        .fold(0.0_f64, f64::max);
    let minimum_temperature_k = temperature_snapshots
        .iter()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let maximum_temperature_k = temperature_snapshots
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let final_energy = energy_balances
        .last()
        .copied()
        .unwrap_or(ThermalEnergyBalance {
            storage_rate_w: 0.0,
            boundary_loss_w: 0.0,
            applied_power_w: 0.0,
            prescribed_boundary_power_w: 0.0,
            residual_w: 0.0,
            relative_residual: 0.0,
        });
    let constitutive = constitutive_stats(model);
    let mut diagnostics = vec![
        FeaDiagnostic {
            code: "FEA_THERMAL_INITIAL_STATE".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "procedure_step_id={} reference_temperature_k={} applied_initial_condition_count={} authored_node_count={} prescribed_node_count={} authored_and_prescribed_node_count={} reference_node_count={} minimum_temperature_k={} maximum_temperature_k={}",
                initial.procedure_step_id,
                initial.reference_temperature_k,
                initial.applied_initial_condition_ids.len(),
                initial.authored_node_count,
                initial.prescribed_node_count,
                initial.authored_and_prescribed_node_count,
                initial.reference_node_count,
                initial.minimum_temperature_k,
                initial.maximum_temperature_k,
            ),
        },
        FeaDiagnostic {
            code: "FEA_THERMAL_STABILITY".into(),
            severity: if maximum_residual <= residual_target {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "all_linear_solves_converged=true max_relative_residual_norm={} residual_target={} minimum_temperature_k={} maximum_temperature_k={}",
                maximum_residual,
                residual_target,
                minimum_temperature_k,
                maximum_temperature_k,
            ),
        },
        FeaDiagnostic {
            code: "FEA_THERMAL_CONSTITUTIVE".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "conductivity_mean={} heat_capacity_mean={} density_mean={} conductivity_spread_ratio={} heat_capacity_spread_ratio={}",
                constitutive.conductivity_mean,
                constitutive.heat_capacity_mean,
                constitutive.density_mean,
                constitutive.conductivity_spread_ratio,
                constitutive.heat_capacity_spread_ratio,
            ),
        },
        FeaDiagnostic {
            code: "FEA_THERMAL_ENERGY_BALANCE".into(),
            severity: if maximum_energy_residual <= residual_target {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Error
            },
            message: format!(
                "storage_rate_w={} boundary_loss_w={} applied_power_w={} prescribed_boundary_power_w={} residual_w={} final_relative_residual={} max_relative_residual={}",
                final_energy.storage_rate_w,
                final_energy.boundary_loss_w,
                final_energy.applied_power_w,
                final_energy.prescribed_boundary_power_w,
                final_energy.residual_w,
                final_energy.relative_residual,
                maximum_energy_residual,
            ),
        },
        FeaDiagnostic {
            code: "FEA_THERMAL_FIELD_RECOVERY".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "basis=physical_quadrature node_count={} element_count={} boundary_face_count={}",
                assembly.projection.nodes.len(),
                assembly.projection.elements.len(),
                assembly.projection.boundary_faces.len(),
            ),
        },
    ];
    diagnostics.extend(material_assignment_diagnostics(&model.material_assignments));
    diagnostics.extend(last_solve.diagnostics.clone());
    diagnostics
}
