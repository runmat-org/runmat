use runmat_analysis_core::{AnalysisModel, IndependentDomain, PhysicsFamily, SignalSampleApproach};

use crate::{
    contracts::{
        fea_thermal_energy_storage_rate_field_id, fea_thermal_heat_flux_field_id,
        fea_thermal_temperature_field_id, fea_thermal_temperature_gradient_field_id,
        ComputeBackend, ExecutionBackend, FeaRunError, FeaRunResult, FeaThermalRunResult,
        ThermalProcedureControl, ThermalSolveOptions,
    },
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    physics::thermal::{
        assembly::{assemble as assemble_thermal_operator, ThermalEnergyBalance},
        initial_state,
    },
    pipeline::observation_history::{
        evaluate_field_histories, HistoryFieldSeries, HistoryTopology,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::{
        backend::{
            build_backend, kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend,
        },
        linear::{solve_operator_system_scoped, LinearSolveResult},
        preconditioner::SpdPreconditionerKind,
    },
};

mod diagnostics;
mod snapshots;
use diagnostics::{diagnostics, ThermalDiagnosticsInput};
use snapshots::{scalar_snapshots, vector_snapshots};

pub(super) fn run(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ThermalSolveOptions,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    thermal_step: &runmat_analysis_core::ProcedureStep,
    schedule: &crate::execution::ThermalSolveSchedule,
    facts: &crate::planning::resource::ThermalPreassemblyResourceFacts,
) -> Result<FeaThermalRunResult, FeaRunError> {
    let operators = &facts.operators;
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
            operators,
        )?,
        ThermalProcedureControl::BackwardEuler { .. } => solve_transient(TransientSolveInput {
            properties: &facts.properties,
            model: &active_model,
            mesh,
            step: thermal_step,
            initial_assembly: &initial_assembly,
            initial_temperature_k: &initial.temperature_k,
            schedule,
            operators,
            requested_solver_backend,
            algebra: algebra.as_ref(),
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
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
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
        let definitions = crate::physics::thermal::history_fields::definitions()
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let series = definitions
            .iter()
            .zip([
                temperature_snapshots.as_slice(),
                temperature_gradient_snapshots.as_slice(),
                heat_flux_snapshots.as_slice(),
                energy_storage_rate_snapshots.as_slice(),
            ])
            .map(|(definition, snapshots)| HistoryFieldSeries {
                quantity: definition.quantity(),
                source_field_id: definition.field_id(),
                location: definition.location(),
                entity_domain: definition.entity_domain(),
                value_shape: definition.value_shape().clone(),
                numeric_representation: definition.numeric_representation(),
                snapshots,
            })
            .collect::<Vec<_>>();
        let evaluated_histories = evaluate_field_histories(
            thermal_step,
            HistoryTopology::Solid(&initial_assembly.projection),
            &coordinates,
            None,
            &series,
        )
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        run.observation_histories = evaluated_histories.histories;
        run.series_reductions = evaluated_histories.series_reductions;
    }

    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA physical thermal run complete",
        Some(5),
        Some(5),
    );
    run.validate_solver_formulation()?;
    let sequence = crate::execution::SequenceIdentity::from_solver_contract(&model.sequence)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let producer = crate::execution::StepIdentity::from_solver_contract(sequence, thermal_step)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    Ok(FeaThermalRunResult {
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
    operators: &crate::planning::resource::ThermalOperatorResourceFacts,
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
        operators,
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
    properties: &'a crate::properties::CompiledPropertyMap,
    model: &'a AnalysisModel,
    mesh: &'a runmat_meshing_core::SolverMeshArtifact,
    step: &'a runmat_analysis_core::ProcedureStep,
    initial_assembly: &'a crate::physics::thermal::assembly::ThermalOperatorAssembly,
    initial_temperature_k: &'a [f64],
    schedule: &'a crate::execution::ThermalSolveSchedule,
    operators: &'a crate::planning::resource::ThermalOperatorResourceFacts,
    requested_solver_backend: LinearAlgebraBackendKind,
    algebra: &'a dyn LinearAlgebraBackend,
}

fn solve_transient(input: TransientSolveInput<'_>) -> Result<ProcedureOutcome, FeaRunError> {
    let solve_times = input.schedule.points_s();
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
        solve_times[1] - solve_times[0],
        &mut recovered,
    )?;
    time_points_s.push(solve_times[0]);
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
        let approach = input.schedule.approach(step_index);
        let sample = sample_point(input.step, time_s, approach)?;
        let excitations = realize_excitations_for_step(input.model, input.step, sample)?;
        let assembly = assemble_thermal_operator(
            input.model,
            input.mesh,
            input.properties,
            input.operators.support(),
            &excitations,
            sample,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        drop(excitations);
        input.operators.verify_matrix(&assembly.conductivity)?;
        input.operators.verify_matrix(&assembly.capacity)?;
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
            input.operators,
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
    facts: &crate::planning::resource::ThermalOperatorResourceFacts,
) -> Result<LinearSolveResult, FeaRunError> {
    for matrix in [
        operator.operators().stiffness(),
        operator.operators().mass(),
        operator.operators().damping(),
    ] {
        facts.verify_matrix(matrix)?;
    }
    let solve = solve_operator_system_scoped(
        operator,
        SpdPreconditionerKind::Jacobi,
        requested_solver_backend,
        algebra,
        initial_guess,
        crate::planning::resource::LocalExecutionPlan::serial(),
        Some(facts.solver_admission()),
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
