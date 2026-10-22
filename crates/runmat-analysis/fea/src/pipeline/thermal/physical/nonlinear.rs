//! Bounded damped Newton for the original physical gray-radiation equation.
use super::*;
use crate::contracts::{
    ThermalNonlinearSampleEvidence, THERMAL_MAXIMUM_BACKTRACKING_TRIALS,
    THERMAL_MAXIMUM_NEWTON_CORRECTIONS,
};
use crate::physics::thermal::assembly::ThermalOperatorAssembly;
use runmat_analysis_core::BoundaryOperator;

const MAX_CORRECTIONS: usize = THERMAL_MAXIMUM_NEWTON_CORRECTIONS;
const MAX_TRIALS: usize = THERMAL_MAXIMUM_BACKTRACKING_TRIALS;

pub(super) fn has_radiation(model: &AnalysisModel) -> bool {
    model
        .boundaries
        .iter()
        .any(|boundary| matches!(boundary.operator, BoundaryOperator::ThermalRadiation { .. }))
}

pub(super) struct NonlinearSolveInput<'a> {
    pub assembly: &'a ThermalOperatorAssembly,
    pub model: &'a AnalysisModel,
    pub sample: ExcitationSamplePoint<'a>,
    pub reference_temperature_k: &'a [f64],
    pub initial_departure_k: &'a [f64],
    pub previous_departure_k: Option<&'a [f64]>,
    pub time_step_s: f64,
    pub time_s: f64,
    pub algebra: &'a dyn LinearAlgebraBackend,
    pub facts: &'a crate::planning::resource::ThermalPreassemblyResourceFacts,
    pub residual_target: f64,
}

pub(super) struct NonlinearOutcome {
    pub solve: LinearSolveResult,
    pub relative_linear_residual: f64,
    pub energy_balance: ThermalEnergyBalance,
    pub evidence: ThermalNonlinearSampleEvidence,
    pub host_sync_count: u32,
    pub device_apply_count: u32,
    pub device_apply_attempt_count: u32,
    pub accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
}

fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0_f64, f64::hypot)
}

fn relative(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        if numerator == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        numerator / denominator
    }
}

pub(super) fn add_count(total: &mut u32, count: u32) -> Result<(), FeaRunError> {
    *total = total
        .checked_add(count)
        .ok_or_else(|| FeaRunError::Solve("thermal nonlinear backend count overflow".into()))?;
    Ok(())
}

pub(super) fn solve(input: NonlinearSolveInput<'_>) -> Result<NonlinearOutcome, FeaRunError> {
    if !input.assembly.relations.is_empty() {
        return Err(FeaRunError::Assembly(
            "thermal radiation with tie, periodic or temperature-jump constraints needs the reduced Newton residual, which the radiation route does not yet form".into(),
        ));
    }
    let mut current = input.initial_departure_k.to_vec();
    for (node, prescribed) in input.assembly.prescribed_temperature_k.iter().enumerate() {
        if let Some(value) = prescribed {
            current[node] = value - input.reference_temperature_k[node];
        }
    }
    let evaluate = |temperature: &[f64]| {
        input.assembly.nonlinear_residual(
            input.model,
            input.sample,
            input.reference_temperature_k,
            temperature,
            input.previous_departure_k,
            input.time_step_s,
        )
    };
    let free_norm = |residual: &[f64]| {
        norm(residual.iter().enumerate().filter_map(|(node, value)| {
            input.assembly.prescribed_temperature_k[node]
                .is_none()
                .then_some(*value)
        }))
    };
    let mut residual =
        evaluate(&current).map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let mut backtracking_count = 0;
    let mut maximum_linear_residual = 0.0_f64;
    let mut host_sync_count = 0;
    let mut device_apply_count = 0;
    let mut device_apply_attempt_count = 0;
    let mut accelerator_evidence = Vec::new();
    for iteration_count in 1..=MAX_CORRECTIONS {
        check_cancelled("fea.run_thermal")?;
        let operator = input
            .assembly
            .nonlinear_increment_operator(
                input.model,
                input.sample,
                input.reference_temperature_k,
                &current,
                input.previous_departure_k,
                input.time_step_s,
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        let linear_rhs_norm = rhs_norm(&operator.linear_rhs());
        // Tightening the correction equation may be needed for the declared physical power
        // closure. It never relaxes either the original stopping policy or acceptance target.
        let linear_target = if residual.physical_power_scale_w > 0.0 {
            input.residual_target.min(
                0.5 * input.residual_target * residual.physical_power_scale_w
                    / ((current.len() as f64).sqrt() * linear_rhs_norm),
            )
        } else {
            input.residual_target
        };
        let mut correction = solve_checked(
            &operator,
            LinearAlgebraBackendKind::CpuReference,
            input.algebra,
            None,
            &input.facts.operators,
            linear_target,
            None,
            &[],
        )?;
        maximum_linear_residual = maximum_linear_residual.max(linear_residual(&correction)?);
        add_count(&mut host_sync_count, correction.host_sync_count)?;
        add_count(&mut device_apply_count, correction.device_apply_k_count)?;
        add_count(
            &mut device_apply_attempt_count,
            correction.device_apply_k_attempt_count,
        )?;
        backend_evidence::retain(
            &mut accelerator_evidence,
            &mut correction.accelerator_evidence,
            &input.facts.metadata.backend_evidence,
        )?;
        let before_norm = free_norm(&residual.residual_w);
        let mut alpha = 1.0;
        let mut accepted = None;
        for trial in 0..MAX_TRIALS {
            let mut candidate = current
                .iter()
                .zip(&correction.solution)
                .map(|(temperature, delta)| temperature + alpha * delta)
                .collect::<Vec<_>>();
            // Known prescribed temperatures remain exact even during backtracking.
            for (node, prescribed) in input.assembly.prescribed_temperature_k.iter().enumerate() {
                if let Some(value) = prescribed {
                    candidate[node] = value - input.reference_temperature_k[node];
                }
            }
            // The physical evaluator checks Kelvin at actual radiation quadrature points;
            // nodal positivity alone would be insufficient for Tri6 interpolation.
            let trial_residual = match evaluate(&candidate) {
                Ok(value) => Some(value),
                Err(crate::physics::thermal::assembly::ThermalAssemblyError::Radiation(
                    crate::physics::thermal::radiation::RadiationError::InvalidState,
                )) => None,
                Err(error) => return Err(FeaRunError::Assembly(error.to_string())),
            };
            if let Some(trial_residual) = trial_residual {
                let after_norm = free_norm(&trial_residual.residual_w);
                let equilibrium = relative(after_norm, trial_residual.physical_power_scale_w);
                if after_norm.is_finite()
                    && equilibrium.is_finite()
                    && (after_norm <= (1.0 - 1.0e-4 * alpha) * before_norm
                        || equilibrium <= input.residual_target)
                {
                    accepted = Some((candidate, trial_residual, equilibrium));
                    break;
                }
            }
            if trial + 1 < MAX_TRIALS {
                backtracking_count += 1;
            }
            alpha *= 0.5;
        }
        let (candidate, trial_residual, equilibrium) = accepted.ok_or_else(|| {
            FeaRunError::Solve(format!(
                "thermal radiation line search exhausted {MAX_TRIALS} trials"
            ))
        })?;
        // Increment scale is the solved departure coordinate, not an absolute-Kelvin floor.
        let increment = relative(
            norm(candidate.iter().zip(&current).map(|(a, b)| a - b)),
            norm(candidate.iter().copied()).max(norm(current.iter().copied())),
        );
        let reference = input.reference_temperature_k[0];
        // Energy recovery accepts one component-constant reference; steady disconnected
        // bodies can use different references, so convert that case to absolute coordinates.
        let balance = if let Some(previous) = input.previous_departure_k {
            input.assembly.energy_balance(
                input.model,
                input.sample,
                previous,
                &candidate,
                input.time_step_s,
                reference,
            )
        } else {
            let absolute = candidate
                .iter()
                .zip(input.reference_temperature_k)
                .map(|(value, reference)| value + reference)
                .collect::<Vec<_>>();
            input
                .assembly
                .energy_balance(input.model, input.sample, &absolute, &absolute, 1.0, 0.0)
        }
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        balance
            .validate()
            .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        current = candidate;
        residual = trial_residual;
        if equilibrium <= input.residual_target
            && increment <= input.residual_target
            && balance.relative_residual <= input.residual_target
        {
            correction.solution = current;
            return Ok(NonlinearOutcome {
                solve: correction,
                relative_linear_residual: maximum_linear_residual,
                energy_balance: balance,
                evidence: ThermalNonlinearSampleEvidence {
                    time_s: input.time_s,
                    iteration_count,
                    backtracking_count,
                    relative_equilibrium_residual: equilibrium,
                    relative_increment: increment,
                    power_scale_w: residual.physical_power_scale_w,
                },
                host_sync_count,
                device_apply_count,
                device_apply_attempt_count,
                accelerator_evidence,
            });
        }
    }
    Err(FeaRunError::Solve(format!(
        "thermal radiation Newton exhausted {MAX_CORRECTIONS} corrections"
    )))
}

pub(super) fn solve_steady(input: SteadySolveInput<'_>) -> Result<ProcedureOutcome, FeaRunError> {
    let references = input
        .assembly
        .steady_reference_temperatures(input.model, input.sample, input.initial_temperature_k[0])
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let initial = input
        .initial_temperature_k
        .iter()
        .zip(&references)
        .map(|(temperature, reference)| temperature - reference)
        .collect::<Vec<_>>();
    let mut accepted = solve(NonlinearSolveInput {
        assembly: input.assembly,
        model: input.model,
        sample: input.sample,
        reference_temperature_k: &references,
        initial_departure_k: &initial,
        previous_departure_k: None,
        time_step_s: 1.0,
        time_s: input.time_origin_s,
        algebra: input.algebra,
        facts: input.facts,
        residual_target: input.residual_target,
    })?;
    for (node, temperature) in accepted.solve.solution.iter_mut().enumerate() {
        *temperature = input.assembly.prescribed_temperature_k[node]
            .unwrap_or(*temperature + references[node]);
    }
    let mut recovered = RecoveredSnapshots::with_capacity(1);
    push_recovered_snapshot(
        input.assembly,
        &accepted.solve.solution,
        None,
        1.0,
        0.0,
        &mut recovered,
    )?;
    Ok(ProcedureOutcome {
        time_points_s: vec![input.time_origin_s],
        recovered,
        residual_norms: vec![accepted.relative_linear_residual],
        energy_balances: vec![accepted.energy_balance],
        nonlinear_evidence: vec![accepted.evidence],
        source_evidence: input.assembly.source_evidence.clone(),
        host_sync_count: accepted.host_sync_count,
        device_apply_count: accepted.device_apply_count,
        device_apply_attempt_count: accepted.device_apply_attempt_count,
        accelerator_evidence: accepted.accelerator_evidence,
        last_solve: accepted.solve,
    })
}

#[cfg(test)]
mod tests;
