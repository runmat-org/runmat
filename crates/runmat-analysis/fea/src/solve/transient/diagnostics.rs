use crate::{
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::backend::kind::LinearAlgebraBackendKind,
};

use super::TransientSolveOptions;

pub(super) fn transient_diagnostics(
    options: &TransientSolveOptions,
    accepted_time_steps_s: &[f64],
    residual_norms: &[f64],
    strain_energy: &[f64],
    kinetic_energy: &[f64],
    relative_energy_balance_error: &[f64],
    backend: LinearAlgebraBackendKind,
) -> Vec<FeaDiagnostic> {
    let maximum_residual = residual_norms.iter().copied().fold(0.0_f64, f64::max);
    let minimum_time_step = accepted_time_steps_s
        .iter()
        .copied()
        .reduce(f64::min)
        .unwrap_or(0.0);
    let maximum_time_step = accepted_time_steps_s
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(0.0);
    let maximum_total_energy = strain_energy
        .iter()
        .zip(kinetic_energy)
        .map(|(strain, kinetic)| strain + kinetic)
        .fold(0.0_f64, f64::max);
    let maximum_relative_energy_balance_error = relative_energy_balance_error
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    vec![
        FeaDiagnostic {
            code: "FEA_TRANSIENT_METHOD".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "solver=newmark_average_acceleration_pcg beta=0.25 gamma=0.5 backend={backend}"
            ),
        },
        FeaDiagnostic {
            code: "FEA_TRANSIENT_CONVERGENCE".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "nominal_interval_count={} accepted_step_count={} rejected_step_count=0",
                options.time_step_policy.interval_count(),
                accepted_time_steps_s.len()
            ),
        },
        FeaDiagnostic {
            code: "FEA_TRANSIENT_STABILITY".into(),
            severity: if maximum_residual <= options.residual_target {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "time_step_policy=fixed dt_min={minimum_time_step} dt_max={maximum_time_step} max_projected_dynamic_residual={maximum_residual}"
            ),
        },
        FeaDiagnostic {
            code: "FEA_TRANSIENT_ENERGY".into(),
            severity: if maximum_total_energy.is_finite()
                && maximum_relative_energy_balance_error.is_finite()
                && maximum_relative_energy_balance_error <= 1.0e-6
            {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "maximum_physical_strain_plus_kinetic_energy_joules={maximum_total_energy} maximum_relative_energy_balance_error={maximum_relative_energy_balance_error} conservation_balance_evaluated=true"
            ),
        },
    ]
}

pub(crate) fn mechanical_energy_history_diagnostic(
    kinetic_energy: &[f64],
    strain_energy: &[f64],
    external_work: &[f64],
    support_work: &[f64],
    damping_dissipation: &[f64],
    energy_balance_residual: &[f64],
    relative_energy_balance_error: &[f64],
) -> FeaDiagnostic {
    let snapshot_count = [
        kinetic_energy.len(),
        strain_energy.len(),
        external_work.len(),
        support_work.len(),
        damping_dissipation.len(),
        energy_balance_residual.len(),
        relative_energy_balance_error.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let total_energy = (0..snapshot_count)
        .map(|index| {
            kinetic_energy.get(index).copied().unwrap_or(f64::NAN)
                + strain_energy.get(index).copied().unwrap_or(f64::NAN)
        })
        .collect::<Vec<_>>();
    let initial_total_energy = total_energy.first().copied().unwrap_or(0.0);
    let final_total_energy = total_energy.last().copied().unwrap_or(0.0);
    let minimum_total_energy = total_energy.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum_total_energy = total_energy
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let maximum_step_change = total_energy
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .fold(0.0_f64, f64::max);
    let final_external_work = external_work.last().copied().unwrap_or(f64::NAN);
    let final_support_work = support_work.last().copied().unwrap_or(f64::NAN);
    let final_damping_dissipation = damping_dissipation.last().copied().unwrap_or(f64::NAN);
    let maximum_absolute_balance_residual = energy_balance_residual
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let maximum_relative_balance_error = relative_energy_balance_error
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let metrics_are_finite = total_energy.iter().all(|value| value.is_finite())
        && minimum_total_energy.is_finite()
        && maximum_total_energy.is_finite()
        && maximum_step_change.is_finite()
        && final_external_work.is_finite()
        && final_support_work.is_finite()
        && final_damping_dissipation.is_finite()
        && maximum_absolute_balance_residual.is_finite()
        && maximum_relative_balance_error.is_finite();

    FeaDiagnostic {
        code: "FEA_TRANSIENT_MECHANICAL_ENERGY_HISTORY".to_string(),
        severity: if metrics_are_finite {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "snapshot_count={snapshot_count} initial_strain_plus_kinetic_energy_joules={initial_total_energy} final_strain_plus_kinetic_energy_joules={final_total_energy} minimum_strain_plus_kinetic_energy_joules={minimum_total_energy} maximum_strain_plus_kinetic_energy_joules={maximum_total_energy} maximum_step_change_joules={maximum_step_change} final_external_work_joules={final_external_work} final_support_work_joules={final_support_work} final_damping_dissipation_joules={final_damping_dissipation} maximum_absolute_energy_balance_residual_joules={maximum_absolute_balance_residual} maximum_relative_energy_balance_error={maximum_relative_balance_error} conservation_balance_evaluated=true"
        ),
    }
}
