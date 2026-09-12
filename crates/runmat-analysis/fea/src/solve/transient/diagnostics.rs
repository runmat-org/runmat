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
            severity: if maximum_total_energy.is_finite() {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "maximum_physical_strain_plus_kinetic_energy_joules={maximum_total_energy}"
            ),
        },
    ]
}

/// Describes the mechanical-energy history without claiming a conservation balance.
///
/// External work and damping dissipation are not integrated by this solver result, so the
/// diagnostic intentionally makes no stability or conservation judgment from energy growth.
pub(crate) fn mechanical_energy_history_diagnostic(
    kinetic_energy: &[f64],
    strain_energy: &[f64],
) -> FeaDiagnostic {
    let snapshot_count = kinetic_energy.len().max(strain_energy.len());
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
    let metrics_are_finite = total_energy.iter().all(|value| value.is_finite())
        && minimum_total_energy.is_finite()
        && maximum_total_energy.is_finite()
        && maximum_step_change.is_finite();

    FeaDiagnostic {
        code: "FEA_TRANSIENT_MECHANICAL_ENERGY_HISTORY".to_string(),
        severity: if metrics_are_finite {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "snapshot_count={snapshot_count} initial_strain_plus_kinetic_energy_joules={initial_total_energy} final_strain_plus_kinetic_energy_joules={final_total_energy} minimum_strain_plus_kinetic_energy_joules={minimum_total_energy} maximum_strain_plus_kinetic_energy_joules={maximum_total_energy} maximum_step_change_joules={maximum_step_change} conservation_balance_evaluated=false"
        ),
    }
}
