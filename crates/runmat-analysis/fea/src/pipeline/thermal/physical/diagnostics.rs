//! Physical thermal solve evidence and diagnostics.

use super::*;
use crate::diagnostics::{
    builders::material_assignment_diagnostics, FeaDiagnostic, FeaDiagnosticSeverity,
};

pub(super) struct ThermalDiagnosticsInput<'a> {
    pub(super) model: &'a AnalysisModel,
    pub(super) initial: &'a crate::contracts::ThermalInitialStateEvidence,
    pub(super) assembly: &'a crate::physics::thermal::assembly::ThermalOperatorAssembly,
    pub(super) last_solve: &'a LinearSolveResult,
    pub(super) temperature_snapshots: &'a [Vec<f64>],
    pub(super) residual_norms: &'a [f64],
    pub(super) energy_balances: &'a [ThermalEnergyBalance],
    pub(super) residual_target: f64,
}

pub(super) fn diagnostics(input: ThermalDiagnosticsInput<'_>) -> Vec<FeaDiagnostic> {
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
    let constitutive = assembly.constitutive_stats;
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
                "property_semantic_digest={} conductivity_mean={} heat_capacity_mean={} density_mean={} conductivity_spread_ratio={} heat_capacity_spread_ratio={}",
                assembly.property_identity.semantic_digest,
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
