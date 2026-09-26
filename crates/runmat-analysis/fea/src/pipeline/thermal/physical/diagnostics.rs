//! Physical thermal solve evidence and diagnostics.

use super::*;
use crate::diagnostics::{
    builders::material_assignment_diagnostics, thermal::ThermalDiagnosticMessage, FeaDiagnostic,
    FeaDiagnosticSeverity,
};

pub(super) struct ThermalDiagnosticsInput<'a> {
    pub(super) metadata: &'a crate::planning::resource::ThermalMetadataResourceFacts,
    pub(super) model: &'a AnalysisModel,
    pub(super) initial: &'a crate::contracts::ThermalInitialStateEvidence,
    pub(super) assembly: &'a crate::physics::thermal::assembly::ThermalOperatorAssembly,
    pub(super) last_solve: &'a LinearSolveResult,
    pub(super) temperature_snapshots: &'a [Vec<f64>],
    pub(super) residual_norms: &'a [f64],
    pub(super) energy_balances: &'a [ThermalEnergyBalance],
    pub(super) residual_target: f64,
}

pub(super) fn diagnostics(
    input: ThermalDiagnosticsInput<'_>,
) -> Result<Vec<FeaDiagnostic>, FeaRunError> {
    let ThermalDiagnosticsInput {
        metadata,
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
    let thermal = [
        ThermalDiagnosticMessage::Initial {
            step: &initial.procedure_step_id,
            temperatures: [
                initial.reference_temperature_k,
                initial.minimum_temperature_k,
                initial.maximum_temperature_k,
            ],
            counts: [
                initial.applied_initial_condition_ids.len(),
                initial.authored_node_count,
                initial.prescribed_node_count,
                initial.authored_and_prescribed_node_count,
                initial.reference_node_count,
            ],
        }
        .diagnostic(FeaDiagnosticSeverity::Info),
        ThermalDiagnosticMessage::Stability([
            maximum_residual,
            residual_target,
            minimum_temperature_k,
            maximum_temperature_k,
        ])
        .diagnostic(if maximum_residual <= residual_target {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        }),
        ThermalDiagnosticMessage::Constitutive {
            digest: assembly.property_identity.semantic_digest,
            values: [
                constitutive.conductivity_mean,
                constitutive.heat_capacity_mean,
                constitutive.density_mean,
                constitutive.conductivity_spread_ratio,
                constitutive.heat_capacity_spread_ratio,
            ],
        }
        .diagnostic(FeaDiagnosticSeverity::Info),
        ThermalDiagnosticMessage::Energy([
            final_energy.storage_rate_w,
            final_energy.boundary_loss_w,
            final_energy.applied_power_w,
            final_energy.prescribed_boundary_power_w,
            final_energy.residual_w,
            final_energy.relative_residual,
            maximum_energy_residual,
        ])
        .diagnostic(if maximum_energy_residual <= residual_target {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Error
        }),
        ThermalDiagnosticMessage::Recovery([
            assembly.projection.nodes.len(),
            assembly.projection.elements.len(),
            assembly.projection.boundary_faces.len(),
        ])
        .diagnostic(FeaDiagnosticSeverity::Info),
    ];
    metadata.verify_physical_diagnostics(&thermal)?;
    let mut diagnostics = Vec::from(thermal);
    diagnostics.extend(material_assignment_diagnostics(&model.material_assignments));
    diagnostics.extend(last_solve.diagnostics.clone());
    Ok(diagnostics)
}
