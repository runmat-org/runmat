//! Integration-point output from the retained solved temperature field.

use super::*;
use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};

pub(crate) fn recover_solved_static_fields(
    assembly: &AssemblySummary,
    displacement: &[AnalysisField],
    von_mises: &[AnalysisField],
    residual_norm: f64,
) -> Result<ThermoMechanicalSnapshotFields, String> {
    let count = assembly.structural_solid_recovery.len();
    let mut temperature = Vec::with_capacity(count * 4);
    let mut strain = Vec::with_capacity(count * 24);
    let mut stress = Vec::with_capacity(count * 24);
    for element in &assembly.structural_solid_recovery {
        let thermal = element
            .solved_thermal_quadrature
            .as_ref()
            .ok_or("solved thermal recovery requires a complete element field")?;
        for point in 0..4 {
            temperature.push(thermal.temperature_k.canonical_values()[point]);
            let eigenstrain = thermal.eigenstrain[point];
            let eigenstress = element.material.youngs_modulus_pa.canonical_value() * eigenstrain
                / (1.0 - 2.0 * element.material.poisson_ratio.canonical_value());
            strain.extend([eigenstrain, eigenstrain, eigenstrain, 0.0, 0.0, 0.0]);
            stress.extend([eigenstress, eigenstress, eigenstress, 0.0, 0.0, 0.0]);
        }
    }
    Ok(ThermoMechanicalSnapshotFields {
        temperature_snapshots: vec![AnalysisField::host_f64(
            fea_thermo_mechanical_temperature_field_id(0), AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::IntegrationPoint).step(0), vec![count * 4], temperature,
        )],
        thermal_strain_snapshots: vec![AnalysisField::host_f64(
            fea_thermo_mechanical_thermal_strain_field_id(0), AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Strain, FieldLocation::IntegrationPoint).step(0), vec![count * 4, 6], strain,
        )],
        thermal_stress_snapshots: vec![AnalysisField::host_f64(
            fea_thermo_mechanical_thermal_stress_field_id(0), AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::IntegrationPoint).step(0), vec![count * 4, 6], stress,
        )],
        displacement_snapshots: displacement.iter().cloned().enumerate().map(|(index, mut field)| {
            field.field_id = fea_thermo_mechanical_displacement_field_id(index); field
        }).collect(),
        von_mises_snapshots: von_mises.iter().cloned().enumerate().map(|(index, mut field)| {
            field.field_id = fea_thermo_mechanical_von_mises_field_id(index); field
        }).collect(),
        coupling_residual_snapshots: vec![AnalysisField::host_f64(
            fea_thermo_mechanical_coupling_residual_field_id(0), AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global).step(0), vec![1], vec![residual_norm],
        )],
        diagnostics: vec![FeaDiagnostic {
            code: "FEA_TM_CONSISTENCY".into(), severity: FeaDiagnosticSeverity::Info,
            message: format!("thermal_state_source=retained_solved_nodal_field temperature_field_basis=solid_integration_point element_count={count} sequence_state_transfer=true coupling=one_way residual=structural_linear_solve"),
        }],
    })
}
