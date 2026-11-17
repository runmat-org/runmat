use runmat_analysis_core::AnalysisField;
use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};
mod solved;
pub(crate) use solved::recover_solved_static_fields;

use crate::{
    assembly::AssemblySummary,
    contracts::{
        fea_thermo_mechanical_coupling_residual_field_id,
        fea_thermo_mechanical_displacement_field_id, fea_thermo_mechanical_temperature_field_id,
        fea_thermo_mechanical_thermal_strain_field_id,
        fea_thermo_mechanical_thermal_stress_field_id, fea_thermo_mechanical_von_mises_field_id,
    },
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
};

const TENSOR_COMPONENT_COUNT: usize = 6;
const NORMAL_COMPONENT_COUNT: usize = 3;

#[derive(Default)]
pub(crate) struct ThermoMechanicalSnapshotFields {
    pub(crate) temperature_snapshots: Vec<AnalysisField>,
    pub(crate) thermal_strain_snapshots: Vec<AnalysisField>,
    pub(crate) thermal_stress_snapshots: Vec<AnalysisField>,
    pub(crate) displacement_snapshots: Vec<AnalysisField>,
    pub(crate) von_mises_snapshots: Vec<AnalysisField>,
    pub(crate) coupling_residual_snapshots: Vec<AnalysisField>,
    pub(crate) diagnostics: Vec<FeaDiagnostic>,
}

/// Recover the prescribed-temperature quantities used by the physical solid operator. The
/// structural stress and von Mises fields supplied here have already subtracted element thermal
/// eigenstrain; the thermal-stress snapshot is the positive constitutive eigenstress D εₜₕ.
pub(crate) fn recover_physical_static_thermo_mechanical_fields(
    assembly: &AssemblySummary,
    displacement: &[AnalysisField],
    von_mises: &[AnalysisField],
    residual_norm: f64,
) -> Result<ThermoMechanicalSnapshotFields, String> {
    let coupling = assembly
        .thermo_mechanical
        .as_ref()
        .ok_or("physical thermoelastic recovery requires a coupling summary")?;
    let element_count = assembly.structural_solid_recovery.len();
    if element_count == 0 {
        return Err("physical thermoelastic recovery requires solid elements".into());
    }
    let mut element_temperatures_k = Vec::with_capacity(element_count);
    let mut thermal_strain = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
    let mut thermal_stress = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
    for (index, element) in assembly.structural_solid_recovery.iter().enumerate() {
        let temperature_k = element.prescribed_temperature_k.ok_or_else(|| {
            format!(
                "physical thermoelastic element {} lacks prescribed temperature",
                element.element_id
            )
        })?;
        if !temperature_k.is_finite() || temperature_k.canonical_value() <= 0.0 {
            return Err(format!(
                "physical thermoelastic element {} has invalid prescribed temperature",
                element.element_id
            ));
        }
        element_temperatures_k.push(temperature_k.canonical_value());
        let eigenstrain = element.thermal_eigenstrain.ok_or_else(|| {
            format!(
                "physical thermoelastic element {} lacks thermal eigenstrain",
                element.element_id
            )
        })?;
        let stress = element.material.youngs_modulus_pa.canonical_value() * eigenstrain
            / (1.0 - 2.0 * element.material.poisson_ratio.canonical_value());
        if !stress.is_finite() {
            return Err(format!(
                "physical thermoelastic element {} has invalid thermal eigenstress",
                element.element_id
            ));
        }
        for component in 0..NORMAL_COMPONENT_COUNT {
            thermal_strain[index * TENSOR_COMPONENT_COUNT + component] = eigenstrain;
            thermal_stress[index * TENSOR_COMPONENT_COUNT + component] = stress;
        }
    }
    let mut fields = ThermoMechanicalSnapshotFields {
        temperature_snapshots: vec![AnalysisField::host_f64(
            fea_thermo_mechanical_temperature_field_id(0), AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Element).step(0),
            vec![element_count],
            element_temperatures_k,
        )],
        thermal_strain_snapshots: vec![AnalysisField::host_f64(
            fea_thermo_mechanical_thermal_strain_field_id(0), AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Strain, FieldLocation::Element).step(0),
            vec![element_count, TENSOR_COMPONENT_COUNT],
            thermal_strain,
        )],
        thermal_stress_snapshots: vec![AnalysisField::host_f64(
            fea_thermo_mechanical_thermal_stress_field_id(0), AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element).step(0),
            vec![element_count, TENSOR_COMPONENT_COUNT],
            thermal_stress,
        )],
        coupling_residual_snapshots: vec![AnalysisField::host_f64(
            fea_thermo_mechanical_coupling_residual_field_id(0), AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global).step(0),
            vec![1],
            vec![residual_norm],
        )],
        diagnostics: vec![
            FeaDiagnostic {
                code: "FEA_TM_CONSISTENCY".into(),
                severity: FeaDiagnosticSeverity::Info,
                message: format!(
                    "constitutive_relation=element_owned_thermal_eigenstress element_count={element_count} thermal_state_source=prescribed_element_constant region_delta_count={} sequence_state_transfer=false",
                    coupling.region_delta_count
                ),
            },
            FeaDiagnostic {
                code: "FEA_TM_TEMPERATURE_FIELD".into(),
                severity: FeaDiagnosticSeverity::Info,
                message: format!(
                    "temperature_field_basis=element_constant element_count={element_count} region_delta_count={}",
                    coupling.region_delta_count
                ),
            },
        ],
        ..ThermoMechanicalSnapshotFields::default()
    };
    fields.displacement_snapshots = displacement
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, mut field)| {
            field.field_id = fea_thermo_mechanical_displacement_field_id(index);
            field
        })
        .collect();
    fields.von_mises_snapshots = von_mises
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, mut field)| {
            field.field_id = fea_thermo_mechanical_von_mises_field_id(index);
            field
        })
        .collect();
    Ok(fields)
}
