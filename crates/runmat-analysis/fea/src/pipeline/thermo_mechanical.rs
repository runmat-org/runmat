use runmat_analysis_core::AnalysisField;

use crate::{
    assembly::ThermoMechanicalAssemblySummary,
    contracts::{
        fea_thermo_mechanical_coupling_residual_field_id,
        fea_thermo_mechanical_displacement_field_id, fea_thermo_mechanical_temperature_field_id,
        fea_thermo_mechanical_thermal_strain_field_id,
        fea_thermo_mechanical_thermal_stress_field_id, fea_thermo_mechanical_von_mises_field_id,
    },
};

const TENSOR_COMPONENT_COUNT: usize = 6;

#[derive(Default)]
pub(crate) struct ThermoMechanicalSnapshotFields {
    pub(crate) temperature_snapshots: Vec<AnalysisField>,
    pub(crate) thermal_strain_snapshots: Vec<AnalysisField>,
    pub(crate) thermal_stress_snapshots: Vec<AnalysisField>,
    pub(crate) displacement_snapshots: Vec<AnalysisField>,
    pub(crate) von_mises_snapshots: Vec<AnalysisField>,
    pub(crate) coupling_residual_snapshots: Vec<AnalysisField>,
}

pub(crate) fn recover_thermo_mechanical_snapshots(
    summary: Option<&ThermoMechanicalAssemblySummary>,
    progress_factors: &[f64],
    displacement_snapshots: &[AnalysisField],
    von_mises_snapshots: &[AnalysisField],
    residual_norms: &[f64],
    element_count: usize,
) -> ThermoMechanicalSnapshotFields {
    let Some(summary) = summary.filter(|summary| summary.enabled) else {
        return ThermoMechanicalSnapshotFields::default();
    };

    let mut fields = ThermoMechanicalSnapshotFields {
        temperature_snapshots: Vec::with_capacity(progress_factors.len()),
        thermal_strain_snapshots: Vec::with_capacity(progress_factors.len()),
        thermal_stress_snapshots: Vec::with_capacity(progress_factors.len()),
        displacement_snapshots: Vec::with_capacity(progress_factors.len()),
        von_mises_snapshots: Vec::with_capacity(progress_factors.len()),
        coupling_residual_snapshots: Vec::with_capacity(progress_factors.len()),
    };

    for (index, progress_factor) in progress_factors.iter().copied().enumerate() {
        let temperature =
            summary.reference_temperature_k + summary.applied_temperature_delta_k * progress_factor;
        fields.temperature_snapshots.push(AnalysisField::host_f64(
            fea_thermo_mechanical_temperature_field_id(index),
            vec![1],
            vec![temperature],
        ));

        let strain_value = summary.thermal_strain_scale * progress_factor;
        let mut thermal_strain = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
        let mut thermal_stress = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
        for element in 0..element_count {
            let base = element * TENSOR_COMPONENT_COUNT;
            for component in 0..3 {
                thermal_strain[base + component] = strain_value;
                thermal_stress[base + component] =
                    strain_value * summary.effective_modulus_scale * 1.0e9;
            }
        }
        fields
            .thermal_strain_snapshots
            .push(AnalysisField::host_f64(
                fea_thermo_mechanical_thermal_strain_field_id(index),
                vec![element_count, TENSOR_COMPONENT_COUNT],
                thermal_strain,
            ));
        fields
            .thermal_stress_snapshots
            .push(AnalysisField::host_f64(
                fea_thermo_mechanical_thermal_stress_field_id(index),
                vec![element_count, TENSOR_COMPONENT_COUNT],
                thermal_stress,
            ));

        if let Some(displacement) = displacement_snapshots.get(index) {
            let mut field = displacement.clone();
            field.field_id = fea_thermo_mechanical_displacement_field_id(index);
            fields.displacement_snapshots.push(field);
        }
        if let Some(von_mises) = von_mises_snapshots.get(index) {
            let mut field = von_mises.clone();
            field.field_id = fea_thermo_mechanical_von_mises_field_id(index);
            fields.von_mises_snapshots.push(field);
        }

        let residual = residual_norms
            .get(index)
            .copied()
            .filter(|value| value.is_finite())
            .unwrap_or(0.0)
            * (1.0
                + summary.spatial_gradient_index
                + summary.temporal_profile_variation
                + summary.assignment_heterogeneity_index);
        fields
            .coupling_residual_snapshots
            .push(AnalysisField::host_f64(
                fea_thermo_mechanical_coupling_residual_field_id(index),
                vec![1],
                vec![residual],
            ));
    }

    fields
}
