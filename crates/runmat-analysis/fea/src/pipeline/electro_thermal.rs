use runmat_analysis_core::AnalysisField;

use crate::{
    assembly::ElectroThermalAssemblySummary,
    contracts::{
        fea_electro_thermal_temperature_field_id, fea_electro_thermal_thermal_residual_field_id,
        FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY, FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD,
        FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL, FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT,
    },
};

const VECTOR_COMPONENT_COUNT: usize = 3;

#[derive(Default)]
pub(crate) struct ElectroThermalFields {
    pub(crate) static_fields: Vec<AnalysisField>,
    pub(crate) temperature_snapshots: Vec<AnalysisField>,
    pub(crate) thermal_residual_snapshots: Vec<AnalysisField>,
}

pub(crate) fn recover_electro_thermal_fields(
    summary: Option<&ElectroThermalAssemblySummary>,
    progress_factors: &[f64],
    residual_norms: &[f64],
    dof_count: usize,
) -> ElectroThermalFields {
    let Some(summary) = summary.filter(|summary| summary.enabled) else {
        return ElectroThermalFields::default();
    };

    let node_count = dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1);
    let voltage = summary.applied_voltage_v;
    let conductivity = summary.base_electrical_conductivity_s_per_m
        * summary.conductivity_spread_ratio.sqrt().max(1.0);
    let field_scale = voltage / node_count.max(1) as f64;
    let current_scale = conductivity * field_scale;
    let joule_scale = summary.joule_heating_scale.max(0.0);

    let potential = (0..node_count)
        .map(|index| {
            if node_count == 1 {
                voltage
            } else {
                voltage * (1.0 - index as f64 / (node_count - 1) as f64)
            }
        })
        .collect::<Vec<_>>();
    let mut electric_field = Vec::with_capacity(node_count * VECTOR_COMPONENT_COUNT);
    let mut current_density = Vec::with_capacity(node_count * VECTOR_COMPONENT_COUNT);
    let mut joule_heat = Vec::with_capacity(node_count);
    for index in 0..node_count {
        let heterogeneity = 1.0
            + summary.temporal_profile_variation * 0.05
            + (index as f64 / node_count.max(1) as f64)
                * (summary.conductivity_spread_ratio - 1.0).max(0.0)
                * 0.02;
        electric_field.extend_from_slice(&[field_scale * heterogeneity, 0.0, 0.0]);
        current_density.extend_from_slice(&[current_scale * heterogeneity, 0.0, 0.0]);
        joule_heat.push(joule_scale * heterogeneity);
    }

    let static_fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL,
            vec![node_count],
            potential,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            electric_field,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            current_density,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT,
            vec![node_count],
            joule_heat,
        ),
    ];

    let mut temperature_snapshots = Vec::with_capacity(progress_factors.len());
    let mut thermal_residual_snapshots = Vec::with_capacity(progress_factors.len());
    for (index, progress_factor) in progress_factors.iter().copied().enumerate() {
        let current_factor = progress_factor.clamp(0.0, 1.0);
        let temperature_rise =
            joule_scale * current_factor * (1.0 + summary.temporal_profile_variation);
        temperature_snapshots.push(AnalysisField::host_f64(
            fea_electro_thermal_temperature_field_id(index),
            vec![node_count],
            vec![summary.reference_temperature_k + temperature_rise; node_count],
        ));

        let residual = residual_norms
            .get(index)
            .copied()
            .filter(|value| value.is_finite())
            .unwrap_or(0.0)
            * (1.0 + summary.conductivity_spread_ratio.ln().max(0.0));
        thermal_residual_snapshots.push(AnalysisField::host_f64(
            fea_electro_thermal_thermal_residual_field_id(index),
            vec![1],
            vec![residual],
        ));
    }

    ElectroThermalFields {
        static_fields,
        temperature_snapshots,
        thermal_residual_snapshots,
    }
}
