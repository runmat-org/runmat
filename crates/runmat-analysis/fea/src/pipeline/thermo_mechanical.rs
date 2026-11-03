use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use runmat_analysis_core::AnalysisField;
mod solved;
pub(crate) use solved::recover_solved_static_fields;

use crate::{
    assembly::{AssemblySummary, ThermoMechanicalAssemblySummary},
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
const CONSISTENCY_TOLERANCE: f64 = 1.0e-12;

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
        diagnostics: Vec::new(),
    };
    let mut consistency = ThermoMechanicalConsistencyAccumulator::new(
        progress_factors.len() * element_count * NORMAL_COMPONENT_COUNT,
    );
    let node_count = infer_thermo_mechanical_node_count(displacement_snapshots, element_count);
    let mut temperature_field = ThermoMechanicalTemperatureFieldAccumulator::new(
        node_count,
        element_count,
        progress_factors.len(),
    );

    for (index, progress_factor) in progress_factors.iter().copied().enumerate() {
        let temperature =
            thermo_mechanical_temperature_profile(summary, progress_factor, node_count);
        temperature_field.observe_temperature(&temperature);
        fields.temperature_snapshots.push(AnalysisField::host_f64(
            fea_thermo_mechanical_temperature_field_id(index), AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node).step(index),
            vec![node_count],
            temperature.clone(),
        ));

        let mut thermal_strain = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
        let mut thermal_stress = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
        for element in 0..element_count {
            let base = element * TENSOR_COMPONENT_COUNT;
            let temperature_delta =
                thermo_mechanical_element_temperature_delta(&temperature, summary, element);
            let strain_value =
                summary.thermal_expansion_coefficient.canonical_value() * temperature_delta;
            for component in 0..3 {
                thermal_strain[base + component] = strain_value;
                thermal_stress[base + component] =
                    strain_value * summary.effective_modulus_scale * 1.0e9;
                temperature_field.observe_strain_temperature_pair(
                    strain_value,
                    summary.thermal_expansion_coefficient.canonical_value(),
                    temperature_delta,
                );
                consistency.observe(
                    thermal_strain[base + component],
                    thermal_stress[base + component],
                    summary.effective_modulus_scale,
                );
            }
        }
        fields
            .thermal_strain_snapshots
            .push(AnalysisField::host_f64(
                fea_thermo_mechanical_thermal_strain_field_id(index), AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Strain, FieldLocation::Element).step(index),
                vec![element_count, TENSOR_COMPONENT_COUNT],
                thermal_strain,
            ));
        fields
            .thermal_stress_snapshots
            .push(AnalysisField::host_f64(
                fea_thermo_mechanical_thermal_stress_field_id(index), AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element).step(index),
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
            * (1.0 + summary.spatial_gradient_index + summary.assignment_heterogeneity_index);
        fields
            .coupling_residual_snapshots
            .push(AnalysisField::host_f64(
                fea_thermo_mechanical_coupling_residual_field_id(index), AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global).step(index),
                vec![1],
                vec![residual],
            ));
    }
    fields
        .diagnostics
        .push(thermo_mechanical_consistency_diagnostic(&consistency));
    fields
        .diagnostics
        .push(thermo_mechanical_temperature_field_diagnostic(
            &temperature_field,
        ));

    fields
}

fn infer_thermo_mechanical_node_count(
    displacement_snapshots: &[AnalysisField],
    element_count: usize,
) -> usize {
    displacement_snapshots
        .iter()
        .find_map(|field| match field.shape.as_slice() {
            [node_count, NORMAL_COMPONENT_COUNT] if *node_count > 0 => Some(*node_count),
            _ => None,
        })
        .unwrap_or_else(|| element_count.max(1))
}

fn thermo_mechanical_temperature_profile(
    summary: &ThermoMechanicalAssemblySummary,
    progress_factor: f64,
    node_count: usize,
) -> Vec<f64> {
    let node_count = node_count.max(1);
    let progress = progress_factor.clamp(0.0, 1.0);
    let mean_delta = summary.applied_temperature_delta_k.canonical_value() * progress;
    let gradient_strength = summary
        .spatial_gradient_index
        .max(if summary.region_delta_count > 0 {
            0.05
        } else {
            0.0
        })
        .clamp(0.0, 0.5);
    (0..node_count)
        .map(|node| {
            let centered_position = if node_count <= 1 {
                0.0
            } else {
                node as f64 / (node_count - 1) as f64 - 0.5
            };
            summary.reference_temperature_k.canonical_value()
                + mean_delta * (1.0 + 2.0 * gradient_strength * centered_position)
        })
        .collect()
}

fn thermo_mechanical_element_temperature_delta(
    temperature: &[f64],
    summary: &ThermoMechanicalAssemblySummary,
    element_index: usize,
) -> f64 {
    if temperature.is_empty() {
        return 0.0;
    }
    let node_index = if temperature.len() == 1 {
        0
    } else if element_index >= temperature.len() {
        temperature.len() - 1
    } else {
        element_index
    };
    temperature[node_index] - summary.reference_temperature_k.canonical_value()
}

struct ThermoMechanicalTemperatureFieldAccumulator {
    node_count: usize,
    element_count: usize,
    snapshot_count: usize,
    max_temperature_span_k: f64,
    max_strain_temperature_residual_ratio: f64,
    checked_strain_temperature_pairs: usize,
}

impl ThermoMechanicalTemperatureFieldAccumulator {
    fn new(node_count: usize, element_count: usize, snapshot_count: usize) -> Self {
        Self {
            node_count,
            element_count,
            snapshot_count,
            max_temperature_span_k: 0.0,
            max_strain_temperature_residual_ratio: 0.0,
            checked_strain_temperature_pairs: 0,
        }
    }

    fn observe_temperature(&mut self, temperature: &[f64]) {
        let (min, max) = temperature
            .iter()
            .copied()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
                (min.min(value), max.max(value))
            });
        if min.is_finite() && max.is_finite() {
            self.max_temperature_span_k = self.max_temperature_span_k.max((max - min).abs());
        }
    }

    fn observe_strain_temperature_pair(
        &mut self,
        observed_strain: f64,
        thermal_expansion_coefficient: f64,
        temperature_delta: f64,
    ) {
        let expected_strain = thermal_expansion_coefficient * temperature_delta;
        if expected_strain.abs() <= CONSISTENCY_TOLERANCE
            && observed_strain.abs() <= CONSISTENCY_TOLERANCE
        {
            return;
        }
        self.checked_strain_temperature_pairs += 1;
        let residual = (observed_strain - expected_strain).abs()
            / observed_strain
                .abs()
                .max(expected_strain.abs())
                .max(CONSISTENCY_TOLERANCE);
        self.max_strain_temperature_residual_ratio =
            self.max_strain_temperature_residual_ratio.max(residual);
    }
}

struct ThermoMechanicalConsistencyAccumulator {
    expected_component_count: usize,
    checked_component_count: usize,
    max_constitutive_residual_ratio: f64,
    strain_energy_density_sum: f64,
    strain_energy_density_count: usize,
}

impl ThermoMechanicalConsistencyAccumulator {
    fn new(expected_component_count: usize) -> Self {
        Self {
            expected_component_count,
            checked_component_count: 0,
            max_constitutive_residual_ratio: 0.0,
            strain_energy_density_sum: 0.0,
            strain_energy_density_count: 0,
        }
    }

    fn observe(&mut self, strain: f64, stress: f64, effective_modulus_scale: f64) {
        let expected_stress = strain * effective_modulus_scale * 1.0e9;
        if expected_stress.abs() <= CONSISTENCY_TOLERANCE && stress.abs() <= CONSISTENCY_TOLERANCE {
            return;
        }
        self.checked_component_count += 1;
        let residual = (stress - expected_stress).abs()
            / stress
                .abs()
                .max(expected_stress.abs())
                .max(CONSISTENCY_TOLERANCE);
        self.max_constitutive_residual_ratio = self.max_constitutive_residual_ratio.max(residual);
        let strain_energy_density = 0.5 * stress * strain;
        if strain_energy_density.is_finite() {
            self.strain_energy_density_sum += strain_energy_density;
            self.strain_energy_density_count += 1;
        }
    }

    fn coverage_ratio(&self) -> f64 {
        if self.expected_component_count == 0 {
            0.0
        } else {
            self.checked_component_count as f64 / self.expected_component_count as f64
        }
    }

    fn mean_strain_energy_density(&self) -> f64 {
        if self.strain_energy_density_count == 0 {
            0.0
        } else {
            self.strain_energy_density_sum / self.strain_energy_density_count as f64
        }
    }
}

fn thermo_mechanical_consistency_diagnostic(
    consistency: &ThermoMechanicalConsistencyAccumulator,
) -> FeaDiagnostic {
    let coverage_ratio = consistency.coverage_ratio();
    let mean_strain_energy_density = consistency.mean_strain_energy_density();
    let severity = if consistency.checked_component_count > 0
        && coverage_ratio >= 0.5
        && consistency.max_constitutive_residual_ratio <= 1.0e-10
        && mean_strain_energy_density >= 0.0
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    FeaDiagnostic {
        code: "FEA_TM_CONSISTENCY".to_string(),
        severity,
        message: format!(
            "constitutive_relation=thermal_stress_equals_effective_modulus_times_strain checked_component_count={} expected_component_count={} constitutive_residual_ratio={} thermal_strain_energy_density_mean={} consistency_coverage_ratio={}",
            consistency.checked_component_count,
            consistency.expected_component_count,
            consistency.max_constitutive_residual_ratio,
            mean_strain_energy_density,
            coverage_ratio,
        ),
    }
}

fn thermo_mechanical_temperature_field_diagnostic(
    field: &ThermoMechanicalTemperatureFieldAccumulator,
) -> FeaDiagnostic {
    let expected_pair_count = field.snapshot_count * field.element_count * NORMAL_COMPONENT_COUNT;
    let strain_temperature_coverage_ratio = if expected_pair_count == 0 {
        0.0
    } else {
        field.checked_strain_temperature_pairs as f64 / expected_pair_count as f64
    };
    let severity = if field.node_count > 0
        && field.element_count > 0
        && field.checked_strain_temperature_pairs > 0
        && strain_temperature_coverage_ratio >= 0.5
        && field.max_strain_temperature_residual_ratio <= 1.0e-10
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    FeaDiagnostic {
        code: "FEA_TM_TEMPERATURE_FIELD".to_string(),
        severity,
        message: format!(
            "temperature_field_basis=nodal_coupling node_count={} element_count={} snapshot_count={} max_temperature_span_k={} strain_temperature_residual_ratio={} strain_temperature_coverage_ratio={}",
            field.node_count,
            field.element_count,
            field.snapshot_count,
            field.max_temperature_span_k,
            field.max_strain_temperature_residual_ratio,
            strain_temperature_coverage_ratio,
        ),
    }
}
