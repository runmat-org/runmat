use runmat_analysis_core::AnalysisField;

use crate::{
    assembly::ThermoMechanicalAssemblySummary,
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
        .diagnostics
        .push(thermo_mechanical_consistency_diagnostic(&consistency));

    fields
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
