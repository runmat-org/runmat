use runmat_analysis_core::AnalysisModel;

pub(crate) use super::material_assignment::diagnostics as material_assignment_diagnostics;

use crate::{
    assembly,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
};

#[derive(Debug, Clone)]
pub(crate) struct CommonRunDiagnosticInputs<'a> {
    pub(crate) model: &'a AnalysisModel,
    pub(crate) summary: &'a assembly::AssemblySummary,
}

pub(crate) fn extend_common_run_diagnostics(
    diagnostics: &mut Vec<FeaDiagnostic>,
    inputs: CommonRunDiagnosticInputs<'_>,
) {
    diagnostics.extend(material_assignment_diagnostics(
        &inputs.model.material_assignments,
    ));
    if inputs.summary.structural_rotational_dof_count > 0
        || inputs.summary.structural_moment_load_count > 0
        || inputs.summary.structural_beam_element_count > 0
        || inputs.summary.structural_shell_element_count > 0
        || inputs.summary.structural_connector_stiffness_count > 0
        || inputs.summary.structural_connector_damping_count > 0
    {
        diagnostics.push(structural_rotational_dof_diagnostic(inputs.summary));
    }
    if let Some(thermo_mechanical) = inputs.summary.thermo_mechanical.as_ref() {
        diagnostics.push(thermo_mechanical_diagnostic(thermo_mechanical));
    }
    if let Some(electro_thermal) = inputs.summary.electro_thermal.as_ref() {
        diagnostics.push(electro_thermal_diagnostic(electro_thermal));
    }
}

pub(crate) fn structural_rotational_dof_diagnostic(
    summary: &assembly::AssemblySummary,
) -> FeaDiagnostic {
    let direct_moment_coverage_ratio = if summary.structural_moment_load_count == 0 {
        1.0
    } else {
        summary.structural_direct_rotational_moment_load_count as f64
            / summary.structural_moment_load_count as f64
    };
    let beam_local_frame_coverage_ratio = 1.0;
    let beam_stiffness_matrix_symmetry_residual = if summary.structural_beam_element_count == 0 {
        0.0
    } else {
        beam_operator_symmetry_residual(summary)
    };

    FeaDiagnostic {
        code: "FEA_STRUCTURAL_ROTATIONAL_DOF".to_string(),
        severity: if direct_moment_coverage_ratio >= 1.0
            && beam_local_frame_coverage_ratio >= 1.0
            && beam_stiffness_matrix_symmetry_residual <= 1.0e-10
        {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "structural_node_count={} structural_translational_dof_count={} structural_rotational_dof_count={} structural_rotation_node_count={} structural_moment_load_count={} structural_direct_rotational_moment_load_count={} structural_direct_rotational_moment_coverage_ratio={} structural_moment_requested_norm_n_m={} structural_rotational_constraint_count={} structural_beam_element_count={} structural_shell_element_count={} structural_connector_stiffness_count={} structural_connector_damping_count={} structural_solid_element_count={} structural_solid_total_mass_kg={} structural_beam_local_frame_coverage_ratio={} structural_beam_stiffness_matrix_symmetry_residual={}",
            summary.structural_node_count,
            summary.structural_translational_dof_count,
            summary.structural_rotational_dof_count,
            summary.structural_rotation_node_count,
            summary.structural_moment_load_count,
            summary.structural_direct_rotational_moment_load_count,
            direct_moment_coverage_ratio,
            summary.structural_requested_moment_norm_n_m,
            summary.structural_rotational_constraint_count,
            summary.structural_beam_element_count,
            summary.structural_shell_element_count,
            summary.structural_connector_stiffness_count,
            summary.structural_connector_damping_count,
            summary.structural_solid_element_count,
            summary.structural_solid_total_mass_kg,
            beam_local_frame_coverage_ratio,
            beam_stiffness_matrix_symmetry_residual
        ),
    }
}

fn beam_operator_symmetry_residual(summary: &assembly::AssemblySummary) -> f64 {
    let stiffness = summary.operator.operators().stiffness();
    let n = summary.operator.dof_count();
    if n == 0 {
        return 1.0;
    }
    let mut max_residual = 0.0_f64;
    let mut max_entry = 0.0_f64;
    for row in 0..n {
        for col in 0..n {
            let a = stiffness.get(row, col);
            let b = stiffness.get(col, row);
            max_residual = max_residual.max((a - b).abs());
            max_entry = max_entry.max(a.abs()).max(b.abs());
        }
    }
    max_residual / max_entry.max(1.0)
}

pub(crate) fn thermo_mechanical_diagnostic(
    summary: &assembly::ThermoMechanicalAssemblySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_TM_COUPLING".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "enabled={} reference_temperature_k={} applied_temperature_delta_k={} thermal_expansion_coefficient={} thermal_strain_scale={} thermal_load_scale={} constitutive_temperature_factor={} constitutive_poisson_coupling={} effective_modulus_scale={} constitutive_material_spread_ratio={} assignment_heterogeneity_index={} spatial_gradient_index={} spatial_coverage_ratio={} region_delta_count={} coupling_fingerprint={}",
            summary.enabled,
            summary.reference_temperature_k.canonical_value(),
            summary.applied_temperature_delta_k.canonical_value(),
            summary.thermal_expansion_coefficient.canonical_value(),
            summary.thermal_strain_scale,
            summary.thermal_load_scale,
            summary.constitutive_temperature_factor,
            summary.constitutive_poisson_coupling,
            summary.effective_modulus_scale,
            summary.constitutive_material_spread_ratio,
            summary.assignment_heterogeneity_index,
            summary.spatial_gradient_index,
            summary.spatial_coverage_ratio,
            summary.region_delta_count,
            summary.coupling_fingerprint,
        ),
    }
}

pub(crate) fn electro_thermal_diagnostic(
    summary: &assembly::ElectroThermalAssemblySummary,
) -> FeaDiagnostic {
    let electrical_power_in_w = summary.applied_voltage_v.powi(2)
        * summary.base_electrical_conductivity_s_per_m.max(1.0e-9)
        * summary
            .resistive_heating_coefficient
            .canonical_value()
            .max(0.0)
        / 1.0e6;
    let integrated_joule_heat_w = summary.joule_heating_scale;
    let power_balance_ratio = if electrical_power_in_w > 1.0e-12 {
        integrated_joule_heat_w / electrical_power_in_w
    } else {
        1.0
    };
    let conservation_residual = (1.0 - power_balance_ratio).abs();
    FeaDiagnostic {
        code: "FEA_ET_COUPLING".to_string(),
        severity: if conservation_residual <= 1.0e-6 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "enabled={} reference_temperature_k={} applied_voltage_v={} base_electrical_conductivity_s_per_m={} resistive_heating_coefficient={} joule_heating_scale={} conductivity_spread_ratio={} region_scale_count={} coupling_fingerprint={} electrical_power_in_w={} integrated_joule_heat_w={} power_balance_ratio={} conservation_residual={}",
            summary.enabled,
            summary.reference_temperature_k.canonical_value(),
            summary.applied_voltage_v,
            summary.base_electrical_conductivity_s_per_m,
            summary.resistive_heating_coefficient.canonical_value(),
            summary.joule_heating_scale,
            summary.conductivity_spread_ratio,
            summary.region_scale_count,
            summary.coupling_fingerprint,
            electrical_power_in_w,
            integrated_joule_heat_w,
            power_balance_ratio,
            conservation_residual,
        ),
    }
}
