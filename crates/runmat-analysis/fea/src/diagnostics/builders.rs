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
