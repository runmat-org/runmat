use runmat_analysis_core::{AnalysisModel, AnalysisProcedure};

use crate::diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity};

use super::recovery::{FsiCouplingOptions, FsiInterfaceClosure};

pub(super) fn coupling_diagnostics(
    model: &AnalysisModel,
    cfd_executed_sample_count: usize,
    nominal_time_step_s: f64,
    structural_compliance_per_pa: f64,
    options: FsiCouplingOptions,
    closure: &FsiInterfaceClosure,
) -> Vec<FeaDiagnostic> {
    let residual_severity = if closure.max_interface_residual <= options.residual_warn_threshold {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    vec![
        FeaDiagnostic {
            code: "FEA_FSI_INTERFACE_RESIDUAL".to_string(),
            severity: residual_severity,
            message: format!(
                "max_interface_residual={} structural_compliance_per_pa={} residual_warn_threshold={} interface_node_count={} interface_face_count={}",
                closure.max_interface_residual,
                structural_compliance_per_pa,
                options.residual_warn_threshold,
                closure.interface_node_count,
                closure.interface_face_count,
            ),
        },
        closure_diagnostic(closure, options),
        known_answer_diagnostic(closure, options.tolerance),
        FeaDiagnostic {
            code: "FEA_FSI_COUPLING".to_string(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "sample_count={} nominal_time_step_s={} fsi_procedure_count={} cfd_executed_sample_count={} authored_interface_count={} interface_node_count={} interface_face_count={} max_linear_iters={} tolerance={}",
                options.sample_count,
                nominal_time_step_s,
                model
                    .sequence
                    .steps
                    .iter()
                    .filter(|step| step.procedure == AnalysisProcedure::TransientResponse)
                    .count(),
                cfd_executed_sample_count,
                model.interfaces.len(),
                closure.interface_node_count,
                closure.interface_face_count,
                options.max_linear_iters,
                options.tolerance,
            ),
        },
    ]
}

fn closure_diagnostic(closure: &FsiInterfaceClosure, options: FsiCouplingOptions) -> FeaDiagnostic {
    let severity = if closure.force_balance_ratio <= 1.0e-9
        && closure
            .max_displacement_transfer_residual_m
            .canonical_value()
            <= 1.0e-12
        && closure.max_pressure_feedback_residual_ratio <= options.tolerance
        && closure.max_two_way_interface_residual_ratio <= options.tolerance
        && closure.max_structural_traction_update_residual_ratio <= options.tolerance
        && closure.max_pressure_displacement_law_residual_ratio <= options.tolerance
        && closure.max_structural_solve_residual_ratio <= options.tolerance
        && closure.max_interface_work_energy_residual_ratio <= options.tolerance
        && closure.structural_coupling_edge_count > 0
        && closure.interface_connectivity_coverage_ratio >= 1.0
        && closure.max_interface_residual <= options.residual_warn_threshold
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    FeaDiagnostic {
        code: "FEA_FSI_INTERFACE_CLOSURE".to_string(),
        severity,
        message: format!(
            "interface_node_count={} interface_face_count={} max_interface_residual={} force_balance_ratio={} max_displacement_transfer_residual_m={} max_interface_displacement_m={} mean_interface_pressure_pa={} max_traction_magnitude_pa={} max_coupling_iteration_count={} pressure_feedback_residual_ratio={} two_way_interface_residual_ratio={} structural_traction_update_residual_ratio={} pressure_displacement_law_residual_ratio={} structural_solve_residual_ratio={} interface_work_j_per_m2={} structural_strain_energy_j_per_m2={} interface_work_energy_residual_ratio={} structural_coupling_edge_count={} interface_connectivity_coverage_ratio={} mesh_backed_interface_connectivity_ratio={} full_topology_edge_count={} full_topology_element_count={} interface_stiffness_pa_per_m={}",
            closure.interface_node_count,
            closure.interface_face_count,
            closure.max_interface_residual,
            closure.force_balance_ratio,
            closure.max_displacement_transfer_residual_m.canonical_value(),
            closure.max_interface_displacement_m.canonical_value(),
            closure.mean_interface_pressure_pa.canonical_value(),
            closure.max_traction_magnitude_pa.canonical_value(),
            closure.max_coupling_iteration_count,
            closure.max_pressure_feedback_residual_ratio,
            closure.max_two_way_interface_residual_ratio,
            closure.max_structural_traction_update_residual_ratio,
            closure.max_pressure_displacement_law_residual_ratio,
            closure.max_structural_solve_residual_ratio,
            closure.max_interface_work_j_per_m2.canonical_value(),
            closure.max_structural_strain_energy_j_per_m2.canonical_value(),
            closure.max_interface_work_energy_residual_ratio,
            closure.structural_coupling_edge_count,
            closure.interface_connectivity_coverage_ratio,
            closure.mesh_backed_interface_connectivity_ratio,
            closure.full_topology_edge_count,
            closure.full_topology_element_count,
            closure.interface_stiffness_pa_per_m.canonical_value(),
        ),
    }
}

fn known_answer_diagnostic(closure: &FsiInterfaceClosure, tolerance: f64) -> FeaDiagnostic {
    let known_answer_coverage_ratio = if closure.interface_node_count > 0
        && closure.interface_face_count > 0
        && closure
            .mean_interface_pressure_pa
            .canonical_value()
            .is_finite()
        && closure.mean_interface_pressure_pa.canonical_value() > 0.0
        && closure
            .max_traction_magnitude_pa
            .canonical_value()
            .is_finite()
        && closure.max_traction_magnitude_pa.canonical_value() > 0.0
        && closure
            .max_interface_displacement_m
            .canonical_value()
            .is_finite()
        && closure.max_interface_displacement_m.canonical_value() > 0.0
        && closure
            .interface_stiffness_pa_per_m
            .canonical_value()
            .is_finite()
        && closure.interface_stiffness_pa_per_m.canonical_value() > 0.0
        && closure.structural_coupling_edge_count > 0
        && closure.interface_connectivity_coverage_ratio >= 1.0
    {
        1.0
    } else {
        0.0
    };
    let severity = if closure.max_pressure_displacement_law_residual_ratio <= tolerance
        && closure.force_balance_ratio <= 1.0e-9
        && closure
            .max_displacement_transfer_residual_m
            .canonical_value()
            <= 1.0e-12
        && closure.max_pressure_feedback_residual_ratio <= tolerance
        && closure.max_two_way_interface_residual_ratio <= tolerance
        && closure.max_structural_traction_update_residual_ratio <= tolerance
        && closure.max_structural_solve_residual_ratio <= tolerance
        && closure.max_interface_work_energy_residual_ratio <= tolerance
        && closure.interface_connectivity_coverage_ratio >= 1.0
        && known_answer_coverage_ratio >= 1.0
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    FeaDiagnostic {
        code: "FEA_FSI_KNOWN_ANSWER".to_string(),
        severity,
        message: format!(
            "basis=pressure_loaded_wall_partitioned pressure_loaded_wall_displacement_law_residual_ratio={} interface_traction_balance_residual_ratio={} interface_displacement_transfer_residual_m={} partitioned_pressure_feedback_residual_ratio={} two_way_interface_residual_ratio={} structural_traction_update_residual_ratio={} structural_solve_residual_ratio={} interface_work_energy_residual_ratio={} interface_connectivity_coverage_ratio={} mesh_backed_interface_connectivity_ratio={} known_answer_coverage_ratio={}",
            closure.max_pressure_displacement_law_residual_ratio,
            closure.force_balance_ratio,
            closure.max_displacement_transfer_residual_m.canonical_value(),
            closure.max_pressure_feedback_residual_ratio,
            closure.max_two_way_interface_residual_ratio,
            closure.max_structural_traction_update_residual_ratio,
            closure.max_structural_solve_residual_ratio,
            closure.max_interface_work_energy_residual_ratio,
            closure.interface_connectivity_coverage_ratio,
            closure.mesh_backed_interface_connectivity_ratio,
            known_answer_coverage_ratio,
        ),
    }
}
