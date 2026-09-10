use runmat_analysis_core::CfdDomain;

use crate::diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity};

use super::{domain::CfdDomainTopology, solve::CfdVelocityPressureSolution};

#[derive(Clone, Copy, Debug)]
pub struct CfdCouplingEvidence {
    pub inlet_velocity_m_per_s: f64,
    pub executed_sample_count: usize,
    pub nominal_time_step_s: f64,
    pub transient: bool,
    pub pressure_drop_pa: f64,
    pub mass_balance_residual: f64,
    pub max_momentum_residual: f64,
    pub max_continuity_residual: f64,
    pub residual_warn_threshold: f64,
}

pub fn coupling_diagnostics(
    domain: &CfdDomain,
    topology: &CfdDomainTopology,
    evidence: CfdCouplingEvidence,
) -> Vec<FeaDiagnostic> {
    let residual_severity = if evidence.max_momentum_residual <= evidence.residual_warn_threshold
        && evidence.max_continuity_residual <= evidence.residual_warn_threshold
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    vec![
        FeaDiagnostic {
            code: "FEA_CFD_FLOW".to_string(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "density={} viscosity={} inlet_velocity={} turbulence_intensity={} reynolds_number={} solve_family={} executed_sample_count={} topology_basis={} control_volume_count={} control_volume_face_count={} control_volume_internal_face_count={} control_volume_boundary_face_count={} control_volume_connectivity_coverage_ratio={} domain_length_m={} hydraulic_diameter_m={}",
                domain.density.kilograms_per_cubic_meter,
                domain.dynamic_viscosity.pascal_seconds,
                evidence.inlet_velocity_m_per_s,
                domain.turbulence_intensity.ratio,
                super::domain::reynolds_number_for_velocity(
                    domain,
                    evidence.inlet_velocity_m_per_s,
                ),
                if evidence.transient { "transient" } else { "steady_state" },
                evidence.executed_sample_count,
                topology.basis.as_str(),
                topology.control_volume_count,
                topology.control_volume_face_count,
                topology.control_volume_internal_face_count,
                topology.control_volume_boundary_face_count,
                topology.control_volume_connectivity_coverage_ratio,
                topology.domain_length_m,
                topology.hydraulic_diameter_m,
            ),
        },
        FeaDiagnostic {
            code: "FEA_CFD_RESIDUAL".to_string(),
            severity: residual_severity,
            message: format!(
                "max_momentum_residual={} max_continuity_residual={} residual_warn_threshold={} cfd_node_count={} executed_sample_count={}",
                evidence.max_momentum_residual,
                evidence.max_continuity_residual,
                evidence.residual_warn_threshold,
                topology.node_count,
                evidence.executed_sample_count,
            ),
        },
        assembly_diagnostic(
            topology,
            domain,
            evidence.inlet_velocity_m_per_s,
            evidence.nominal_time_step_s,
            evidence.pressure_drop_pa,
            evidence.mass_balance_residual,
            evidence.residual_warn_threshold,
        ),
    ]
}

#[derive(Clone, Debug)]
pub(super) struct CfdKnownAnswerMetrics {
    pub pressure_drop_balance_ratio: f64,
    pub mass_flux_uniformity_ratio: f64,
    pub pressure_monotonic_cell_fraction: f64,
    pub known_answer_coverage_ratio: f64,
}

pub(super) fn known_answer_metrics(
    solution: &CfdVelocityPressureSolution,
) -> CfdKnownAnswerMetrics {
    let node_count = solution.pressure.len();
    let pressure_drop_observed = match (solution.pressure.first(), solution.pressure.last()) {
        (Some(first), Some(last)) => first - last,
        _ => 0.0,
    };
    let pressure_drop_balance_ratio = if solution.pressure_drop_pa.abs() > 1.0e-12 {
        pressure_drop_observed / solution.pressure_drop_pa
    } else if pressure_drop_observed.abs() <= 1.0e-12 {
        1.0
    } else {
        0.0
    };

    let axial_values = (0..node_count)
        .map(|node| solution.velocity.get(node * 3).copied().unwrap_or(0.0))
        .collect::<Vec<_>>();
    let mean_axial = if axial_values.is_empty() {
        0.0
    } else {
        axial_values.iter().sum::<f64>() / axial_values.len() as f64
    };
    let max_axial_deviation = axial_values
        .iter()
        .map(|value| (value - mean_axial).abs())
        .fold(0.0_f64, f64::max);
    let mass_flux_uniformity_ratio = max_axial_deviation / mean_axial.abs().max(1.0e-12);

    let pressure_edge_count = node_count.saturating_sub(1);
    let pressure_monotonic_cell_fraction = if pressure_edge_count == 0 {
        1.0
    } else {
        let tolerance = solution.pressure_drop_pa.abs().max(1.0) * 1.0e-12;
        let monotonic_edges = solution
            .pressure
            .windows(2)
            .filter(|pair| pair[0] + tolerance >= pair[1])
            .count();
        monotonic_edges as f64 / pressure_edge_count as f64
    };

    let known_answer_coverage_ratio = if node_count >= 2
        && solution.control_volume_count == node_count - 1
        && solution.velocity.len() == node_count * 3
        && solution.pressure_drop_pa.is_finite()
        && solution.pressure.iter().all(|value| value.is_finite())
        && solution.velocity.iter().all(|value| value.is_finite())
    {
        1.0
    } else {
        0.0
    };

    CfdKnownAnswerMetrics {
        pressure_drop_balance_ratio,
        mass_flux_uniformity_ratio,
        pressure_monotonic_cell_fraction,
        known_answer_coverage_ratio,
    }
}

pub(super) fn known_answer_diagnostic(
    metrics: &CfdKnownAnswerMetrics,
    topology: &CfdDomainTopology,
) -> FeaDiagnostic {
    let severity = if (metrics.pressure_drop_balance_ratio - 1.0).abs() <= 1.0e-10
        && metrics.mass_flux_uniformity_ratio <= 1.0e-8
        && metrics.pressure_monotonic_cell_fraction >= 1.0
        && metrics.known_answer_coverage_ratio >= 1.0
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    FeaDiagnostic {
        code: "FEA_CFD_KNOWN_ANSWER".to_string(),
        severity,
        message: format!(
            "basis=finite_volume_channel topology_basis={} pressure_drop_balance_ratio={} mass_flux_uniformity_ratio={} pressure_monotonic_cell_fraction={} known_answer_coverage_ratio={}",
            topology.basis.as_str(),
            metrics.pressure_drop_balance_ratio,
            metrics.mass_flux_uniformity_ratio,
            metrics.pressure_monotonic_cell_fraction,
            metrics.known_answer_coverage_ratio,
        ),
    }
}

pub fn assembly_diagnostic(
    topology: &CfdDomainTopology,
    domain: &CfdDomain,
    inlet_velocity_m_per_s: f64,
    time_step_s: f64,
    pressure_drop_pa: f64,
    mass_balance_residual: f64,
    residual_warn_threshold: f64,
) -> FeaDiagnostic {
    let face_area_m2 = topology.face_area_m2();
    let control_volume_volume_m3 = topology.control_volume_volume_m3();
    let nominal_mass_flow_rate_kg_per_s =
        domain.density.kilograms_per_cubic_meter * inlet_velocity_m_per_s * face_area_m2;
    let courant_number =
        inlet_velocity_m_per_s.abs() * time_step_s.max(0.0) / topology.dx_m.max(1.0e-12);
    FeaDiagnostic {
        code: "FEA_CFD_ASSEMBLY".to_string(),
        severity: if mass_balance_residual <= residual_warn_threshold {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "basis=finite_volume_velocity_pressure topology_basis={} topology_geometry_source={} control_volume_count={} control_volume_face_count={} control_volume_internal_face_count={} control_volume_boundary_face_count={} control_volume_connectivity_coverage_ratio={} hydraulic_diameter_m={} domain_length_m={} dx_m={} face_area_m2={} control_volume_volume_m3={} nominal_mass_flow_rate_kg_per_s={} courant_number={} active_dimension_count={} element_geometry_node_count={} element_geometry_edge_count={} element_geometry_coverage_ratio={} pressure_drop_pa={} mass_balance_residual={}",
            topology.basis.as_str(),
            topology.geometry_source.as_str(),
            topology.control_volume_count,
            topology.control_volume_face_count,
            topology.control_volume_internal_face_count,
            topology.control_volume_boundary_face_count,
            topology.control_volume_connectivity_coverage_ratio,
            topology.hydraulic_diameter_m,
            topology.domain_length_m,
            topology.dx_m,
            face_area_m2,
            control_volume_volume_m3,
            nominal_mass_flow_rate_kg_per_s,
            courant_number,
            topology.active_dimension_count,
            topology.element_geometry_node_count,
            topology.element_geometry_edge_count,
            topology.element_geometry_coverage_ratio,
            pressure_drop_pa,
            mass_balance_residual,
        ),
    }
}
