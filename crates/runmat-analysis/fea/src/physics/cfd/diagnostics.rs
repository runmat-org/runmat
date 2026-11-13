use runmat_analysis_core::CfdDomain;

use crate::diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity};

use super::domain::CfdDomainTopology;

#[derive(Clone, Copy, Debug)]
pub struct CfdCouplingEvidence {
    pub inlet_velocity_m_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, f64>,
    pub executed_sample_count: usize,
    pub nominal_time_step_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Time, f64>,
    pub transient: bool,
    pub pressure_drop_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, f64>,
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
                domain.density.canonical_value(),
                domain.dynamic_viscosity.canonical_value(),
                evidence.inlet_velocity_m_per_s.canonical_value(),
                domain.turbulence_intensity.canonical_value(),
                super::domain::reynolds_number_for_velocity(
                    domain,
                    evidence.inlet_velocity_m_per_s.canonical_value(),
                ),
                if evidence.transient { "transient" } else { "steady_state" },
                evidence.executed_sample_count,
                topology.basis.as_str(),
                topology.control_volume_count,
                topology.control_volume_face_count,
                topology.control_volume_internal_face_count,
                topology.control_volume_boundary_face_count,
                topology.control_volume_connectivity_coverage_ratio,
                topology.domain_length_m.canonical_value(),
                topology.hydraulic_diameter_m.canonical_value(),
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
            evidence.inlet_velocity_m_per_s.canonical_value(),
            evidence.nominal_time_step_s.canonical_value(),
            evidence.pressure_drop_pa.canonical_value(),
            evidence.mass_balance_residual,
            evidence.residual_warn_threshold,
        ),
    ]
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
        domain.density.canonical_value() * inlet_velocity_m_per_s * face_area_m2;
    let courant_number = inlet_velocity_m_per_s.abs() * time_step_s.max(0.0)
        / topology.dx_m.canonical_value().max(1.0e-12);
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
            topology.hydraulic_diameter_m.canonical_value(),
            topology.domain_length_m.canonical_value(),
            topology.dx_m.canonical_value(),
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
