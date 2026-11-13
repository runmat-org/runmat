use runmat_analysis_core::CfdDomain;

use crate::diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity};

use super::recovery::{ChtCouplingOptions, ChtInterfaceClosure};

#[derive(Debug, Clone, Copy)]
struct ChtKnownAnswerMetrics {
    heated_channel_energy_residual_ratio: f64,
    conjugate_slab_flux_law_residual_ratio: f64,
    interface_temperature_continuity_ratio: f64,
    advection_shift_coverage_ratio: f64,
    coupled_interface_residual_ratio: f64,
    heat_flux_realization_residual_ratio: f64,
    interface_connectivity_coverage_ratio: f64,
    mesh_backed_interface_connectivity_ratio: f64,
    thermal_network_residual_ratio: f64,
    known_answer_coverage_ratio: f64,
}

pub(super) struct ChtDiagnosticInputs<'a> {
    pub(super) authored_interface_count: usize,
    pub(super) reference_temperature_k: runmat_types::physical::Temperature,
    pub(super) applied_temperature_delta_k: runmat_types::physical::TemperatureDifference,
    pub(super) inlet_velocity_m_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, f64>,
    pub(super) nominal_time_step_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Time, f64>,
    pub(super) options: ChtCouplingOptions,
    pub(super) domain: &'a CfdDomain,
    pub(super) closure: &'a ChtInterfaceClosure,
}

pub(super) fn coupling_diagnostics(input: ChtDiagnosticInputs<'_>) -> Vec<FeaDiagnostic> {
    let known_answer = known_answer_metrics(
        super::super::cfd::reynolds_number_for_velocity(
            input.domain,
            input.inlet_velocity_m_per_s.canonical_value(),
        ),
        input.closure,
    );
    vec![
        FeaDiagnostic {
            code: "FEA_CHT_COUPLING".to_string(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "reference_temperature_k={} applied_temperature_delta_k={} sample_count={} nominal_time_step_s={} authored_interface_count={}",
                input.reference_temperature_k.canonical_value(),
                input.applied_temperature_delta_k.canonical_value(),
                input.options.sample_count,
                input.nominal_time_step_s.canonical_value(),
                input.authored_interface_count,
            ),
        },
        closure_diagnostic(input.closure, input.options.residual_warn_threshold),
        known_answer_diagnostic(&known_answer, input.options.residual_warn_threshold),
    ]
}

fn closure_diagnostic(closure: &ChtInterfaceClosure, threshold: f64) -> FeaDiagnostic {
    let severity = if closure.heat_flux_balance_ratio <= 1.0e-9
        && closure.max_energy_residual <= threshold
        && closure.max_temperature_jump_k.canonical_value() <= 0.1
        && closure.max_thermal_transport_residual_ratio <= threshold
        && closure.max_flux_temperature_law_residual_ratio <= threshold
        && closure.max_heat_flux_realization_residual_ratio <= threshold
        && closure.max_coupled_interface_residual_ratio <= threshold
        && closure.interface_connectivity_coverage_ratio >= 1.0
        && closure.max_thermal_network_residual_ratio <= threshold
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    FeaDiagnostic {
        code: "FEA_CHT_INTERFACE_CLOSURE".to_string(),
        severity,
        message: format!(
            "interface_face_count={} max_temperature_jump_k={} max_energy_residual={} heat_flux_balance_ratio={} mean_interface_heat_flux_w_per_m2={} thermal_transport_residual_ratio={} interface_temperature_continuity_ratio={} max_advection_temperature_shift_k={} interface_conductance_w_per_m2k={} flux_temperature_law_residual_ratio={} heat_flux_realization_residual_ratio={} coupled_interface_iteration_count={} coupled_interface_residual_ratio={} thermal_network_node_count={} thermal_network_edge_count={} interface_connectivity_coverage_ratio={} mesh_backed_interface_connectivity_ratio={} full_topology_edge_count={} full_topology_element_count={} thermal_network_residual_ratio={}",
            closure.interface_face_count,
            closure.max_temperature_jump_k.canonical_value(),
            closure.max_energy_residual,
            closure.heat_flux_balance_ratio,
            closure.mean_interface_heat_flux_w_per_m2.canonical_value(),
            closure.max_thermal_transport_residual_ratio,
            closure.interface_temperature_continuity_ratio,
            closure.max_advection_temperature_shift_k.canonical_value(),
            closure.interface_conductance_w_per_m2k.canonical_value(),
            closure.max_flux_temperature_law_residual_ratio,
            closure.max_heat_flux_realization_residual_ratio,
            closure.max_coupled_interface_iteration_count,
            closure.max_coupled_interface_residual_ratio,
            closure.thermal_network_node_count,
            closure.thermal_network_edge_count,
            closure.interface_connectivity_coverage_ratio,
            closure.mesh_backed_interface_connectivity_ratio,
            closure.full_topology_edge_count,
            closure.full_topology_element_count,
            closure.max_thermal_network_residual_ratio,
        ),
    }
}

fn known_answer_metrics(reynolds: f64, closure: &ChtInterfaceClosure) -> ChtKnownAnswerMetrics {
    let advection_shift_coverage_ratio = if reynolds.is_finite()
        && reynolds > 0.0
        && closure
            .max_advection_temperature_shift_k
            .canonical_value()
            .is_finite()
        && closure.max_advection_temperature_shift_k.canonical_value() >= 0.0
    {
        1.0
    } else {
        0.0
    };
    let known_answer_coverage_ratio = if closure.interface_face_count > 0
        && closure
            .interface_conductance_w_per_m2k
            .canonical_value()
            .is_finite()
        && closure.interface_conductance_w_per_m2k.canonical_value() > 0.0
        && closure.max_energy_residual.is_finite()
        && closure.max_flux_temperature_law_residual_ratio.is_finite()
        && closure.max_heat_flux_realization_residual_ratio.is_finite()
        && closure.interface_temperature_continuity_ratio.is_finite()
        && closure.max_coupled_interface_residual_ratio.is_finite()
        && closure.thermal_network_node_count > 0
        && closure.interface_connectivity_coverage_ratio >= 1.0
        && closure.mesh_backed_interface_connectivity_ratio.is_finite()
        && closure.max_thermal_network_residual_ratio.is_finite()
        && advection_shift_coverage_ratio >= 1.0
    {
        1.0
    } else {
        0.0
    };
    ChtKnownAnswerMetrics {
        heated_channel_energy_residual_ratio: closure.max_energy_residual,
        conjugate_slab_flux_law_residual_ratio: closure.max_flux_temperature_law_residual_ratio,
        interface_temperature_continuity_ratio: closure.interface_temperature_continuity_ratio,
        advection_shift_coverage_ratio,
        coupled_interface_residual_ratio: closure.max_coupled_interface_residual_ratio,
        heat_flux_realization_residual_ratio: closure.max_heat_flux_realization_residual_ratio,
        interface_connectivity_coverage_ratio: closure.interface_connectivity_coverage_ratio,
        mesh_backed_interface_connectivity_ratio: closure.mesh_backed_interface_connectivity_ratio,
        thermal_network_residual_ratio: closure.max_thermal_network_residual_ratio,
        known_answer_coverage_ratio,
    }
}

fn known_answer_diagnostic(
    metrics: &ChtKnownAnswerMetrics,
    residual_threshold: f64,
) -> FeaDiagnostic {
    let severity = if metrics.heated_channel_energy_residual_ratio <= residual_threshold
        && metrics.conjugate_slab_flux_law_residual_ratio <= residual_threshold
        && metrics.interface_temperature_continuity_ratio >= 0.999
        && metrics.advection_shift_coverage_ratio >= 1.0
        && metrics.coupled_interface_residual_ratio <= residual_threshold
        && metrics.heat_flux_realization_residual_ratio <= residual_threshold
        && metrics.interface_connectivity_coverage_ratio >= 1.0
        && metrics.thermal_network_residual_ratio <= residual_threshold
        && metrics.known_answer_coverage_ratio >= 1.0
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    FeaDiagnostic {
        code: "FEA_CHT_KNOWN_ANSWER".to_string(),
        severity,
        message: format!(
            "basis=heated_channel_conjugate_slab heated_channel_energy_residual_ratio={} conjugate_slab_flux_law_residual_ratio={} interface_temperature_continuity_ratio={} advection_shift_coverage_ratio={} coupled_interface_residual_ratio={} heat_flux_realization_residual_ratio={} interface_connectivity_coverage_ratio={} mesh_backed_interface_connectivity_ratio={} thermal_network_residual_ratio={} known_answer_coverage_ratio={}",
            metrics.heated_channel_energy_residual_ratio,
            metrics.conjugate_slab_flux_law_residual_ratio,
            metrics.interface_temperature_continuity_ratio,
            metrics.advection_shift_coverage_ratio,
            metrics.coupled_interface_residual_ratio,
            metrics.heat_flux_realization_residual_ratio,
            metrics.interface_connectivity_coverage_ratio,
            metrics.mesh_backed_interface_connectivity_ratio,
            metrics.thermal_network_residual_ratio,
            metrics.known_answer_coverage_ratio,
        ),
    }
}
