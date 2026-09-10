//! Experimental incompressible finite-volume formulation and shared fluid-domain behavior.

mod diagnostics;
mod domain;
mod interface;
mod recovery;
mod reduced;
mod solve;

pub use diagnostics::{assembly_diagnostic, coupling_diagnostics, CfdCouplingEvidence};
pub use domain::{
    nominal_inlet_velocity_m_per_s, reynolds_number_for_velocity,
    validate_authored_boundary_conditions, velocity_scales, CfdDomainTopology,
};
pub use interface::{
    cell_centered_scalar, connectivity_coverage_ratio as interface_connectivity_coverage_ratio,
    face_count as interface_face_count, graph_edges_for_topology as interface_graph_edges,
    mean_axial_velocity, pressure_drop, resample_scalar_profile,
};
pub use recovery::recover_velocity_pressure;
pub use reduced::{solve_reduced_finite_volume, ReducedCfdSolveOptions};
pub use solve::residual_norms;

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{CfdDomain, DynamicViscosity, MassDensity, TurbulenceIntensity};

    use super::{
        diagnostics::known_answer_metrics,
        domain::{CfdBoundarySummary, CfdDomainTopology},
        recovery::build_run_fields,
        solve::{solve_velocity_pressure, CfdIterationControl},
    };

    #[test]
    fn reduced_channel_solve_closes_pressure_and_mass_balances() {
        let domain = CfdDomain {
            density: MassDensity::new(1.225),
            dynamic_viscosity: DynamicViscosity::new(1.81e-5),
            turbulence_intensity: TurbulenceIntensity::new(0.0),
        };
        let topology = CfdDomainTopology::implicit_channel(9);
        let boundary = CfdBoundarySummary {
            inlet_boundary_count: 1,
            outlet_boundary_count: 1,
            no_slip_wall_boundary_count: 2,
            slip_wall_boundary_count: 0,
            symmetry_boundary_count: 0,
            authored_boundary_count: 4,
            boundary_coverage_ratio: 1.0,
            wall_boundary_coverage_ratio: 1.0,
            nominal_inlet_velocity_m_per_s: 4.25,
            outlet_pressure_pa: 101_325.0,
        };

        let solution = solve_velocity_pressure(
            &domain,
            &boundary,
            &topology,
            0,
            &[1.0],
            CfdIterationControl {
                step_count: 1,
                max_linear_iters: 64,
                tolerance: 1.0e-8,
            },
        );
        let metrics = known_answer_metrics(&solution);

        assert!(
            (metrics.pressure_drop_balance_ratio - 1.0).abs() <= 1.0e-10,
            "pressure-drop balance ratio was {}",
            metrics.pressure_drop_balance_ratio
        );
        assert!(
            metrics.mass_flux_uniformity_ratio <= 1.0e-8,
            "mass-flux uniformity ratio was {}",
            metrics.mass_flux_uniformity_ratio
        );
        assert_eq!(metrics.pressure_monotonic_cell_fraction, 1.0);
        assert_eq!(metrics.known_answer_coverage_ratio, 1.0);
        assert!(
            solution.mass_balance_residual <= 1.0e-12,
            "mass-balance residual was {}",
            solution.mass_balance_residual
        );
    }

    #[test]
    fn recovered_fields_follow_control_volume_and_boundary_shapes() {
        let domain = CfdDomain {
            density: MassDensity::new(1.225),
            dynamic_viscosity: DynamicViscosity::new(1.81e-5),
            turbulence_intensity: TurbulenceIntensity::new(0.02),
        };
        let topology = CfdDomainTopology::implicit_channel(5);
        let boundary = CfdBoundarySummary {
            inlet_boundary_count: 1,
            outlet_boundary_count: 1,
            no_slip_wall_boundary_count: 2,
            slip_wall_boundary_count: 0,
            symmetry_boundary_count: 0,
            authored_boundary_count: 4,
            boundary_coverage_ratio: 1.0,
            wall_boundary_coverage_ratio: 1.0,
            nominal_inlet_velocity_m_per_s: 2.0,
            outlet_pressure_pa: 0.0,
        };
        let solution = solve_velocity_pressure(
            &domain,
            &boundary,
            &topology,
            0,
            &[1.0],
            CfdIterationControl {
                step_count: 3,
                max_linear_iters: 64,
                tolerance: 1.0e-8,
            },
        );

        let fields = build_run_fields(&domain, &solution);
        assert_eq!(fields[0].shape, vec![4, 3]);
        assert_eq!(fields[1].shape, vec![4]);
        assert_eq!(fields[2].shape, vec![4, 3]);
        assert_eq!(fields[3].shape, vec![2, 3]);
        assert_eq!(fields[4].shape, vec![3]);
        assert_eq!(fields[5].shape, vec![3]);
        assert_eq!(fields[6].shape, vec![1]);
    }
}
