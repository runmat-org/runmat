use runmat_analysis_core::CfdDomain;

use super::{
    domain::{CfdBoundarySummary, CfdDomainTopology},
    solve::{solve_velocity_pressure, CfdIterationControl},
};

pub fn recover_velocity_pressure(
    domain: &CfdDomain,
    topology: &CfdDomainTopology,
    step_index: usize,
    nominal_velocity_m_per_s: f64,
    profile_scales: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut boundary_summary = CfdBoundarySummary::implicit_channel();
    boundary_summary.nominal_inlet_velocity_m_per_s =
        runmat_types::physical::Velocity::new(nominal_velocity_m_per_s);
    let solution = solve_velocity_pressure(
        domain,
        &boundary_summary,
        topology,
        step_index,
        profile_scales,
        CfdIterationControl {
            max_linear_iters: 32,
            tolerance: 1.0e-8,
        },
    );
    (solution.velocity, solution.pressure)
}
