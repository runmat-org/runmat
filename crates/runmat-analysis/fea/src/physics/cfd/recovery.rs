use runmat_analysis_core::{AnalysisField, AnalysisModel, CfdDomain};

use crate::{
    FEA_FIELD_CFD_PRESSURE, FEA_FIELD_CFD_RESIDUAL_CONTINUITY, FEA_FIELD_CFD_RESIDUAL_MOMENTUM,
    FEA_FIELD_CFD_REYNOLDS_NUMBER, FEA_FIELD_CFD_VELOCITY, FEA_FIELD_CFD_VORTICITY,
    FEA_FIELD_CFD_WALL_SHEAR_STRESS,
};

use super::{
    domain::{reynolds_number_for_velocity, CfdBoundarySummary, CfdDomainTopology},
    solve::{solve_velocity_pressure, CfdIterationControl, CfdVelocityPressureSolution},
};

pub fn recover_velocity_pressure(
    model: &AnalysisModel,
    domain: &CfdDomain,
    topology: &CfdDomainTopology,
    step_index: usize,
    profile_scales: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let boundary_summary = CfdBoundarySummary::implicit_channel(model, topology.node_count);
    let solution = solve_velocity_pressure(
        domain,
        &boundary_summary,
        topology,
        step_index,
        profile_scales,
        CfdIterationControl {
            step_count: 1,
            max_linear_iters: 32,
            tolerance: 1.0e-8,
        },
    );
    (solution.velocity, solution.pressure)
}

pub(super) fn build_run_fields(
    domain: &CfdDomain,
    solution: &CfdVelocityPressureSolution,
) -> Vec<AnalysisField> {
    let node_count = solution.pressure.len();
    let velocity = cell_centered_vector_from_nodal(&solution.velocity, node_count);
    let pressure = cell_centered_scalar_from_nodal(&solution.pressure);
    let cell_count = pressure.len().max(1);
    let vorticity = recover_vorticity(&velocity, cell_count, solution.topology.dx_m);
    let boundary_face_count = solution.wall_boundary_count.max(1);
    let wall_shear_stress = recover_wall_shear_stress(domain, &velocity, boundary_face_count);
    let residual_count = solution.residual_momentum.len();

    vec![
        AnalysisField::host_f64(FEA_FIELD_CFD_VELOCITY, vec![cell_count, 3], velocity),
        AnalysisField::host_f64(FEA_FIELD_CFD_PRESSURE, vec![cell_count], pressure),
        AnalysisField::host_f64(FEA_FIELD_CFD_VORTICITY, vec![cell_count, 3], vorticity),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_WALL_SHEAR_STRESS,
            vec![boundary_face_count, 3],
            wall_shear_stress,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_RESIDUAL_MOMENTUM,
            vec![residual_count],
            solution.residual_momentum.clone(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_RESIDUAL_CONTINUITY,
            vec![residual_count],
            solution.residual_continuity.clone(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CFD_REYNOLDS_NUMBER,
            vec![1],
            vec![reynolds_number_for_velocity(
                domain,
                solution.nominal_inlet_velocity_m_per_s,
            )],
        ),
    ]
}

fn recover_vorticity(velocity: &[f64], node_count: usize, dx_m: f64) -> Vec<f64> {
    let mut vorticity = vec![0.0; node_count * 3];
    if node_count < 2 {
        return vorticity;
    }
    let dx_m = dx_m.max(1.0e-12);

    for node in 0..node_count {
        let prev = node.saturating_sub(1);
        let next = (node + 1).min(node_count - 1);
        let prev_base = prev * 3;
        let next_base = next * 3;
        let dvx = velocity.get(next_base).copied().unwrap_or(0.0)
            - velocity.get(prev_base).copied().unwrap_or(0.0);
        let dvy = velocity.get(next_base + 1).copied().unwrap_or(0.0)
            - velocity.get(prev_base + 1).copied().unwrap_or(0.0);
        let base = node * 3;
        vorticity[base] = 0.0;
        vorticity[base + 1] = -dvx / (2.0 * dx_m);
        vorticity[base + 2] = dvy / (2.0 * dx_m);
    }
    vorticity
}

fn recover_wall_shear_stress(domain: &CfdDomain, velocity: &[f64], field_count: usize) -> Vec<f64> {
    let mut shear = vec![0.0; field_count * 3];
    let viscosity = domain.dynamic_viscosity.pascal_seconds;
    for index in 0..field_count {
        let base = index * 3;
        shear[base] = viscosity * velocity.get(base).copied().unwrap_or(0.0);
        shear[base + 1] = viscosity * velocity.get(base + 1).copied().unwrap_or(0.0);
    }
    shear
}

fn cell_centered_vector_from_nodal(nodal: &[f64], node_count: usize) -> Vec<f64> {
    let cell_count = node_count.saturating_sub(1).max(1);
    let mut cell_values = Vec::with_capacity(cell_count * 3);
    for cell in 0..cell_count {
        let left = cell.min(node_count.saturating_sub(1));
        let right = (cell + 1).min(node_count.saturating_sub(1));
        for component in 0..3 {
            let left_value = nodal.get(left * 3 + component).copied().unwrap_or(0.0);
            let right_value = nodal
                .get(right * 3 + component)
                .copied()
                .unwrap_or(left_value);
            cell_values.push(0.5 * (left_value + right_value));
        }
    }
    cell_values
}

fn cell_centered_scalar_from_nodal(nodal: &[f64]) -> Vec<f64> {
    let node_count = nodal.len();
    let cell_count = node_count.saturating_sub(1).max(1);
    let mut cell_values = Vec::with_capacity(cell_count);
    for cell in 0..cell_count {
        let left = cell.min(node_count.saturating_sub(1));
        let right = (cell + 1).min(node_count.saturating_sub(1));
        let left_value = nodal.get(left).copied().unwrap_or(0.0);
        let right_value = nodal.get(right).copied().unwrap_or(left_value);
        cell_values.push(0.5 * (left_value + right_value));
    }
    cell_values
}
