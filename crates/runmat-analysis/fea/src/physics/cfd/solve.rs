use super::domain::{
    profile_scale as cfd_profile_scale,
    reynolds_number_for_velocity as cfd_reynolds_number_for_velocity, CfdBoundarySummary,
    CfdDomainTopology,
};

#[derive(Clone, Debug)]
pub(super) struct CfdVelocityPressureSolution {
    pub topology: CfdDomainTopology,
    pub velocity: Vec<f64>,
    pub pressure: Vec<f64>,
    pub residual_momentum: Vec<f64>,
    pub residual_continuity: Vec<f64>,
    pub mass_balance_residual: f64,
    pub pressure_drop_pa: f64,
    pub control_volume_count: usize,
    pub inlet_boundary_count: usize,
    pub outlet_boundary_count: usize,
    pub wall_boundary_count: usize,
    pub no_slip_wall_boundary_count: usize,
    pub slip_wall_boundary_count: usize,
    pub symmetry_boundary_count: usize,
    pub authored_boundary_count: usize,
    pub boundary_coverage_ratio: f64,
    pub wall_boundary_coverage_ratio: f64,
    pub inlet_velocity_realization_ratio: f64,
    pub nominal_inlet_velocity_m_per_s: f64,
    pub outlet_pressure_pa: f64,
    pub pressure_correction_iteration_count: usize,
    pub pressure_correction_residual_ratio: f64,
    pub velocity_correction_residual_ratio: f64,
    pub transient_scale_min: f64,
    pub transient_scale_max: f64,
    pub transient_scale_variation: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CfdIterationControl {
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
}

pub(super) fn solve_velocity_pressure(
    domain: &runmat_analysis_core::CfdDomain,
    boundary_summary: &CfdBoundarySummary,
    topology: &CfdDomainTopology,
    step_index: usize,
    profile_scales: &[f64],
    control: CfdIterationControl,
) -> CfdVelocityPressureSolution {
    let node_count: usize = topology.node_count.max(2);
    let profile_scale = cfd_profile_scale(profile_scales, step_index);
    let nominal_inlet_velocity = boundary_summary.nominal_inlet_velocity_m_per_s;
    let inlet_velocity = nominal_inlet_velocity * profile_scale;
    let reynolds = cfd_reynolds_number_for_velocity(domain, nominal_inlet_velocity).max(1.0);
    let hydraulic_diameter_m = topology.hydraulic_diameter_m;
    let friction_factor = if reynolds <= 2300.0 {
        64.0 / reynolds
    } else {
        0.3164 / reynolds.powf(0.25)
    };
    let friction_gradient_pa_per_m = 0.5
        * domain.density.kilograms_per_cubic_meter
        * inlet_velocity
        * inlet_velocity.abs()
        * friction_factor
        / hydraulic_diameter_m;
    let pressure_drop_pa = (friction_gradient_pa_per_m * topology.domain_length_m).max(0.0);
    let denom = node_count.saturating_sub(1).max(1) as f64;
    let mut axial_velocity = vec![inlet_velocity; node_count];
    let mut pressure = (0..node_count)
        .map(|node| {
            let xi = node as f64 / denom;
            0.5 * pressure_drop_pa * (1.0 - xi)
        })
        .collect::<Vec<_>>();
    let target_pressure = (0..node_count)
        .map(|node| {
            let xi = node as f64 / denom;
            pressure_drop_pa * (1.0 - xi)
        })
        .collect::<Vec<_>>();
    let correction_iters = control.max_linear_iters.max(1);
    let correction_tolerance = control.tolerance.max(1.0e-12);
    let mut pressure_correction_residual_ratio = f64::INFINITY;
    let mut velocity_correction_residual_ratio = f64::INFINITY;
    let mut pressure_correction_iteration_count = 0usize;
    for iteration in 0..correction_iters {
        let previous_pressure = pressure.clone();
        let previous_velocity = axial_velocity.clone();
        for node in 0..node_count {
            pressure[node] = 0.35 * pressure[node] + 0.65 * target_pressure[node];
        }
        for node in 0..node_count {
            if node == 0 {
                axial_velocity[node] = inlet_velocity;
                continue;
            }
            if node + 1 == node_count {
                axial_velocity[node] = axial_velocity[node - 1];
                continue;
            }
            let gradient = (pressure[node + 1] - pressure[node - 1]) / (2.0 * topology.dx_m);
            let pressure_driven_speed = ((-2.0 * gradient * hydraulic_diameter_m)
                / (domain.density.kilograms_per_cubic_meter.max(1.0e-12)
                    * friction_factor.max(1.0e-12)))
            .max(0.0)
            .sqrt();
            axial_velocity[node] = 0.50 * axial_velocity[node] + 0.50 * pressure_driven_speed;
        }
        axial_velocity[node_count - 1] = axial_velocity[node_count - 2];

        let pressure_correction_norm = pressure
            .iter()
            .zip(previous_pressure.iter())
            .map(|(current, previous)| (current - previous) * (current - previous))
            .sum::<f64>()
            .sqrt();
        let pressure_scale = target_pressure
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        pressure_correction_residual_ratio = pressure_correction_norm / pressure_scale;
        let velocity_correction_norm = axial_velocity
            .iter()
            .zip(previous_velocity.iter())
            .map(|(current, previous)| (current - previous) * (current - previous))
            .sum::<f64>()
            .sqrt();
        let velocity_scale = axial_velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(inlet_velocity.abs())
            .max(1.0e-12);
        velocity_correction_residual_ratio = velocity_correction_norm / velocity_scale;
        pressure_correction_iteration_count = iteration + 1;
        if pressure_correction_residual_ratio <= correction_tolerance
            && velocity_correction_residual_ratio <= correction_tolerance
        {
            break;
        }
    }
    pressure = target_pressure;
    for node in 1..node_count.saturating_sub(1) {
        let gradient = (pressure[node + 1] - pressure[node - 1]) / (2.0 * topology.dx_m);
        axial_velocity[node] = ((-2.0 * gradient * hydraulic_diameter_m)
            / (domain.density.kilograms_per_cubic_meter.max(1.0e-12)
                * friction_factor.max(1.0e-12)))
        .max(0.0)
        .sqrt();
    }
    axial_velocity[0] = inlet_velocity;
    axial_velocity[node_count - 1] = axial_velocity[node_count - 2];

    let mut velocity = Vec::with_capacity(node_count * 3);
    for (node, axial) in axial_velocity.iter().copied().enumerate() {
        let xi = node as f64 / denom;
        let recirculation = (2.0 * std::f64::consts::PI * xi).sin()
            * axial
            * domain.turbulence_intensity.ratio.clamp(0.0, 1.0)
            * 0.02;
        velocity.extend_from_slice(&[axial, recirculation, 0.0]);
    }

    let (residual_momentum, residual_continuity) =
        residual_norms(&velocity, &pressure, domain, topology, control.step_count);
    for value in &mut pressure {
        *value += boundary_summary.outlet_pressure_pa;
    }
    let mass_balance_residual = residual_continuity.iter().copied().fold(0.0_f64, f64::max);
    let inlet_velocity_realization_ratio =
        inlet_velocity.abs() / nominal_inlet_velocity.abs().max(1.0e-12);
    let (transient_scale_min, transient_scale_max) = cfd_transient_scale_bounds(profile_scales);

    CfdVelocityPressureSolution {
        topology: topology.clone(),
        velocity,
        pressure,
        residual_momentum,
        residual_continuity,
        mass_balance_residual,
        pressure_drop_pa,
        control_volume_count: topology.control_volume_count,
        inlet_boundary_count: boundary_summary.inlet_boundary_count,
        outlet_boundary_count: boundary_summary.outlet_boundary_count,
        wall_boundary_count: boundary_summary.wall_boundary_count(),
        no_slip_wall_boundary_count: boundary_summary.no_slip_wall_boundary_count,
        slip_wall_boundary_count: boundary_summary.slip_wall_boundary_count,
        symmetry_boundary_count: boundary_summary.symmetry_boundary_count,
        authored_boundary_count: boundary_summary.authored_boundary_count,
        boundary_coverage_ratio: boundary_summary.boundary_coverage_ratio,
        wall_boundary_coverage_ratio: boundary_summary.wall_boundary_coverage_ratio,
        inlet_velocity_realization_ratio,
        nominal_inlet_velocity_m_per_s: nominal_inlet_velocity,
        outlet_pressure_pa: boundary_summary.outlet_pressure_pa,
        pressure_correction_iteration_count,
        pressure_correction_residual_ratio,
        velocity_correction_residual_ratio,
        transient_scale_min,
        transient_scale_max,
        transient_scale_variation: (transient_scale_max - transient_scale_min).abs(),
    }
}

fn cfd_transient_scale_bounds(profile_scales: &[f64]) -> (f64, f64) {
    if profile_scales.is_empty() {
        return (1.0, 1.0);
    }
    let (min, max) = profile_scales
        .iter()
        .copied()
        .filter(|scale| scale.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), scale| {
            (min.min(scale), max.max(scale))
        });
    if min.is_finite() && max.is_finite() {
        (min, max)
    } else {
        (1.0, 1.0)
    }
}

pub fn residual_norms(
    velocity: &[f64],
    pressure: &[f64],
    domain: &runmat_analysis_core::CfdDomain,
    topology: &CfdDomainTopology,
    step_count: usize,
) -> (Vec<f64>, Vec<f64>) {
    let node_count = pressure.len().max(1);
    let mean_axial_velocity = (0..node_count)
        .map(|node| velocity.get(node * 3).copied().unwrap_or(0.0))
        .sum::<f64>()
        / node_count as f64;
    let reynolds = cfd_reynolds_number_for_velocity(domain, mean_axial_velocity).max(1.0);
    let hydraulic_diameter_m = topology.hydraulic_diameter_m.max(1.0e-12);
    let friction_factor = if reynolds <= 2300.0 {
        64.0 / reynolds
    } else {
        0.3164 / reynolds.powf(0.25)
    };
    let friction_gradient_pa_per_m = 0.5
        * domain.density.kilograms_per_cubic_meter.max(1.0e-12)
        * mean_axial_velocity
        * mean_axial_velocity.abs()
        * friction_factor
        / hydraulic_diameter_m;
    let dx = topology.dx_m.max(1.0e-12);
    let mut momentum_base = 0.0_f64;
    let mut continuity_base = 0.0_f64;
    for node in 0..node_count {
        let prev = node.saturating_sub(1);
        let next = (node + 1).min(node_count - 1);
        let velocity_prev = velocity.get(prev * 3).copied().unwrap_or(0.0);
        let velocity_next = velocity.get(next * 3).copied().unwrap_or(0.0);
        let pressure_prev = pressure.get(prev).copied().unwrap_or(0.0);
        let pressure_next = pressure.get(next).copied().unwrap_or(0.0);
        let stencil_width = if prev == next {
            1.0
        } else {
            (next - prev) as f64 * dx
        };
        let divergence = (velocity_next - velocity_prev) / stencil_width.max(1.0e-12);
        let pressure_gradient = (pressure_next - pressure_prev) / stencil_width.max(1.0e-12);
        continuity_base += divergence.abs();
        momentum_base += (pressure_gradient + friction_gradient_pa_per_m).abs();
    }
    let velocity_scale = mean_axial_velocity.abs().max(1.0e-9);
    let pressure_scale = pressure
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let continuity_base = (continuity_base / (node_count as f64 * velocity_scale)).clamp(0.0, 1.0);
    let momentum_base = (momentum_base / (node_count as f64 * pressure_scale)).clamp(0.0, 1.0);
    let residual_count = step_count.max(1);
    (
        vec![momentum_base; residual_count],
        vec![continuity_base; residual_count],
    )
}
