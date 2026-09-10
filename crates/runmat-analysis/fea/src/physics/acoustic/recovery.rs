use super::domain::AcousticDomainTopology;

#[derive(Debug, Clone, Copy)]
pub(super) struct AcousticKnownAnswerMetrics {
    pub tube_mode_alignment_error_ratio: f64,
    pub tube_pressure_variation_ratio: f64,
    pub cavity_mode_spacing_ratio: f64,
    pub cavity_reference_mode_count: usize,
    pub known_answer_coverage_ratio: f64,
}

pub(super) fn known_answer_metrics(
    topology: AcousticDomainTopology,
    pressure_magnitude: &[f64],
    drive_frequency_hz: f64,
    speed_of_sound_m_per_s: f64,
) -> AcousticKnownAnswerMetrics {
    let tube_length_m = topology.spacing[0] * topology.dims[0].saturating_sub(1).max(1) as f64;
    let fundamental_hz = speed_of_sound_m_per_s.max(1.0) / (2.0 * tube_length_m.max(1.0e-9));
    let nearest_mode = (drive_frequency_hz / fundamental_hz).round().max(1.0);
    let nearest_mode_frequency_hz = nearest_mode * fundamental_hz;
    let tube_mode_alignment_error_ratio = (drive_frequency_hz - nearest_mode_frequency_hz).abs()
        / drive_frequency_hz.abs().max(fundamental_hz);
    let (min_pressure, max_pressure) = pressure_magnitude
        .iter()
        .copied()
        .fold((f64::INFINITY, 0.0_f64), |(min_value, max_value), value| {
            (min_value.min(value), max_value.max(value))
        });
    let tube_pressure_variation_ratio = if min_pressure.is_finite() {
        (max_pressure - min_pressure).max(0.0) / max_pressure.max(1.0e-12)
    } else {
        0.0
    };
    let lengths: [f64; 3] = std::array::from_fn(|axis| {
        topology.spacing[axis] * topology.dims[axis].saturating_sub(1).max(1) as f64
    });
    let mut cavity_reference_modes = Vec::new();
    for nx in 0..=1 {
        for ny in 0..=1 {
            for nz in 0..=1 {
                if nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }
                let mode_sum = (nx as f64 / lengths[0].max(1.0e-9)).powi(2)
                    + (ny as f64 / lengths[1].max(1.0e-9)).powi(2)
                    + (nz as f64 / lengths[2].max(1.0e-9)).powi(2);
                cavity_reference_modes
                    .push(0.5 * speed_of_sound_m_per_s.max(1.0) * mode_sum.sqrt());
            }
        }
    }
    cavity_reference_modes.sort_by(f64::total_cmp);
    let cavity_reference_mode_count = cavity_reference_modes.len();
    let cavity_mode_spacing_ratio = cavity_reference_modes
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .filter(|spacing| *spacing > 1.0e-9)
        .fold(f64::INFINITY, f64::min)
        / fundamental_hz.max(1.0e-12);
    AcousticKnownAnswerMetrics {
        tube_mode_alignment_error_ratio,
        tube_pressure_variation_ratio,
        cavity_mode_spacing_ratio: if cavity_mode_spacing_ratio.is_finite() {
            cavity_mode_spacing_ratio
        } else {
            0.0
        },
        cavity_reference_mode_count,
        known_answer_coverage_ratio: if cavity_reference_mode_count > 0
            && !pressure_magnitude.is_empty()
            && drive_frequency_hz.is_finite()
        {
            1.0
        } else {
            0.0
        },
    }
}

pub(super) fn particle_velocity(
    pressure_real: &[f64],
    topology: AcousticDomainTopology,
    drive_frequency_hz: f64,
    density_kg_per_m3: f64,
) -> Vec<f64> {
    let omega = (2.0 * std::f64::consts::PI * drive_frequency_hz).max(1.0e-12);
    let impedance_scale = (density_kg_per_m3.max(1.0e-12) * omega).max(1.0e-12);
    let mut velocity = vec![0.0; pressure_real.len().max(1) * 3];
    for node in 0..pressure_real.len() {
        for axis in 0..3 {
            velocity[node * 3 + axis] =
                -axis_derivative(pressure_real, topology, node, axis) / impedance_scale;
        }
    }
    velocity
}

fn axis_derivative(
    pressure: &[f64],
    topology: AcousticDomainTopology,
    index: usize,
    axis: usize,
) -> f64 {
    if topology.dims[axis] <= 1 {
        return 0.0;
    }
    let previous = topology.neighbor(index, axis, -1);
    let next = topology.neighbor(index, axis, 1);
    let spacing = topology.spacing[axis].max(1.0e-12);
    match (previous, next) {
        (Some(previous), Some(next)) => (pressure[next] - pressure[previous]) / (2.0 * spacing),
        (Some(previous), None) => (pressure[index] - pressure[previous]) / spacing,
        (None, Some(next)) => (pressure[next] - pressure[index]) / spacing,
        (None, None) => 0.0,
    }
}
