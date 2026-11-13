use super::*;

fn multiply(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
fn wave(x: f64, amplitude: [f64; 2]) -> [f64; 2] {
    let phase = std::f64::consts::TAU * 0.2 * x / 2.0;
    multiply(amplitude, [phase.cos(), -phase.sin()])
}
// Exact uniform-simplex moments: E[x^n]=6h_n/[(n+1)(n+2)(n+3)].
fn mean_wave(xs: [f64; 4], amplitude: [f64; 2]) -> [f64; 2] {
    let k = std::f64::consts::TAU * 0.2 / 2.0;
    let mut h = [0.0; 32];
    h[0] = 1.0;
    for x in xs {
        for n in 1..32 {
            h[n] += x * h[n - 1];
        }
    }
    let mut coefficient = [1.0, 0.0];
    let mut mean = [0.0, 0.0];
    for (n, hn) in h.into_iter().enumerate() {
        if n > 0 {
            coefficient = [
                coefficient[1] * k / n as f64,
                -coefficient[0] * k / n as f64,
            ];
        }
        let moment = 6.0 * hn / ((n + 1) * (n + 2) * (n + 3)) as f64;
        for component in 0..2 {
            mean[component] += coefficient[component] * moment;
        }
    }
    multiply(amplitude, mean)
}

pub(super) fn errors(result: &CanonicalAcousticResponse, amplitude: [f64; 2]) -> (f64, f64, f64) {
    let norm = amplitude[0] * amplitude[0] + amplitude[1] * amplitude[1];
    let pressure = result
        .projection
        .nodes
        .iter()
        .zip(result.pressure_pa.canonical_values())
        .map(|(node, p)| {
            let reference = wave(node.coordinates_m.canonical_value()[0], amplitude);
            (p.real - reference[0]).powi(2) + (p.imaginary - reference[1]).powi(2)
        })
        .sum::<f64>()
        / (result.projection.nodes.len() as f64 * norm);
    let velocity = result
        .projection
        .elements
        .iter()
        .zip(result.element_velocity_m_per_s.canonical_values())
        .map(|(element, v)| {
            let p = mean_wave(
                std::array::from_fn(|i| {
                    result.projection.nodes[element.node_indices[i]]
                        .coordinates_m
                        .canonical_value()[0]
                }),
                amplitude,
            );
            (v.real[0] - p[0] / 6.0).powi(2)
                + (v.imaginary[0] - p[1] / 6.0).powi(2)
                + v.real[1].powi(2)
                + v.real[2].powi(2)
                + v.imaginary[1].powi(2)
                + v.imaginary[2].powi(2)
        })
        .sum::<f64>()
        / result.projection.elements.len() as f64;
    let intensity = result
        .element_mean_intensity_w_per_m2
        .canonical_values()
        .iter()
        .map(|value| (value[0] - norm / 12.0).powi(2) + value[1].powi(2) + value[2].powi(2))
        .sum::<f64>()
        / result.projection.elements.len() as f64;
    (pressure.sqrt(), velocity.sqrt(), intensity.sqrt())
}

pub(super) fn check_paired_planes(result: &CanonicalAcousticResponse, amplitude: [f64; 2]) {
    for (node, p) in result
        .projection
        .nodes
        .iter()
        .zip(result.pressure_pa.canonical_values())
    {
        if node.coordinates_m.canonical_value()[0] == 0.0
            || node.coordinates_m.canonical_value()[0] == 1.0
        {
            let reference = wave(node.coordinates_m.canonical_value()[0], amplitude);
            assert!(
                (p.real - reference[0]).abs() < 1e-10 && (p.imaginary - reference[1]).abs() < 1e-10
            );
        }
    }
}
