use crate::operator::{apply_m, OperatorSystem};

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

pub(super) fn normalize_mass(system: &OperatorSystem, vector: &mut [f64]) {
    let mv = apply_m(system, vector);
    let mass_norm = dot(vector, &mv).abs().sqrt().max(1.0e-12);
    for value in vector.iter_mut() {
        *value /= mass_norm;
    }
}

pub(super) fn orthonormalize_mass(system: &OperatorSystem, vector: &mut [f64], basis: &[Vec<f64>]) {
    for mode in basis {
        let mv = apply_m(system, mode);
        let projection = dot(vector, &mv);
        for (value, base) in vector.iter_mut().zip(mode.iter()) {
            *value -= projection * *base;
        }
    }
}

pub(super) fn relative_l2_update(previous: &[f64], current: &[f64]) -> f64 {
    if previous.len() != current.len() || previous.is_empty() {
        return 0.0;
    }
    let delta_norm = previous
        .iter()
        .zip(current.iter())
        .map(|(a, b)| {
            let d = b - a;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    let current_norm = current
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    delta_norm / current_norm.max(1.0e-12)
}
