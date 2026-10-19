#[derive(Clone, Copy)]
pub(super) struct DenseSecondOrderOperators<'a> {
    pub size: usize,
    pub stiffness: &'a [f64],
    pub mass: &'a [f64],
    pub damping: &'a [f64],
}

pub(super) fn multiply(matrix: &[f64], size: usize, vector: &[f64]) -> Vec<f64> {
    (0..size)
        .map(|row| {
            (0..size)
                .map(|column| matrix[row * size + column] * vector[column])
                .sum()
        })
        .collect()
}

pub(super) fn subtract_three(left: &[f64], middle: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(middle)
        .zip(right)
        .map(|((left, middle), right)| left - middle - right)
        .collect()
}

pub(super) fn linear_combination(
    mass: &[f64],
    damping_scale: f64,
    damping: &[f64],
    stiffness_scale: f64,
    stiffness: &[f64],
) -> Vec<f64> {
    mass.iter()
        .zip(damping)
        .zip(stiffness)
        .map(|((mass, damping), stiffness)| {
            mass + damping_scale * damping + stiffness_scale * stiffness
        })
        .collect()
}

pub(super) fn relative_residual(
    operators: DenseSecondOrderOperators<'_>,
    displacement: &[f64],
    velocity: &[f64],
    acceleration: &[f64],
    load: &[f64],
) -> f64 {
    let inertia = multiply(operators.mass, operators.size, acceleration);
    let damping = multiply(operators.damping, operators.size, velocity);
    let elastic = multiply(operators.stiffness, operators.size, displacement);
    let residual = load
        .iter()
        .zip(&inertia)
        .zip(&damping)
        .zip(&elastic)
        .map(|(((load, inertia), damping), elastic)| load - inertia - damping - elastic)
        .collect::<Vec<_>>();
    norm(&residual)
        / (norm(load) + norm(&inertia) + norm(&damping) + norm(&elastic)).max(f64::MIN_POSITIVE)
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}
