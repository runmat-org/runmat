use crate::{assembly::dofs::StructuralDofKind, assembly::AssemblySummary};

use super::{ModalBasisMode, ModalResidualVector};

pub(super) fn residual_directions_are_canonical(residuals: &[ModalResidualVector]) -> bool {
    let mut directions = Vec::<[f64; 3]>::with_capacity(residuals.len());
    for residual in residuals {
        let direction = residual.direction();
        if direction.iter().any(|value| !value.is_finite()) {
            return false;
        }
        let norm = direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if !scalar_close(norm, 1.0)
            || directions.iter().any(|existing| {
                existing
                    .iter()
                    .zip(direction)
                    .all(|(left, right)| scalar_close(*left, right))
            })
        {
            return false;
        }
        directions.push(direction);
    }
    true
}

pub(super) fn directional_mass_gram_is_canonical(gram: [[f64; 3]; 3], diagonal: [f64; 3]) -> bool {
    let entries_are_valid = (0..3).all(|row| {
        (0..3).all(|column| {
            gram[row][column].is_finite() && scalar_close(gram[row][column], gram[column][row])
        }) && gram[row][row] >= 0.0
            && scalar_close(gram[row][row], diagonal[row])
    });
    let principal_minors_are_nonnegative = (0..3).all(|row| {
        ((row + 1)..3).all(|column| {
            let bound = (gram[row][row] * gram[column][column]).sqrt();
            gram[row][column].abs() <= bound + 1.0e-12 * bound.max(1.0)
        })
    });
    let determinant = gram[0][0] * (gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1])
        - gram[0][1] * (gram[1][0] * gram[2][2] - gram[1][2] * gram[2][0])
        + gram[0][2] * (gram[1][0] * gram[2][1] - gram[1][1] * gram[2][0]);
    let determinant_scale = diagonal
        .into_iter()
        .fold(1.0_f64, |scale, value| scale * value.abs().max(1.0));
    entries_are_valid
        && principal_minors_are_nonnegative
        && determinant >= -1.0e-12 * determinant_scale
}

pub(super) fn directional_mass(gram: [[f64; 3]; 3], direction: [f64; 3]) -> f64 {
    direction
        .iter()
        .enumerate()
        .fold(0.0, |total, (row, left)| {
            total
                + direction
                    .iter()
                    .enumerate()
                    .map(|(column, right)| left * gram[row][column] * right)
                    .sum::<f64>()
        })
}

pub(super) fn scalar_close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 128.0 * f64::EPSILON * left.abs().max(right.abs()).max(1.0)
}

pub(super) fn directional_influence_vectors(summary: &AssemblySummary) -> [Vec<f64>; 3] {
    std::array::from_fn(|axis| {
        let mut vector = vec![0.0; summary.dof_count];
        for (row, value) in vector.iter_mut().enumerate() {
            if summary
                .operator
                .constrained_dofs()
                .get(row)
                .copied()
                .unwrap_or(true)
            {
                continue;
            }
            let expected = [
                StructuralDofKind::Ux,
                StructuralDofKind::Uy,
                StructuralDofKind::Uz,
            ][axis];
            if summary
                .structural_dof_layout
                .address(row)
                .is_some_and(|address| address.kind == expected)
            {
                *value = 1.0;
            }
        }
        vector
    })
}

pub(crate) fn directional_influence_vector(
    summary: &AssemblySummary,
    direction: [f64; 3],
) -> Vec<f64> {
    let axes = directional_influence_vectors(summary);
    let mut vector = vec![0.0; summary.dof_count];
    for (component, axis) in direction.into_iter().zip(axes) {
        for (value, axis_value) in vector.iter_mut().zip(axis) {
            *value += component * axis_value;
        }
    }
    vector
}

pub(super) fn relative_frequency_separation(frequencies: &[f64]) -> Vec<f64> {
    if frequencies.len() < 2 {
        return vec![1.0; frequencies.len()];
    }
    let mut separation = vec![f64::INFINITY; frequencies.len()];
    for (index, window) in frequencies.windows(2).enumerate() {
        let value =
            (window[1] - window[0]).abs() / window[0].abs().max(window[1].abs()).max(1.0e-12);
        separation[index] = separation[index].min(value);
        separation[index + 1] = separation[index + 1].min(value);
    }
    separation
        .into_iter()
        .map(|value| if value.is_finite() { value } else { 1.0 })
        .collect()
}

pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

pub(super) fn directional_metrics_are_consistent(
    modes: &[ModalBasisMode],
    total_mass: [f64; 3],
) -> bool {
    let mut cumulative = [0.0; 3];
    for mode in modes {
        for axis in 0..3 {
            let metric = mode.directional[axis];
            cumulative[axis] += metric.effective_mass;
            let expected_ratio = if total_mass[axis] > 0.0 {
                cumulative[axis] / total_mass[axis]
            } else {
                0.0
            };
            if metric.cumulative_effective_mass.to_bits() != cumulative[axis].to_bits()
                || metric.cumulative_mass_ratio.to_bits() != expected_ratio.to_bits()
            {
                return false;
            }
        }
    }
    true
}
