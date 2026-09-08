//! Recovery of independent constraint-row multipliers from generalized reactions.

use std::collections::BTreeMap;

pub(super) fn solve_constraint_multipliers(
    rows: &[BTreeMap<usize, f64>],
    right_hand_side: &[f64],
) -> Option<Vec<f64>> {
    if rows.len() != right_hand_side.len() {
        return None;
    }
    if rows.is_empty() {
        return Some(Vec::new());
    }
    let inverse_diagonal = rows
        .iter()
        .map(|row| {
            let diagonal = row.values().map(|value| value * value).sum::<f64>();
            (diagonal > 1.0e-18 && diagonal.is_finite()).then_some(diagonal.recip())
        })
        .collect::<Option<Vec<_>>>()?;
    let mut solution = vec![0.0; rows.len()];
    let mut residual = right_hand_side.to_vec();
    let mut preconditioned = residual
        .iter()
        .zip(&inverse_diagonal)
        .map(|(residual, inverse)| residual * inverse)
        .collect::<Vec<_>>();
    let mut direction = preconditioned.clone();
    let mut rz = dot(&residual, &preconditioned);
    let scale = dot(right_hand_side, right_hand_side).sqrt().max(1.0);
    if dot(&residual, &residual).sqrt() / scale <= 1.0e-10 {
        return Some(solution);
    }
    for _ in 0..rows.len().saturating_mul(4).max(256) {
        let applied = apply_constraint_gram(rows, &direction);
        let denominator = dot(&direction, &applied);
        if !denominator.is_finite() || denominator <= 1.0e-18 {
            return None;
        }
        let alpha = rz / denominator;
        for index in 0..rows.len() {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * applied[index];
        }
        if dot(&residual, &residual).sqrt() / scale <= 1.0e-10 {
            return solution
                .iter()
                .all(|value| value.is_finite())
                .then_some(solution);
        }
        preconditioned = residual
            .iter()
            .zip(&inverse_diagonal)
            .map(|(residual, inverse)| residual * inverse)
            .collect();
        let next_rz = dot(&residual, &preconditioned);
        let beta = next_rz / rz;
        for (direction, preconditioned) in direction.iter_mut().zip(&preconditioned) {
            *direction = preconditioned + beta * *direction;
        }
        rz = next_rz;
    }
    None
}

fn apply_constraint_gram(rows: &[BTreeMap<usize, f64>], values: &[f64]) -> Vec<f64> {
    let mut dofs = BTreeMap::<usize, f64>::new();
    for (row, value) in rows.iter().zip(values) {
        for (dof, coefficient) in row {
            *dofs.entry(*dof).or_insert(0.0) += coefficient * value;
        }
    }
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|(dof, coefficient)| coefficient * dofs.get(dof).copied().unwrap_or(0.0))
                .sum()
        })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
