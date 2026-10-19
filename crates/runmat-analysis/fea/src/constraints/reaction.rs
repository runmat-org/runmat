//! Recovery of independent constraint-row multipliers from generalized reactions.

use std::collections::BTreeMap;

pub(crate) fn solve_constraint_multipliers(
    rows: &[BTreeMap<usize, f64>],
    right_hand_side: &[f64],
) -> Option<Vec<f64>> {
    if rows.len() != right_hand_side.len() {
        return None;
    }
    if rows.is_empty() {
        return Some(Vec::new());
    }
    let load_scale = right_hand_side
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !load_scale.is_finite() {
        return None;
    }
    let inverse_diagonal = rows
        .iter()
        .map(|row| {
            let diagonal = row.values().map(|value| value * value).sum::<f64>();
            (diagonal > 1.0e-18 && diagonal.is_finite()).then_some(diagonal.recip())
        })
        .collect::<Option<Vec<_>>>()?;
    if load_scale == 0.0 {
        return Some(vec![0.0; rows.len()]);
    }
    let mut solution = vec![0.0; rows.len()];
    // Normalize the right-hand side so that the convergence and breakdown
    // checks are independent of reaction units or the overall force scale.
    let mut residual = right_hand_side
        .iter()
        .map(|value| value / load_scale)
        .collect::<Vec<_>>();
    let scale = dot(&residual, &residual).sqrt();
    let mut preconditioned = residual
        .iter()
        .zip(&inverse_diagonal)
        .map(|(residual, inverse)| residual * inverse)
        .collect::<Vec<_>>();
    let mut direction = preconditioned.clone();
    let mut rz = dot(&residual, &preconditioned);
    for _ in 0..rows.len().saturating_mul(4).max(256) {
        let applied = apply_constraint_gram(rows, &direction);
        let denominator = dot(&direction, &applied);
        // The Gram quadratic form is positive for an independent direction.
        // A fixed absolute cutoff rejects valid small reactions even when the
        // row coefficients are well scaled (for example, an affine patch with
        // support reactions near solver roundoff).
        if !denominator.is_finite() || denominator <= 0.0 {
            return None;
        }
        let alpha = rz / denominator;
        for index in 0..rows.len() {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * applied[index];
        }
        if dot(&residual, &residual).sqrt() / scale <= 1.0e-10 {
            let solution = solution
                .into_iter()
                .map(|value| value * load_scale)
                .collect::<Vec<_>>();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_independent_reactions_do_not_trigger_a_fixed_breakdown_cutoff() {
        let rows = vec![BTreeMap::from([(0, 1.0)]), BTreeMap::from([(1, 1.0)])];
        for magnitude in [1.0e-10, 1.0e-20] {
            let rhs = [4.0 * magnitude, -2.0 * magnitude];
            let multipliers = solve_constraint_multipliers(&rows, &rhs)
                .expect("positive small Gram denominator is not a breakdown");
            for (actual, expected) in multipliers.iter().zip(rhs) {
                assert!((actual - expected).abs() < magnitude * 1.0e-12);
            }
        }
    }
}
