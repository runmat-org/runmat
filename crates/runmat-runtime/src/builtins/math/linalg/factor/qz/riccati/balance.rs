//! Reciprocal state/costate balancing of an extended Riccati pencil.
//!
//! For S = diag(D, D^-1, I), the similarity (S H S^-1, S J S^-1)
//! preserves the Hamiltonian/symplectic pair and its eigenvalues. It also
//! preserves cross weights: their two occurrences receive reciprocal row
//! and column factors. The graph of the selected invariant subspace changes
//! from X to D^-1 X D^-1, so the caller restores X = D X_bal D.
//!
//! This is the paired, power-of-two scaling used in the extended-pencil
//! approach of Benner, "Symplectic Balancing of Hamiltonian Matrices",
//! SIAM J. Sci. Comput. 22 (2001), and SciPy's CARE/DARE implementation.
//! Only off-diagonal entries guide the scaling. Unlike a general matrix
//! balancer, no state/costate permutation can change the Riccati partition.

use faer::c64;
use faer::mat::MatMut;

/// In-place, bounded equilibration. `state_scales` is exact powers of two.
pub(super) fn balance_extended_pencil(
    h: &mut MatMut<'_, c64>,
    j: &mut MatMut<'_, c64>,
    n: usize,
    state_scales: &mut [f64],
) -> Result<(), &'static str> {
    if state_scales.len() != n
        || n.checked_mul(2).is_none_or(|doubled| doubled > h.nrows())
        || h.nrows() != h.ncols()
        || j.shape() != h.shape()
    {
        return Err("Riccati balancing workspace dimensions are invalid");
    }
    state_scales.fill(1.0);
    let order = h.nrows();
    // Each accepted step strictly reduces the affected off-diagonal 1-norm.
    // Four sweeps bound preparation cost. The original-equation residual and
    // closed-loop stability checks govern publication after preconditioning.
    for _ in 0..4 {
        let mut changed = false;
        for state in 0..n {
            let mut baseline = 0.0;
            for matrix in [h.as_ref(), j.as_ref()] {
                baseline += affected_norm(matrix, state, n, 1.0);
            }
            if !baseline.is_finite() || baseline == 0.0 {
                continue;
            }
            let mut best = baseline;
            let mut factor = 1.0;
            for exponent in -8..=8 {
                if exponent == 0 {
                    continue;
                }
                let candidate = 2.0_f64.powi(exponent);
                let next_scale = state_scales[state] * candidate;
                if !(2.0_f64.powi(-256)..=2.0_f64.powi(256)).contains(&next_scale) {
                    continue;
                }
                let mut cost = 0.0;
                for matrix in [h.as_ref(), j.as_ref()] {
                    cost += affected_norm(matrix, state, n, candidate);
                }
                if cost.is_finite() && cost < best {
                    best = cost;
                    factor = candidate;
                }
            }
            if factor != 1.0 && best < 0.95 * baseline {
                for matrix in [h.as_mut(), j.as_mut()] {
                    apply_pair_scale(matrix, state, n, order, factor);
                }
                state_scales[state] *= factor;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(())
}

fn pair_factor(index: usize, state: usize, n: usize, factor: f64) -> f64 {
    if index == state {
        factor
    } else if index == n + state {
        factor.recip()
    } else {
        1.0
    }
}

fn affected_norm(matrix: faer::mat::MatRef<'_, c64>, state: usize, n: usize, factor: f64) -> f64 {
    let order = matrix.nrows();
    let dual = n + state;
    let mut total = 0.0;
    for row in [state, dual] {
        for col in 0..order {
            if row != col {
                let value = matrix[(row, col)];
                let scale = pair_factor(row, state, n, factor) / pair_factor(col, state, n, factor);
                total += scaled_magnitude(value, scale);
            }
        }
    }
    for row in 0..order {
        if row == state || row == dual {
            continue;
        }
        for col in [state, dual] {
            let value = matrix[(row, col)];
            let scale = pair_factor(row, state, n, factor) / pair_factor(col, state, n, factor);
            total += scaled_magnitude(value, scale);
        }
    }
    total
}

fn scaled_magnitude(value: c64, scale: f64) -> f64 {
    let original = value.re.abs().max(value.im.abs());
    let scaled = original * scale;
    if original != 0.0 && (scaled == 0.0 || !scaled.is_finite()) {
        f64::INFINITY
    } else {
        scaled
    }
}

fn apply_pair_scale(
    mut matrix: MatMut<'_, c64>,
    state: usize,
    n: usize,
    order: usize,
    factor: f64,
) {
    for col in 0..order {
        let column = pair_factor(col, state, n, factor);
        for row in 0..order {
            let ratio = pair_factor(row, state, n, factor) / column;
            let value = matrix[(row, col)];
            matrix[(row, col)] = c64::new(value.re * ratio, value.im * ratio);
        }
    }
}
