//! Small matrix-free Krylov authority shared by indefinite structural solve domains.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GmresPolicy {
    pub relative_tolerance: f64,
    pub restart: usize,
    pub maximum_iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct GmresResult {
    pub solution: Vec<f64>,
    pub iterations: u32,
    pub residual_norm: f64,
    pub converged: bool,
    pub cancelled: bool,
    pub breakdown: bool,
}

pub(super) fn restarted_gmres<E>(
    right_hand_side: &[f64],
    initial: &[f64],
    policy: GmresPolicy,
    apply: impl Fn(&[f64]) -> Result<Vec<f64>, E>,
    dot: impl Fn(&[f64], &[f64]) -> Result<f64, E>,
    cancelled: impl Fn() -> bool,
) -> Result<Option<GmresResult>, E> {
    let size = right_hand_side.len();
    if size == 0
        || initial.len() != size
        || policy.restart == 0
        || policy.maximum_iterations == 0
        || !policy.relative_tolerance.is_finite()
        || policy.relative_tolerance <= 0.0
        || right_hand_side
            .iter()
            .chain(initial)
            .any(|value| !value.is_finite())
    {
        return Ok(None);
    }
    let restart = policy.restart.min(size).min(policy.maximum_iterations);
    let mut solution = initial.to_vec();
    let Some(mut residual) = subtract(right_hand_side, &apply(&solution)?) else {
        return Ok(None);
    };
    let rhs_norm = dot(right_hand_side, right_hand_side)?.sqrt().max(1.0);
    let mut iterations = 0_u32;
    let Some(initial_norm) = norm(&residual, &dot)? else {
        return Ok(None);
    };
    let mut converged = initial_norm / rhs_norm <= policy.relative_tolerance;
    let mut observed_cancellation = false;
    let mut breakdown = false;

    while !converged && (iterations as usize) < policy.maximum_iterations {
        if cancelled() {
            observed_cancellation = true;
            break;
        }
        let Some(beta) = norm(&residual, &dot)? else {
            return Ok(None);
        };
        if beta <= f64::MIN_POSITIVE {
            breakdown = true;
            break;
        }
        let mut basis = vec![residual
            .iter()
            .map(|value| value / beta)
            .collect::<Vec<_>>()];
        let mut hessenberg = vec![vec![0.0; restart]; restart + 1];
        let mut cosines = vec![0.0; restart];
        let mut sines = vec![0.0; restart];
        let mut projected_rhs = vec![0.0; restart + 1];
        projected_rhs[0] = beta;
        let mut completed = 0;

        for column in 0..restart {
            if (iterations as usize) >= policy.maximum_iterations || cancelled() {
                observed_cancellation = cancelled();
                break;
            }
            let mut next = apply(&basis[column])?;
            if next.len() != size || next.iter().any(|value| !value.is_finite()) {
                return Ok(None);
            }
            for row in 0..=column {
                hessenberg[row][column] = dot(&basis[row], &next)?;
                axpy(-hessenberg[row][column], &basis[row], &mut next);
            }
            let Some(next_norm) = norm(&next, &dot)? else {
                return Ok(None);
            };
            hessenberg[column + 1][column] = next_norm;
            if hessenberg[column + 1][column] > f64::MIN_POSITIVE {
                basis.push(
                    next.iter()
                        .map(|value| value / hessenberg[column + 1][column])
                        .collect(),
                );
            }
            for row in 0..column {
                let first = hessenberg[row][column];
                let second = hessenberg[row + 1][column];
                hessenberg[row][column] = cosines[row] * first + sines[row] * second;
                hessenberg[row + 1][column] = -sines[row] * first + cosines[row] * second;
            }
            let diagonal = hessenberg[column][column];
            let subdiagonal = hessenberg[column + 1][column];
            let rotation_norm = diagonal.hypot(subdiagonal);
            if !rotation_norm.is_finite() || rotation_norm <= f64::MIN_POSITIVE {
                breakdown = true;
                break;
            }
            cosines[column] = diagonal / rotation_norm;
            sines[column] = subdiagonal / rotation_norm;
            hessenberg[column][column] = rotation_norm;
            hessenberg[column + 1][column] = 0.0;
            let projected = projected_rhs[column];
            projected_rhs[column] = cosines[column] * projected;
            projected_rhs[column + 1] = -sines[column] * projected;
            iterations = iterations.saturating_add(1);
            completed = column + 1;
            if projected_rhs[column + 1].abs() / rhs_norm <= policy.relative_tolerance
                || basis.len() == column + 1
            {
                break;
            }
        }
        if completed == 0 {
            breakdown = !observed_cancellation;
            break;
        }
        let Some(coefficients) = back_substitute(&hessenberg, &projected_rhs, completed) else {
            breakdown = true;
            break;
        };
        for (coefficient, direction) in coefficients.iter().zip(&basis) {
            axpy(*coefficient, direction, &mut solution);
        }
        let Some(next_residual) = subtract(right_hand_side, &apply(&solution)?) else {
            return Ok(None);
        };
        residual = next_residual;
        let Some(residual_norm) = norm(&residual, &dot)? else {
            return Ok(None);
        };
        converged = residual_norm / rhs_norm <= policy.relative_tolerance;
    }
    let Some(residual_norm) = norm(&residual, &dot)? else {
        return Ok(None);
    };
    Ok(Some(GmresResult {
        solution,
        iterations,
        residual_norm,
        converged,
        cancelled: observed_cancellation,
        breakdown,
    }))
}

fn norm<E>(
    vector: &[f64],
    dot: &impl Fn(&[f64], &[f64]) -> Result<f64, E>,
) -> Result<Option<f64>, E> {
    let norm = dot(vector, vector)?.sqrt();
    Ok(norm.is_finite().then_some(norm))
}

fn subtract(left: &[f64], right: &[f64]) -> Option<Vec<f64>> {
    (left.len() == right.len()).then(|| left.iter().zip(right).map(|(a, b)| a - b).collect())
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

fn back_substitute(matrix: &[Vec<f64>], rhs: &[f64], size: usize) -> Option<Vec<f64>> {
    let mut solution = vec![0.0; size];
    for row in (0..size).rev() {
        let remainder = ((row + 1)..size)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        if !matrix[row][row].is_finite() || matrix[row][row].abs() <= f64::MIN_POSITIVE {
            return None;
        }
        solution[row] = (rhs[row] - remainder) / matrix[row][row];
        if !solution[row].is_finite() {
            return None;
        }
    }
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(left: &[f64], right: &[f64]) -> Result<f64, std::convert::Infallible> {
        Ok(left.iter().zip(right).map(|(a, b)| a * b).sum())
    }

    #[test]
    fn restarted_solver_handles_nonsymmetric_system_and_cancellation() {
        let apply = |x: &[f64]| {
            Ok::<_, std::convert::Infallible>(vec![4.0 * x[0] + x[1], -2.0 * x[0] + 3.0 * x[1]])
        };
        let result = restarted_gmres(
            &[6.0, 4.0],
            &[0.0; 2],
            GmresPolicy {
                relative_tolerance: 1.0e-12,
                restart: 2,
                maximum_iterations: 20,
            },
            apply,
            dot,
            || false,
        )
        .unwrap()
        .unwrap();
        assert!(result.converged);
        assert!((result.solution[0] - 1.0).abs() <= 1.0e-12);
        assert!((result.solution[1] - 2.0).abs() <= 1.0e-12);

        let cancelled = restarted_gmres(
            &[6.0, 4.0],
            &[0.0; 2],
            GmresPolicy {
                relative_tolerance: 1.0e-12,
                restart: 2,
                maximum_iterations: 20,
            },
            apply,
            dot,
            || true,
        )
        .unwrap()
        .unwrap();
        assert!(cancelled.cancelled);
        assert!(!cancelled.converged);
    }

    #[test]
    fn operator_failure_is_not_converted_into_a_breakdown() {
        let result = restarted_gmres(
            &[1.0],
            &[0.0],
            GmresPolicy {
                relative_tolerance: 1.0e-12,
                restart: 1,
                maximum_iterations: 2,
            },
            |_value| Err::<Vec<f64>, _>("scoped worker failed"),
            |left, right| Ok::<_, &str>(dot(left, right).unwrap()),
            || false,
        );
        assert_eq!(result, Err("scoped worker failed"));
    }

    #[test]
    fn reduction_failure_is_not_converted_into_a_breakdown() {
        let result = restarted_gmres(
            &[1.0],
            &[0.0],
            GmresPolicy {
                relative_tolerance: 1.0e-12,
                restart: 1,
                maximum_iterations: 2,
            },
            |value| Ok::<_, &str>(value.to_vec()),
            |_left, _right| Err::<f64, _>("scoped reduction failed"),
            || false,
        );
        assert_eq!(result, Err("scoped reduction failed"));
    }
}
