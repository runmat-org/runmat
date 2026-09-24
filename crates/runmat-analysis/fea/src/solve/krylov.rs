//! Small matrix-free Krylov authority shared by indefinite structural solve domains.

use crate::{algebra::ScopedSpmvError, planning::resource::LocalExecutionPlan};

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
    restarted_gmres_with_updates(
        right_hand_side,
        initial,
        policy,
        apply,
        dot,
        cancelled,
        SerialUpdates,
    )
}

/// Runs the same GMRES recurrence with allocation-bound in-place vector updates. The
/// direct-harmonic caller retains the serial numerical policy through `restarted_gmres`.
pub(super) fn restarted_gmres_scoped(
    right_hand_side: &[f64],
    initial: &[f64],
    policy: GmresPolicy,
    apply: impl Fn(&[f64]) -> Result<Vec<f64>, ScopedSpmvError>,
    dot: impl Fn(&[f64], &[f64]) -> Result<f64, ScopedSpmvError>,
    cancelled: impl Fn() -> bool,
    plan: LocalExecutionPlan,
) -> Result<Option<GmresResult>, ScopedSpmvError> {
    restarted_gmres_with_updates(
        right_hand_side,
        initial,
        policy,
        apply,
        dot,
        cancelled,
        ScopedUpdates(plan),
    )
}

trait VectorUpdates<E> {
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) -> Result<(), E>;
    fn subtract(&self, left: &[f64], right: &[f64]) -> Result<Option<Vec<f64>>, E>;
    fn normalized(&self, input: &[f64], divisor: f64) -> Result<Vec<f64>, E>;
}

struct SerialUpdates;

impl<E> VectorUpdates<E> for SerialUpdates {
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) -> Result<(), E> {
        axpy(alpha, x, y);
        Ok(())
    }

    fn subtract(&self, left: &[f64], right: &[f64]) -> Result<Option<Vec<f64>>, E> {
        Ok(subtract(left, right))
    }

    fn normalized(&self, input: &[f64], divisor: f64) -> Result<Vec<f64>, E> {
        Ok(input.iter().map(|value| value / divisor).collect())
    }
}

struct ScopedUpdates(LocalExecutionPlan);

impl VectorUpdates<ScopedSpmvError> for ScopedUpdates {
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) -> Result<(), ScopedSpmvError> {
        crate::algebra::axpy_scoped(alpha, x, y, self.0)
    }

    fn subtract(&self, left: &[f64], right: &[f64]) -> Result<Option<Vec<f64>>, ScopedSpmvError> {
        if left.len() != right.len() {
            return Ok(None);
        }
        Ok(Some(crate::algebra::vector_sub_scoped(
            left, right, self.0,
        )?))
    }

    fn normalized(&self, input: &[f64], divisor: f64) -> Result<Vec<f64>, ScopedSpmvError> {
        crate::algebra::normalized_copy_scoped(input, divisor, self.0)
    }
}

fn restarted_gmres_with_updates<E>(
    right_hand_side: &[f64],
    initial: &[f64],
    policy: GmresPolicy,
    apply: impl Fn(&[f64]) -> Result<Vec<f64>, E>,
    dot: impl Fn(&[f64], &[f64]) -> Result<f64, E>,
    cancelled: impl Fn() -> bool,
    updates: impl VectorUpdates<E>,
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
    let Some(mut residual) = updates.subtract(right_hand_side, &apply(&solution)?)? else {
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
        let mut basis = vec![updates.normalized(&residual, beta)?];
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
                updates.axpy(-hessenberg[row][column], &basis[row], &mut next)?;
            }
            let Some(next_norm) = norm(&next, &dot)? else {
                return Ok(None);
            };
            hessenberg[column + 1][column] = next_norm;
            if hessenberg[column + 1][column] > f64::MIN_POSITIVE {
                basis.push(updates.normalized(&next, hessenberg[column + 1][column])?);
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
            updates.axpy(*coefficient, direction, &mut solution)?;
        }
        let Some(next_residual) = updates.subtract(right_hand_side, &apply(&solution)?)? else {
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
    use crate::planning::resource::{
        execute_scoped_chunks, LocalThreadCount, ScopedLocalExecutionError,
    };

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

    #[test]
    fn scoped_gmres_updates_match_the_independent_dense_solution_and_repeat() {
        let apply = |x: &[f64]| {
            Ok::<_, ScopedSpmvError>(vec![
                4.0 * x[0] + x[1],
                -2.0 * x[0] + 3.0 * x[1] + x[2],
                -x[1] + 2.0 * x[2],
            ])
        };
        let dot = |left: &[f64], right: &[f64]| {
            Ok::<_, ScopedSpmvError>(left.iter().zip(right).map(|(a, b)| a * b).sum())
        };
        let policy = GmresPolicy {
            relative_tolerance: 1.0e-12,
            restart: 3,
            maximum_iterations: 24,
        };
        let serial = restarted_gmres(&[6.0, 3.0, -4.0], &[0.0; 3], policy, apply, dot, || false)
            .unwrap()
            .unwrap();
        assert!(serial.converged);
        for (actual, expected) in serial.solution.iter().zip([1.0, 2.0, -1.0]) {
            assert!((actual - expected).abs() < 1.0e-11);
        }
        for threads in [2, 3, 4] {
            let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(threads).unwrap()).unwrap();
            let run = || {
                restarted_gmres_scoped(
                    &[6.0, 3.0, -4.0],
                    &[0.0; 3],
                    policy,
                    apply,
                    dot,
                    || false,
                    plan,
                )
                .unwrap()
                .unwrap()
            };
            assert_eq!(run(), serial);
            assert_eq!(run(), run());
        }
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap();
        let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
            Ok::<_, ()>(restarted_gmres_scoped(
                &[6.0, 3.0, -4.0],
                &[0.0; 3],
                policy,
                apply,
                dot,
                || false,
                plan,
            ))
        })
        .unwrap();
        assert!(matches!(
            &nested[0],
            Err(ScopedSpmvError::Execution(
                ScopedLocalExecutionError::NestedExecution
            ))
        ));
    }
}
