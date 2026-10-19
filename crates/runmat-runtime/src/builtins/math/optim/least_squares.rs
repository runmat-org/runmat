//! Shared finite-difference Levenberg-Marquardt least-squares solver.

use std::future::Future;
use std::pin::Pin;

use nalgebra::{DMatrix, DVector};

use crate::builtins::math::optim::common::{optim_error, ScalarCooperation};
use crate::BuiltinResult;

mod traversal;
pub(crate) use traversal::{bound_gradient, jacobian_value, residual_norm_inf};
use traversal::{first_order_optimality, max_abs_difference, norm2_squared};

pub(crate) type ResidualFuture<'a> = Pin<Box<dyn Future<Output = BuiltinResult<Vec<f64>>> + 'a>>;
const MAX_LEAST_SQUARES_WORK_ELEMENTS: usize = 50_000_000;

pub(crate) trait LeastSquaresEvaluator {
    fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a>;
    fn cooperation(&mut self) -> Option<&mut ScalarCooperation> {
        None
    }
}

async fn checkpoint<E: LeastSquaresEvaluator>(
    evaluator: &mut E,
    elements: usize,
) -> BuiltinResult<()> {
    if let Some(cooperation) = evaluator.cooperation() {
        cooperation
            .completed(u64::try_from(elements).map_err(|_| {
                optim_error("optim", "cooperation work count exceeds supported range")
            })?)
            .await?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct LeastSquaresOptions {
    pub tol_x: f64,
    pub tol_fun: f64,
    pub max_iter: usize,
    pub max_fun_evals: usize,
    pub final_jacobian: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct LeastSquaresBounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl LeastSquaresBounds {
    pub(crate) fn unbounded(n: usize) -> Self {
        Self {
            lower: vec![f64::NEG_INFINITY; n],
            upper: vec![f64::INFINITY; n],
        }
    }

    pub(crate) fn validate(&self, name: &str, n: usize) -> BuiltinResult<()> {
        if self.lower.len() != n || self.upper.len() != n {
            return Err(optim_error(
                name,
                format!("{name}: bound vectors must match x0 length"),
            ));
        }
        for i in 0..n {
            if self.lower[i].is_nan()
                || self.upper[i].is_nan()
                || self.lower[i] > self.upper[i]
                || self.lower[i] == f64::INFINITY
                || self.upper[i] == f64::NEG_INFINITY
            {
                return Err(optim_error(
                    name,
                    format!("{name}: bounds must define a finite feasible iterate"),
                ));
            }
        }
        Ok(())
    }

    fn project(&self, x: &mut [f64]) {
        for (i, value) in x.iter_mut().enumerate() {
            *value = value.clamp(self.lower[i], self.upper[i]);
        }
    }

    fn active_lower(&self, i: usize, x: &[f64]) -> bool {
        self.lower[i].is_finite() && (x[i] - self.lower[i]).abs() <= 1.0e-10 * (1.0 + x[i].abs())
    }

    fn active_upper(&self, i: usize, x: &[f64]) -> bool {
        self.upper[i].is_finite() && (x[i] - self.upper[i]).abs() <= 1.0e-10 * (1.0 + x[i].abs())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LeastSquaresResult {
    pub x: Vec<f64>,
    pub residual: Vec<f64>,
    pub jacobian: Vec<f64>,
    pub residual_len: usize,
    pub variable_len: usize,
    pub resnorm: f64,
    pub exitflag: i32,
    pub iterations: usize,
    pub func_count: usize,
    pub first_order_optimality: f64,
    pub step_size: f64,
    pub message: String,
}

fn checked_least_squares_work(name: &str, m: usize, n: usize) -> BuiltinResult<usize> {
    let jacobian = m.checked_mul(n).ok_or_else(|| {
        optim_error(
            name,
            format!("{name}: least-squares work array size overflow"),
        )
    })?;
    let normal = n.checked_mul(n).ok_or_else(|| {
        optim_error(
            name,
            format!("{name}: least-squares normal-equation size overflow"),
        )
    })?;
    let total = jacobian
        .checked_add(normal)
        .and_then(|value| value.checked_add(m))
        .and_then(|value| value.checked_add(n))
        .ok_or_else(|| {
            optim_error(
                name,
                format!("{name}: least-squares workspace size overflow"),
            )
        })?;
    if total > MAX_LEAST_SQUARES_WORK_ELEMENTS {
        return Err(optim_error(
            name,
            format!(
                "{name}: least-squares workspace requires {total} elements, exceeding the supported limit of {MAX_LEAST_SQUARES_WORK_ELEMENTS}"
            ),
        ));
    }
    Ok(jacobian)
}

pub(crate) async fn solve_least_squares<E: LeastSquaresEvaluator>(
    name: &str,
    evaluator: &mut E,
    mut x: Vec<f64>,
    bounds: &LeastSquaresBounds,
    options: &LeastSquaresOptions,
) -> BuiltinResult<LeastSquaresResult> {
    let n = x.len();
    if n == 0 {
        return Err(optim_error(
            name,
            format!("{name}: initial guess cannot be empty"),
        ));
    }
    bounds.validate(name, n)?;
    bounds.project(&mut x);

    checkpoint(evaluator, 0).await?;
    let mut residual = evaluator.residual(&x).await?;
    if residual.is_empty() {
        return Err(optim_error(
            name,
            format!("{name}: function value must not be empty"),
        ));
    }
    let m = residual.len();
    let jacobian_len = checked_least_squares_work(name, m, n)?;
    let mut func_count = 1usize;
    let mut iterations = 0usize;
    let mut lambda = 1.0e-3;
    let mut last_jacobian = vec![0.0; jacobian_len];
    let mut last_optimality = f64::INFINITY;
    let mut last_step_size = 0.0;

    if residual_norm_inf(&residual, evaluator.cooperation()).await? <= options.tol_fun {
        let jacobian = finite_difference_jacobian(
            name,
            evaluator,
            &x,
            bounds,
            &residual,
            &mut func_count,
            options.max_fun_evals,
        )
        .await?
        .values;
        let optimality = first_order_optimality(&jacobian, &residual, n, evaluator).await?;
        return final_result(
            name,
            evaluator,
            x,
            residual,
            jacobian,
            bounds,
            options,
            1,
            iterations,
            func_count,
            optimality,
            0.0,
            "Local minimum found. Residual is within function tolerance.",
        )
        .await;
    }

    for iter in 0..options.max_iter {
        checkpoint(evaluator, x.len()).await?;
        iterations = iter + 1;
        if func_count >= options.max_fun_evals {
            return result(
                evaluator,
                x,
                residual,
                last_jacobian,
                0,
                iterations.saturating_sub(1),
                func_count,
                last_optimality,
                last_step_size,
                "Exceeded maximum function evaluations.",
            )
            .await;
        }

        let jacobian_eval = finite_difference_jacobian(
            name,
            evaluator,
            &x,
            bounds,
            &residual,
            &mut func_count,
            options.max_fun_evals,
        )
        .await?;
        let jacobian = jacobian_eval.values;
        let optimality = first_order_optimality(&jacobian, &residual, n, evaluator).await?;
        last_jacobian = jacobian.clone();
        last_optimality = optimality;
        if jacobian_eval.exhausted_budget {
            return result(
                evaluator,
                x,
                residual,
                jacobian,
                0,
                iterations,
                func_count,
                optimality,
                last_step_size,
                "Exceeded maximum function evaluations.",
            )
            .await;
        }
        if optimality <= options.tol_fun {
            return result(
                evaluator,
                x,
                residual,
                jacobian,
                1,
                iterations,
                func_count,
                optimality,
                last_step_size,
                "Local minimum found. First-order optimality is within tolerance.",
            )
            .await;
        }

        let j = DMatrix::from_row_slice(m, n, &jacobian);
        let f = DVector::from_column_slice(&residual);
        let gradient = j.transpose() * &f;
        let current_norm = norm2_squared(&residual, evaluator).await?;
        let mut accepted = false;

        for _ in 0..12 {
            if func_count >= options.max_fun_evals {
                return result(
                    evaluator,
                    x,
                    residual,
                    jacobian,
                    0,
                    iterations,
                    func_count,
                    optimality,
                    last_step_size,
                    "Exceeded maximum function evaluations.",
                )
                .await;
            }
            checkpoint(evaluator, 0).await?;
            let normal = j.transpose() * &j + DMatrix::<f64>::identity(n, n) * lambda;
            let rhs = -&gradient;
            let delta = normal.lu().solve(&rhs);
            checkpoint(evaluator, jacobian_len).await?;
            let Some(delta) = delta else {
                lambda *= 10.0;
                continue;
            };

            let mut trial = x
                .iter()
                .zip(delta.iter())
                .map(|(xi, di)| xi + di)
                .collect::<Vec<_>>();
            bounds.project(&mut trial);
            let step_norm = max_abs_difference(&trial, &x, evaluator).await?;
            if step_norm == 0.0 {
                lambda *= 10.0;
                continue;
            }
            let trial_residual = evaluator.residual(&trial).await?;
            func_count += 1;
            if trial_residual.len() != m {
                return Err(optim_error(
                    name,
                    format!("{name}: function output size changed during iteration"),
                ));
            }
            let trial_norm = norm2_squared(&trial_residual, evaluator).await?;
            if trial_norm <= current_norm {
                let x_norm = residual_norm_inf(&x, evaluator.cooperation()).await?;
                let improvement = current_norm - trial_norm;
                x = trial;
                residual = trial_residual;
                last_step_size = step_norm;
                lambda = (lambda * 0.3).max(1.0e-12);
                accepted = true;
                if residual_norm_inf(&residual, evaluator.cooperation()).await? <= options.tol_fun {
                    return final_result(
                        name,
                        evaluator,
                        x,
                        residual,
                        last_jacobian,
                        bounds,
                        options,
                        1,
                        iterations,
                        func_count,
                        optimality,
                        step_norm,
                        "Local minimum found. Residual is within function tolerance.",
                    )
                    .await;
                }
                if step_norm <= options.tol_x * (1.0 + x_norm) {
                    return final_result(
                        name,
                        evaluator,
                        x,
                        residual,
                        last_jacobian,
                        bounds,
                        options,
                        2,
                        iterations,
                        func_count,
                        optimality,
                        step_norm,
                        "Local minimum possible. Step size is within tolerance.",
                    )
                    .await;
                }
                if improvement <= options.tol_fun * (1.0 + current_norm) {
                    return final_result(
                        name,
                        evaluator,
                        x,
                        residual,
                        last_jacobian,
                        bounds,
                        options,
                        3,
                        iterations,
                        func_count,
                        optimality,
                        step_norm,
                        "Local minimum possible. Change in residual norm is within tolerance.",
                    )
                    .await;
                }
                break;
            }

            lambda *= 10.0;
            if func_count >= options.max_fun_evals {
                return result(
                    evaluator,
                    x,
                    residual,
                    last_jacobian,
                    0,
                    iterations,
                    func_count,
                    optimality,
                    last_step_size,
                    "Exceeded maximum function evaluations.",
                )
                .await;
            }
        }

        if !accepted {
            return result(
                evaluator,
                x,
                residual,
                last_jacobian,
                0,
                iterations,
                func_count,
                last_optimality,
                last_step_size,
                "Iteration stalled before convergence.",
            )
            .await;
        }
    }

    result(
        evaluator,
        x,
        residual,
        last_jacobian,
        0,
        iterations,
        func_count,
        last_optimality,
        last_step_size,
        "Exceeded maximum iterations.",
    )
    .await
}

struct FiniteDifferenceJacobian {
    values: Vec<f64>,
    exhausted_budget: bool,
}

async fn finite_difference_jacobian<E: LeastSquaresEvaluator>(
    name: &str,
    evaluator: &mut E,
    x: &[f64],
    bounds: &LeastSquaresBounds,
    residual: &[f64],
    func_count: &mut usize,
    max_fun_evals: usize,
) -> BuiltinResult<FiniteDifferenceJacobian> {
    let m = residual.len();
    let n = x.len();
    let mut jacobian = vec![0.0; checked_least_squares_work(name, m, n)?];
    for col in 0..n {
        checkpoint(evaluator, m).await?;
        if *func_count >= max_fun_evals {
            return Ok(FiniteDifferenceJacobian {
                values: jacobian,
                exhausted_budget: true,
            });
        }
        let mut perturbed = x.to_vec();
        let forward_step = f64::EPSILON.sqrt() * (x[col].abs() + 1.0);
        perturbed[col] = (x[col] + forward_step).clamp(bounds.lower[col], bounds.upper[col]);
        let mut actual_step = perturbed[col] - x[col];
        if actual_step == 0.0 {
            perturbed[col] = (x[col] - forward_step).clamp(bounds.lower[col], bounds.upper[col]);
            actual_step = perturbed[col] - x[col];
        }
        if actual_step == 0.0 || (bounds.active_lower(col, x) && bounds.active_upper(col, x)) {
            continue;
        }
        let next = evaluator.residual(&perturbed).await?;
        *func_count += 1;
        if next.len() != m {
            return Err(optim_error(
                name,
                format!("{name}: function output size changed during finite differencing"),
            ));
        }
        for first in (0..m).step_by(4096) {
            let end = m.min(first + 4096);
            for row in first..end {
                jacobian[row * n + col] = (next[row] - residual[row]) / actual_step;
            }
            checkpoint(evaluator, end - first).await?;
        }
    }
    Ok(FiniteDifferenceJacobian {
        values: jacobian,
        exhausted_budget: false,
    })
}

#[allow(clippy::too_many_arguments)]
async fn final_result<E: LeastSquaresEvaluator>(
    name: &str,
    evaluator: &mut E,
    x: Vec<f64>,
    residual: Vec<f64>,
    fallback_jacobian: Vec<f64>,
    bounds: &LeastSquaresBounds,
    options: &LeastSquaresOptions,
    exitflag: i32,
    iterations: usize,
    func_count: usize,
    fallback_optimality: f64,
    step_size: f64,
    message: &str,
) -> BuiltinResult<LeastSquaresResult> {
    if !options.final_jacobian || func_count.saturating_add(x.len()) > options.max_fun_evals {
        return result(
            evaluator,
            x,
            residual,
            fallback_jacobian,
            exitflag,
            iterations,
            func_count,
            fallback_optimality,
            step_size,
            message,
        )
        .await;
    }

    let mut final_func_count = func_count;
    let jacobian_eval = finite_difference_jacobian(
        name,
        evaluator,
        &x,
        bounds,
        &residual,
        &mut final_func_count,
        options.max_fun_evals,
    )
    .await?;
    let jacobian = jacobian_eval.values;
    let optimality = first_order_optimality(&jacobian, &residual, x.len(), evaluator).await?;
    result(
        evaluator,
        x,
        residual,
        jacobian,
        exitflag,
        iterations,
        final_func_count,
        optimality,
        step_size,
        message,
    )
    .await
}

async fn result<E: LeastSquaresEvaluator>(
    evaluator: &mut E,
    x: Vec<f64>,
    residual: Vec<f64>,
    jacobian: Vec<f64>,
    exitflag: i32,
    iterations: usize,
    func_count: usize,
    first_order_optimality: f64,
    step_size: f64,
    message: &str,
) -> BuiltinResult<LeastSquaresResult> {
    let resnorm = norm2_squared(&residual, evaluator).await?;
    Ok(LeastSquaresResult {
        variable_len: x.len(),
        residual_len: residual.len(),
        x,
        residual,
        jacobian,
        resnorm,
        exitflag,
        iterations,
        func_count,
        first_order_optimality,
        step_size,
        message: message.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    struct CountingEvaluator {
        calls: usize,
    }

    impl LeastSquaresEvaluator for CountingEvaluator {
        fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
            self.calls += 1;
            let residual = vec![x[0] - 1.0];
            Box::pin(async move { Ok(residual) })
        }
    }

    fn options(max_fun_evals: usize) -> LeastSquaresOptions {
        LeastSquaresOptions {
            tol_x: 1.0e-12,
            tol_fun: 1.0e-12,
            max_iter: 10,
            max_fun_evals,
            final_jacobian: false,
        }
    }

    #[test]
    fn bounds_reject_infinite_projected_iterates() {
        let bounds = LeastSquaresBounds {
            lower: vec![f64::INFINITY],
            upper: vec![f64::INFINITY],
        };
        let err = bounds
            .validate("lsqcurvefit", 1)
            .expect_err("invalid bounds");
        assert!(err.message().contains("finite feasible iterate"));
    }

    #[test]
    fn least_squares_workspace_guard_rejects_oversized_problem() {
        let err = checked_least_squares_work("lsqnonlin", MAX_LEAST_SQUARES_WORK_ELEMENTS, 2)
            .expect_err("oversized workspace");
        assert!(err.message().contains("workspace"));
    }

    #[test]
    fn solve_stops_before_extra_callback_after_budget_exhausted() {
        let mut evaluator = CountingEvaluator { calls: 0 };
        let bounds = LeastSquaresBounds::unbounded(2);
        let result = block_on(solve_least_squares(
            "lsqcurvefit",
            &mut evaluator,
            vec![0.0, 0.0],
            &bounds,
            &options(2),
        ))
        .expect("result");

        assert_eq!(result.func_count, 2);
        assert_eq!(evaluator.calls, 2);
        assert_eq!(result.exitflag, 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn already_zero_residual_does_not_hide_final_jacobian_callback_failure() {
        struct FinalFailure {
            calls: usize,
        }
        impl LeastSquaresEvaluator for FinalFailure {
            fn residual<'a>(&'a mut self, _: &'a [f64]) -> ResidualFuture<'a> {
                self.calls += 1;
                let first = self.calls == 1;
                Box::pin(async move {
                    if first {
                        Ok(vec![0.0])
                    } else {
                        Err(crate::build_runtime_error("original callback failure")
                            .with_identifier("RunMat:test:OriginalCallback")
                            .with_source(std::io::Error::other("original cause"))
                            .build())
                    }
                })
            }
        }
        let mut evaluator = FinalFailure { calls: 0 };
        let error = block_on(solve_least_squares(
            "fsolve",
            &mut evaluator,
            vec![0.0],
            &LeastSquaresBounds::unbounded(1),
            &options(10),
        ))
        .expect_err("not a synthetic zero Jacobian");
        assert_eq!(error.identifier(), Some("RunMat:test:OriginalCallback"));
        assert!(error.source.is_some());
        assert_eq!(evaluator.calls, 2);
    }
}
