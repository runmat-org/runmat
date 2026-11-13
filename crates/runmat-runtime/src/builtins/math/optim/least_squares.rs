//! Shared original residual/Jacobian owner and distinct LM/reflective controllers.

use std::future::Future;
use std::pin::Pin;

use nalgebra::{DMatrix, DVector};

use crate::builtins::math::optim::common::{optim_error, ScalarCooperation};
use crate::BuiltinResult;

pub(super) mod equation;
mod terminal;
use terminal::{finish, result, DerivativeResult};
mod derivative;
pub(crate) use derivative::{Jacobian, Opaque};
mod criteria;
pub(crate) use criteria::Criteria;
mod traversal;
mod initial;
mod lm;
mod selection;
mod reflective;
mod display;
pub(crate) use selection::{solve_controlled, solve_selected, Method};
pub(crate) use traversal::{
    bound_gradient, jacobian_value, original_derivative_columns, residual_norm_inf,
};
use traversal::{first_order_optimality, max_abs_difference, norm2_squared};

pub(crate) type ResidualFuture<'a> = Pin<Box<dyn Future<Output = BuiltinResult<Vec<f64>>> + 'a>>;
const MAX_LEAST_SQUARES_WORK_ELEMENTS: usize = 50_000_000;

pub(crate) trait LeastSquaresEvaluator {
    /// A dense numerical method may borrow a supplied carrier without retiring
    /// the original accepted owner needed by terminal publication.
    async fn dense_jacobian(&mut self) -> BuiltinResult<Option<Vec<f64>>> {
        Ok(None)
    }

    /// Move the derivative paired to the accepted residual, never a trial probe.
    fn take_jacobian(&mut self) -> Option<Jacobian> {
        None
    }
    fn supplies_jacobian(&self) -> bool {
        false
    }
    /// Commit only a residual accepted as the current solver iterate.
    fn accept_residual(&mut self) {}
    /// Terminal transfer of the original Jacobian paired to that accepted iterate.
    fn take_analytic_jacobian(&mut self) -> Option<Vec<f64>> {
        None
    }
    /// The exact Jacobian paired with the current accepted residual.
    /// Row-major numerical workspace, not a separate callback evaluation.
    fn analytic_jacobian(&self) -> Option<&[f64]> {
        None
    }
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
    pub criteria: Option<Criteria>,
    pub tol_x: f64,
    pub tol_fun: f64,
    pub max_iter: usize,
    pub max_fun_evals: usize,
    pub final_gradient: bool,
    pub final_jacobian: bool,
    pub scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JacobianSource {
    Supplied,
    FiniteDifference,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PointEvidence {
    Evaluated,
    Supplied,
}
#[derive(Clone, Debug)]
pub(crate) enum PublishedJacobian {
    RowMajor(Vec<f64>),
    Supplied(crate::execution::RootedValue),
}
impl PublishedJacobian {
    pub(crate) fn row_major(&self) -> Option<&[f64]> {
        match self {
            Self::RowMajor(values) => Some(values),
            Self::Supplied(_) => None,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct LeastSquaresResult {
    pub point_evidence: PointEvidence,
    pub jacobian_source: JacobianSource,
    pub gradient: Option<Vec<f64>>,
    pub cg_iterations: usize,
    pub method: Method,
    pub x: Vec<f64>,
    pub residual: Vec<f64>,
    pub jacobian: PublishedJacobian,
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

pub(super) fn checked_least_squares_work(name: &str, m: usize, n: usize) -> BuiltinResult<usize> {
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

fn checked_jacobian_work(name: &str, m: usize, n: usize) -> BuiltinResult<usize> {
    let matrix = m
        .checked_mul(n)
        .ok_or_else(|| optim_error(name, "Jacobian workspace extent overflow"))?;
    let total = matrix
        .checked_add(m)
        .and_then(|v| v.checked_add(n))
        .ok_or_else(|| optim_error(name, "Jacobian workspace extent overflow"))?;
    if total > MAX_LEAST_SQUARES_WORK_ELEMENTS {
        return Err(optim_error(
            name,
            "Jacobian and residual workspace exceeds the supported element limit",
        ));
    }
    Ok(matrix)
}

pub(crate) fn solve_least_squares<'a, E: LeastSquaresEvaluator + 'a>(
    name: &'a str,
    evaluator: &'a mut E,
    x: Vec<f64>,
    bounds: &'a LeastSquaresBounds,
    options: &'a LeastSquaresOptions,
) -> impl Future<Output = BuiltinResult<LeastSquaresResult>> + 'a {
    solve_selected(
        name,
        evaluator,
        x,
        bounds,
        options,
        Method::LevenbergMarquardt,
        runmat_builtins::OptimDisplay::Off,
    )
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
    scaling: &crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling,
) -> BuiltinResult<FiniteDifferenceJacobian> {
    let m = residual.len();
    let n = x.len();
    if let Some(values) = evaluator.dense_jacobian().await? {
        if m.checked_mul(n) != Some(values.len()) {
            return Err(optim_error(
                name,
                "supplied dense Jacobian geometry changed",
            ));
        }
        return Ok(FiniteDifferenceJacobian {
            values,
            exhausted_budget: false,
        });
    }
    if let Some(jacobian) = evaluator.take_jacobian() {
        if jacobian.dimensions() != (m, n) {
            return Err(optim_error(
                name,
                "accepted Jacobian shape disagrees with residual",
            ));
        }
        let c = evaluator.cooperation().ok_or_else(|| {
            optim_error(
                name,
                "supplied matrix Jacobian requires original cooperation",
            )
        })?;
        return Ok(FiniteDifferenceJacobian {
            values: jacobian.into_row_major(c).await?,
            exhausted_budget: false,
        });
    }
    if let Some(jacobian) = evaluator.analytic_jacobian() {
        if jacobian.len() != checked_jacobian_work(name, m, n)? {
            return Err(optim_error(
                name,
                "supplied Jacobian dimensions do not match residual and variables",
            ));
        }
        return Ok(FiniteDifferenceJacobian {
            values: jacobian.to_vec(),
            exhausted_budget: false,
        });
    }
    let mut jacobian = vec![0.0; checked_jacobian_work(name, m, n)?];
    for col in 0..n {
        checkpoint(evaluator, m).await?;
        if bounds.lower[col] == bounds.upper[col] {
            continue;
        }
        if *func_count >= max_fun_evals {
            // Unvisited free-variable coefficients were never evaluated.
            // Fixed-bound columns remain the existing known-zero convention.
            for pending in col..n {
                if bounds.lower[pending] == bounds.upper[pending] {
                    continue;
                }
                for first in (0..m).step_by(4096) {
                    let end = m.min(first + 4096);
                    for row in first..end {
                        jacobian[row * n + pending] = f64::NAN;
                    }
                    checkpoint(evaluator, end - first).await?;
                }
            }
            return Ok(FiniteDifferenceJacobian {
                values: jacobian,
                exhausted_budget: true,
            });
        }
        let mut perturbed = x.to_vec();
        let Some(points) = scaling.points(
            name,
            x[col],
            col,
            bounds.lower[col],
            bounds.upper[col],
            false,
        )?
        else {
            continue;
        };
        perturbed[col] = points.first;
        let actual_step = perturbed[col] - x[col];
        if actual_step == 0.0 || (bounds.lower[col] == bounds.upper[col]) {
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
    if let Some(source) = evaluator.take_jacobian() {
        return finish(
            evaluator,
            bounds,
            options,
            x,
            residual,
            DerivativeResult::RetainedWithWorkspace(source, fallback_jacobian),
            exitflag,
            iterations,
            func_count,
            fallback_optimality,
            step_size,
            message,
        )
        .await;
    }
    if (!options.final_gradient && !options.final_jacobian)
        || (evaluator.analytic_jacobian().is_none()
            && func_count.saturating_add(x.len()) > options.max_fun_evals)
    {
        return result(
            evaluator,
            bounds,
            options,
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
        &options.scaling,
    )
    .await?;
    let jacobian = jacobian_eval.values;
    let optimality = first_order_optimality(&jacobian, &residual, x.len(), evaluator).await?;
    result(
        evaluator,
        bounds,
        options,
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

#[cfg(test)]
mod tests;
