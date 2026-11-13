//! Brent's method primitives shared across optimization builtins.
//!
//! Two related Brent algorithms live here:
//!
//! * [`brent_zero`] — root-finding by inverse-quadratic / secant / bisection.
//!   Powers [`crate::builtins::math::optim::fzero`].
//! * [`brent_min`] — bounded scalar minimization using golden-section search
//!   plus parabolic interpolation.  Powers [`crate::builtins::math::optim::fminbnd`].
//!
//! Both routines reuse [`call_scalar_function`] from the optim `common` module
//! so RunMat's function-handle dispatch path (closures, anonymous functions,
//! named handles) flows through a single helper.

use runmat_builtins::OptimSolver;
use runmat_value::Value;

use crate::builtins::math::optim::common::{
    call_scalar_function, call_scalar_function_with_precision, optim_error, ScalarCooperation,
};
use crate::BuiltinResult;
mod zero;
pub(crate) use zero::{zero_with_receipts, ZeroFunction, ZeroResult};

/// Result of a successful (or terminated) bounded scalar minimization.
#[derive(Debug, Clone)]
pub(crate) struct BrentMinResult {
    pub x: f64,
    pub fval: f64,
    pub iterations: usize,
    pub func_count: usize,
    pub converged: bool,
}

/// Result of a scalar root-finding run.
#[derive(Debug, Clone)]
pub(crate) struct BrentZeroResult {
    pub x: f64,
    pub fval: f64,
    pub iterations: usize,
    pub func_count: usize,
    pub converged: bool,
}

/// Tuning parameters shared by both Brent variants.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BrentParams {
    pub tol_x: f64,
    pub max_iter: usize,
    pub max_fun_evals: usize,
}

/// Per-iteration hook used for bounded minimization `Display = 'iter'` output.
pub(crate) trait BrentMinObserver {
    fn on_iteration(
        &mut self,
        iter: usize,
        func_count: usize,
        x: f64,
        fx: f64,
        step_kind: BrentStepKind,
    ) -> crate::BuiltinResult<()>;
}

/// Per-iteration hook used for scalar root-finding `Display = 'iter'` output.
pub(crate) trait BrentZeroObserver {
    fn on_iteration(
        &mut self,
        iter: usize,
        func_count: usize,
        x: f64,
        fx: f64,
        step_kind: BrentZeroStepKind,
    ) -> crate::BuiltinResult<()>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BrentStepKind {
    Initial,
    GoldenSection,
    Parabolic,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BrentZeroStepKind {
    Initial,
    Bisection,
    Interpolation,
}

/// Find a zero of a scalar function on a sign-changing bracket using Brent's method.
///
/// `bracket` must have opposite-sign endpoint values unless an endpoint is
/// already zero (in which case that endpoint is returned). The evaluation counter
/// tracks calls performed by the caller while constructing the bracket.
pub(crate) fn brent_zero<'a>(
    name: &'a str,
    cooperation: &'a mut ScalarCooperation,
    function: &'a Value,
    bracket: BrentZeroBracket,
    params: BrentParams,
    single: bool,
    observer: Option<&'a mut dyn BrentZeroObserver>,
) -> impl std::future::Future<Output = BuiltinResult<BrentZeroResult>> + 'a {
    use futures::FutureExt;
    zero_with_receipts(
        name,
        cooperation,
        RawZero {
            name,
            function,
            single,
        },
        bracket,
        ((), ()),
        params,
        observer,
    )
    .map(|result| result.map(|value| value.result))
}
struct RawZero<'a> {
    name: &'a str,
    function: &'a Value,
    single: bool,
}
impl ZeroFunction for RawZero<'_> {
    type Receipt = ();
    async fn evaluate(
        &mut self,
        x: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<(f64, ())> {
        let value = call_scalar_function_with_precision(
            self.name,
            OptimSolver::Fzero,
            self.function,
            x,
            self.single,
            cooperation,
        )
        .await?;
        Ok((value, ()))
    }
}

/// Bracket consumed by [`brent_zero`].
#[derive(Clone, Copy)]
pub(crate) struct BrentZeroBracket {
    pub a: f64,
    pub b: f64,
    pub fa: f64,
    pub fb: f64,
    pub evals: usize,
}

pub(crate) fn interpolation_step_accepted(p: f64, q: f64, midpoint: f64, tol: f64, e: f64) -> bool {
    let min_a = 3.0 * midpoint * q - (tol * q).abs();
    let min_b = (e * q).abs();
    2.0 * p < min_a.min(min_b)
}

/// Inverse golden ratio used by Brent minimization: `(3 - sqrt(5)) / 2`.
const CGOLD: f64 = 0.381_966_011_250_105_15;

fn half_difference(a: f64, b: f64) -> f64 {
    let difference = a - b;
    if difference.is_finite() {
        0.5 * difference
    } else {
        0.5 * a - 0.5 * b
    }
}

fn interval_point(a: f64, b: f64, fraction: f64) -> f64 {
    let difference = b - a;
    if difference.is_finite() {
        a + fraction * difference
    } else {
        (1.0 - fraction).mul_add(a, fraction * b)
    }
}

/// Bounded scalar minimization following Brent's method (Numerical Recipes §10.2).
///
/// The function `function` is evaluated through the standard scalar dispatcher.
/// `a` and `b` must be finite and may be supplied in any order.  The optional
/// `observer` receives a callback for each accepted iteration (used to drive
/// `Display = 'iter'`).
pub(crate) async fn brent_min(
    name: &str,
    cooperation: &mut ScalarCooperation,
    function: &Value,
    a: f64,
    b: f64,
    params: BrentParams,
    mut observer: Option<&mut dyn BrentMinObserver>,
) -> BuiltinResult<BrentMinResult> {
    cooperation.completed(0).await?;
    if !a.is_finite() || !b.is_finite() {
        return Err(optim_error(
            name,
            format!("{name}: bounds must be finite real scalars"),
        ));
    }
    let (mut a, mut b) = (a.min(b), a.max(b));
    let width = b - a;
    let scale = a.abs().max(b.abs()).max(1.0);
    if width.abs() <= f64::EPSILON * scale {
        let x = a + width * 0.5;
        let fx = call_scalar_function(name, OptimSolver::Fminbnd, function, x, cooperation).await?;
        cooperation.completed(0).await?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.on_iteration(0, 1, x, fx, BrentStepKind::Initial)?;
        }
        return Ok(BrentMinResult {
            x,
            fval: fx,
            iterations: 0,
            func_count: 1,
            converged: true,
        });
    }

    let mut x = interval_point(a, b, CGOLD);
    let mut w = x;
    let mut v = x;
    let mut fx = call_scalar_function(name, OptimSolver::Fminbnd, function, x, cooperation).await?;
    cooperation.completed(0).await?;
    let mut fw = fx;
    let mut fv = fx;
    let mut func_count = 1usize;
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;

    if let Some(observer) = observer.as_deref_mut() {
        observer.on_iteration(0, func_count, x, fx, BrentStepKind::Initial)?;
    }

    for iter in 1..=params.max_iter {
        cooperation.completed(1).await?;
        let xm = interval_point(a, b, 0.5);
        let tol1 = brent_min_tolerance(x, params);
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return Ok(BrentMinResult {
                x,
                fval: fx,
                iterations: iter - 1,
                func_count,
                converged: true,
            });
        }

        let mut step_kind = BrentStepKind::GoldenSection;
        let mut use_parabolic = false;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            let proposed = p / q;
            if proposed.is_finite()
                && etemp.is_finite()
                && proposed.abs() < 0.5 * etemp.abs()
                && proposed > a - x
                && proposed < b - x
            {
                d = proposed;
                let u = x + d;
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = with_sign(tol1, xm - x);
                }
                use_parabolic = true;
                step_kind = BrentStepKind::Parabolic;
            }
        }
        if !use_parabolic {
            let edge = if x >= xm { a } else { b };
            e = edge - x;
            d = interval_point(x, edge, CGOLD) - x;
        }

        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + with_sign(tol1, d)
        };
        if func_count >= params.max_fun_evals {
            return Ok(BrentMinResult {
                x,
                fval: fx,
                iterations: iter - 1,
                func_count,
                converged: false,
            });
        }
        let fu = call_scalar_function(name, OptimSolver::Fminbnd, function, u, cooperation).await?;
        cooperation.completed(0).await?;
        func_count += 1;

        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }

        if let Some(observer) = observer.as_deref_mut() {
            observer.on_iteration(iter, func_count, x, fx, step_kind)?;
        }
    }

    Ok(BrentMinResult {
        x,
        fval: fx,
        iterations: params.max_iter,
        func_count,
        converged: false,
    })
}

fn with_sign(magnitude: f64, sign_source: f64) -> f64 {
    if sign_source >= 0.0 {
        magnitude.abs()
    } else {
        -magnitude.abs()
    }
}

pub(crate) fn brent_min_tolerance(x: f64, params: BrentParams) -> f64 {
    params.tol_x + (3.0 * f64::EPSILON.sqrt()) * x.abs()
}

#[cfg(test)]
mod tests;
