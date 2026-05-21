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

use runmat_builtins::Value;

use crate::builtins::math::optim::common::{call_scalar_function, optim_error};
use crate::BuiltinResult;

/// Result of a successful (or terminated) bounded scalar minimization.
#[derive(Debug, Clone)]
pub(crate) struct BrentMinResult {
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

/// Per-iteration hook used for `Display = 'iter'` output.
pub(crate) trait BrentMinObserver {
    fn on_iteration(
        &mut self,
        iter: usize,
        func_count: usize,
        x: f64,
        fx: f64,
        step_kind: BrentStepKind,
    );
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BrentStepKind {
    Initial,
    GoldenSection,
    Parabolic,
}

/// Find a zero of a scalar function on a sign-changing bracket using Brent's method.
///
/// `bracket` must satisfy `fa * fb <= 0` (otherwise the contract is violated and
/// the function returns an error).  The function evaluation counter
/// (`initial_evals`) tracks calls performed by the caller while constructing the
/// bracket; the returned tuple's second element is the total invocations.
pub(crate) async fn brent_zero(
    name: &str,
    function: &Value,
    bracket: BrentZeroBracket,
    params: BrentParams,
) -> BuiltinResult<f64> {
    if bracket.fa == 0.0 || bracket.a == bracket.b {
        return Ok(bracket.a);
    }
    if bracket.fb == 0.0 {
        return Ok(bracket.b);
    }

    let mut a = bracket.a;
    let mut b = bracket.b;
    let mut c = a;
    let mut fa = bracket.fa;
    let mut fb = bracket.fb;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    let mut evals = bracket.evals;

    for _ in 0..params.max_iter {
        if fb.signum() == fc.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            let old_b = b;
            let old_fb = fb;
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = old_b;
            fc = old_fb;
        }

        let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * params.tol_x;
        let midpoint = 0.5 * (c - b);
        if midpoint.abs() <= tol || fb == 0.0 {
            return Ok(b);
        }
        if evals >= params.max_fun_evals {
            return Err(optim_error(
                name,
                format!("{name}: exceeded maximum function evaluations"),
            ));
        }

        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q) = if a == c {
                (2.0 * midpoint * s, 1.0 - s)
            } else {
                let q = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * midpoint * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            if interpolation_step_accepted(p, q, midpoint, tol, e) {
                e = d;
                d = p / q;
            } else {
                d = midpoint;
                e = d;
            }
        } else {
            d = midpoint;
            e = d;
        }

        a = b;
        fa = fb;
        b += if d.abs() > tol {
            d
        } else if midpoint >= 0.0 {
            tol
        } else {
            -tol
        };
        fb = call_scalar_function(name, function, b).await?;
        evals += 1;
    }

    Err(optim_error(
        name,
        format!("{name}: exceeded maximum iterations"),
    ))
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

/// Bounded scalar minimization following Brent's method (Numerical Recipes §10.2).
///
/// The function `function` is evaluated through the standard scalar dispatcher.
/// `a` and `b` must be finite and may be supplied in any order.  The optional
/// `observer` receives a callback for each accepted iteration (used to drive
/// `Display = 'iter'`).
pub(crate) async fn brent_min(
    name: &str,
    function: &Value,
    a: f64,
    b: f64,
    params: BrentParams,
    mut observer: Option<&mut dyn BrentMinObserver>,
) -> BuiltinResult<BrentMinResult> {
    if !a.is_finite() || !b.is_finite() {
        return Err(optim_error(
            name,
            format!("{name}: bounds must be finite real scalars"),
        ));
    }
    let (mut a, mut b) = (a.min(b), a.max(b));
    if (b - a).abs() <= f64::EPSILON * (a.abs() + b.abs()) {
        let x = 0.5 * (a + b);
        let fx = call_scalar_function(name, function, x).await?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.on_iteration(0, 1, x, fx, BrentStepKind::Initial);
        }
        return Ok(BrentMinResult {
            x,
            fval: fx,
            iterations: 0,
            func_count: 1,
            converged: true,
        });
    }

    let mut x = a + CGOLD * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = call_scalar_function(name, function, x).await?;
    let mut fw = fx;
    let mut fv = fx;
    let mut func_count = 1usize;
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;

    if let Some(observer) = observer.as_deref_mut() {
        observer.on_iteration(0, func_count, x, fx, BrentStepKind::Initial);
    }

    for iter in 1..=params.max_iter {
        let xm = 0.5 * (a + b);
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
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = with_sign(tol1, xm - x);
                }
                use_parabolic = true;
                step_kind = BrentStepKind::Parabolic;
            }
        }
        if !use_parabolic {
            e = if x >= xm { a - x } else { b - x };
            d = CGOLD * e;
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
        let fu = call_scalar_function(name, function, u).await?;
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
            observer.on_iteration(iter, func_count, x, fx, step_kind);
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
    params.tol_x + 3.0 * x.abs() * f64::EPSILON.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_acceptance_uses_signed_q() {
        assert!(!interpolation_step_accepted(1.0, -2.0, 1.0, 0.1, 10.0));
        assert!(interpolation_step_accepted(1.0, -2.0, -1.0, 0.1, 10.0));
    }

    #[test]
    fn with_sign_matches_fortran_semantics() {
        assert_eq!(with_sign(1.0, 0.5), 1.0);
        assert_eq!(with_sign(1.0, -0.5), -1.0);
        assert_eq!(with_sign(1.0, 0.0), 1.0);
        assert_eq!(with_sign(-1.0, -1.0), -1.0);
    }
}
