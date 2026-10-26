//! The same Brent zero iteration carries each active sample's receipt.
use super::*;
pub(crate) trait ZeroFunction {
    type Receipt: Clone;
    async fn evaluate(
        &mut self,
        x: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<(f64, Self::Receipt)>;
}
pub(crate) struct ZeroResult<R> {
    pub result: BrentZeroResult,
    pub receipt: R,
}
pub(crate) async fn zero_with_receipts<F: ZeroFunction>(
    name: &str,
    cooperation: &mut ScalarCooperation,
    mut function: F,
    bracket: BrentZeroBracket,
    receipts: (F::Receipt, F::Receipt),
    params: BrentParams,
    mut observer: Option<&mut dyn BrentZeroObserver>,
) -> BuiltinResult<ZeroResult<F::Receipt>> {
    cooperation.completed(0).await?;
    if bracket.fa == 0.0 {
        drop(receipts.1);
        return Ok(ZeroResult {
            receipt: receipts.0,
            result: BrentZeroResult {
                x: bracket.a,
                fval: bracket.fa,
                iterations: 0,
                func_count: bracket.evals,
                converged: true,
            },
        });
    }
    if bracket.fb == 0.0 {
        drop(receipts.0);
        return Ok(ZeroResult {
            receipt: receipts.1,
            result: BrentZeroResult {
                x: bracket.b,
                fval: bracket.fb,
                iterations: 0,
                func_count: bracket.evals,
                converged: true,
            },
        });
    }
    if bracket.fa.signum() == bracket.fb.signum() {
        return Err(optim_error(
            name,
            format!("{name}: invalid bracket; endpoint function values must have opposite signs"),
        ));
    }

    let mut ra = receipts.0;
    let mut rb = receipts.1;
    let mut rc = ra.clone();
    let mut a = bracket.a;
    let mut b = bracket.b;
    let mut c = a;
    let mut fa = bracket.fa;
    let mut fb = bracket.fb;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    let mut evals = bracket.evals;

    if let Some(observer) = observer.as_deref_mut() {
        observer.on_iteration(0, evals, b, fb, BrentZeroStepKind::Initial)?;
    }

    for iter in 1..=params.max_iter {
        cooperation.completed(1).await?;
        if fb.signum() == fc.signum() {
            c = a;
            fc = fa;
            rc = ra.clone();
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            let old_b = b;
            let old_fb = fb;
            let old_rb = rb.clone();
            a = b;
            fa = fb;
            ra = rb.clone();
            b = c;
            fb = fc;
            rb = rc.clone();
            c = old_b;
            fc = old_fb;
            rc = old_rb;
        }

        let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * params.tol_x;
        let midpoint = half_difference(c, b);
        if midpoint.abs() <= tol || fb == 0.0 {
            drop(ra);
            drop(rc);
            return Ok(ZeroResult {
                receipt: rb,
                result: BrentZeroResult {
                    x: b,
                    fval: fb,
                    iterations: iter - 1,
                    func_count: evals,
                    converged: true,
                },
            });
        }
        if evals >= params.max_fun_evals {
            drop(ra);
            drop(rc);
            return Ok(ZeroResult {
                receipt: rb,
                result: BrentZeroResult {
                    x: b,
                    fval: fb,
                    iterations: iter - 1,
                    func_count: evals,
                    converged: false,
                },
            });
        }

        let mut step_kind = BrentZeroStepKind::Bisection;
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
                step_kind = BrentZeroStepKind::Interpolation;
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
        ra = rb.clone();
        b += if d.abs() > tol {
            d
        } else if midpoint >= 0.0 {
            tol
        } else {
            -tol
        };
        (fb, rb) = function.evaluate(b, cooperation).await?;
        cooperation.completed(0).await?;
        evals += 1;

        if let Some(observer) = observer.as_deref_mut() {
            observer.on_iteration(iter, evals, b, fb, step_kind)?;
        }
        if fb == 0.0 {
            drop(ra);
            drop(rc);
            return Ok(ZeroResult {
                receipt: rb,
                result: BrentZeroResult {
                    x: b,
                    fval: fb,
                    iterations: iter,
                    func_count: evals,
                    converged: true,
                },
            });
        }
    }

    drop(ra);
    drop(rc);
    Ok(ZeroResult {
        receipt: rb,
        result: BrentZeroResult {
            x: b,
            fval: fb,
            iterations: params.max_iter,
            func_count: evals,
            converged: false,
        },
    })
}
