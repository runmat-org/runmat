//! Shared scalar initial-point expansion; every active sample keeps its receipt.
use super::*;
use crate::builtins::math::optim::brent::ZeroFunction;
pub(super) struct SampledBracket<R> {
    pub numeric: BrentZeroBracket,
    pub receipts: (R, R),
}
pub(super) async fn expand<F: ZeroFunction>(
    function: &mut F,
    x0: f64,
    max_fun_evals: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<SampledBracket<F::Receipt>> {
    cooperation.completed(0).await?;
    if !x0.is_finite() {
        return Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            "initial point must be finite",
        ));
    }
    let (f0, r0) = function.evaluate(x0, cooperation).await?;
    cooperation.completed(1).await?;
    if f0 == 0.0 {
        return Ok(SampledBracket {
            numeric: BrentZeroBracket {
                a: x0,
                b: x0,
                fa: f0,
                fb: f0,
                evals: 1,
            },
            receipts: (r0.clone(), r0),
        });
    }

    let mut evals = 1usize;
    let mut step = (x0.abs() * 0.01).max(0.01);
    let mut previous_interval = None;
    while evals + 2 <= max_fun_evals {
        cooperation.completed(1).await?;
        // A finite initial point may be only one representable step from a
        // numerical boundary. Search through the finite domain, never pass
        // an overflowed abscissa to user code, and stop once expansion can
        // no longer reach a new pair of endpoints.
        let a = (x0 - step).max(-f64::MAX);
        let b = (x0 + step).min(f64::MAX);
        if previous_interval == Some((a, b)) {
            break;
        }
        previous_interval = Some((a, b));
        let (fa, ra) = function.evaluate(a, cooperation).await?;
        cooperation.completed(1).await?;
        let (fb, rb) = function.evaluate(b, cooperation).await?;
        cooperation.completed(1).await?;
        evals += 2;
        if fa == 0.0 {
            return Ok(SampledBracket {
                numeric: BrentZeroBracket {
                    a,
                    b: a,
                    fa,
                    fb: fa,
                    evals,
                },
                receipts: (ra.clone(), ra),
            });
        }
        if fa.signum() != f0.signum() {
            return Ok(SampledBracket {
                numeric: BrentZeroBracket {
                    a,
                    b: x0,
                    fa,
                    fb: f0,
                    evals,
                },
                receipts: (ra, r0),
            });
        }
        if fb.signum() != f0.signum() {
            return Ok(SampledBracket {
                numeric: BrentZeroBracket {
                    a: x0,
                    b,
                    fa: f0,
                    fb,
                    evals,
                },
                receipts: (r0, rb),
            });
        }
        if fb == 0.0 || fa.signum() != fb.signum() {
            return Ok(SampledBracket {
                numeric: BrentZeroBracket {
                    a,
                    b,
                    fa,
                    fb,
                    evals,
                },
                receipts: (ra, rb),
            });
        }
        step *= 1.6;
    }

    Err(fzero_error_with_detail(
        &FZERO_ERROR_INVALID_INPUT,
        "could not find a sign-changing bracket around the initial point",
    ))
}
