//! Shared adaptive Simpson traversal. Public form and callback admission stay
//! at the owning `integral` or `quad` entrypoint.

use crate::{BuiltinResult, RuntimeError};

use super::{simpson, ScaledEstimate};

pub(crate) trait QuadratureEvaluator {
    async fn checkpoint(&mut self) -> BuiltinResult<()>;
    async fn evaluate(&mut self, x: f64) -> BuiltinResult<f64>;
    async fn evaluate_many<const N: usize>(&mut self, points: [f64; N]) -> BuiltinResult<[f64; N]> {
        let mut values = [0.0; N];
        for (index, point) in points.into_iter().enumerate() {
            values[index] = self.evaluate(point).await?;
        }
        Ok(values)
    }
    fn midpoint(&self, a: f64, b: f64) -> f64;
    fn tolerances(&self) -> (f64, f64);
    fn max_evaluations(&self) -> usize;
    fn max_depth(&self) -> usize;
    fn evaluation_limit_error(&self) -> RuntimeError;
    fn convergence_error(&self) -> RuntimeError;
    fn nonrepresentable_error(&self) -> RuntimeError;
    fn refined(&mut self, _evals: usize, _a: f64, _b: f64, _estimate: f64, _error: f64) {}
}

#[derive(Debug)]
pub(crate) struct AdaptiveSimpsonResult {
    pub(crate) q: f64,
    pub(crate) evaluations: usize,
}

#[derive(Clone, Copy)]
struct Interval {
    a: f64,
    b: f64,
    fa: f64,
    fm: f64,
    fb: f64,
    whole: ScaledEstimate,
    tol: ScaledEstimate,
    depth: usize,
}

pub(crate) async fn integrate<E: QuadratureEvaluator>(
    evaluator: &mut E,
    a: f64,
    b: f64,
) -> BuiltinResult<AdaptiveSimpsonResult> {
    evaluator.checkpoint().await?;
    let m = evaluator.midpoint(a, b);
    let [fa, fm, fb] = evaluator.evaluate_many([a, m, b]).await?;
    let whole = simpson(a, b, fa, fm, fb);
    let mut evaluations = 3_usize;
    let (absolute_tolerance, relative_tolerance) = evaluator.tolerances();
    let tol = ScaledEstimate::from_f64(absolute_tolerance)
        .max(whole.abs().multiply_f64(relative_tolerance));
    let depth = evaluator.max_depth();
    let q = adapt(
        evaluator,
        Interval {
            a,
            b,
            fa,
            fm,
            fb,
            whole,
            tol,
            depth,
        },
        &mut evaluations,
    )
    .await?;
    let q = q
        .to_f64()
        .ok_or_else(|| evaluator.nonrepresentable_error())?;
    Ok(AdaptiveSimpsonResult { q, evaluations })
}

async fn adapt<E: QuadratureEvaluator>(
    evaluator: &mut E,
    state: Interval,
    evaluations: &mut usize,
) -> BuiltinResult<ScaledEstimate> {
    evaluator.checkpoint().await?;
    if (*evaluations)
        .checked_add(2)
        .is_none_or(|next| next > evaluator.max_evaluations())
    {
        return Err(evaluator.evaluation_limit_error());
    }

    let c = evaluator.midpoint(state.a, state.b);
    let d = evaluator.midpoint(state.a, c);
    let e = evaluator.midpoint(c, state.b);
    let [fd, fe] = evaluator.evaluate_many([d, e]).await?;
    *evaluations += 2;

    let left = simpson(state.a, c, state.fa, fd, state.fm);
    let right = simpson(c, state.b, state.fm, fe, state.fb);
    let refined = left.add(right);
    let error = refined.subtract(state.whole);
    evaluator.refined(
        *evaluations,
        state.a,
        state.b,
        refined.trace_f64(),
        error.trace_f64(),
    );
    if error.abs_le(state.tol.multiply_f64(15.0)) {
        return Ok(refined.add(error.multiply_f64(1.0 / 15.0)));
    }
    if state.depth == 0 {
        return Err(evaluator.convergence_error());
    }

    let left_value = Box::pin(adapt(
        evaluator,
        Interval {
            a: state.a,
            b: c,
            fa: state.fa,
            fm: fd,
            fb: state.fm,
            whole: left,
            tol: state.tol.multiply_f64(0.5),
            depth: state.depth - 1,
        },
        evaluations,
    ))
    .await?;
    let right_value = Box::pin(adapt(
        evaluator,
        Interval {
            a: c,
            b: state.b,
            fa: state.fm,
            fm: fe,
            fb: state.fb,
            whole: right,
            tol: state.tol.multiply_f64(0.5),
            depth: state.depth - 1,
        },
        evaluations,
    ))
    .await?;
    Ok(left_value.add(right_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Polynomial {
        evaluations: usize,
        trace_rows: usize,
        limit: usize,
    }

    impl QuadratureEvaluator for Polynomial {
        async fn checkpoint(&mut self) -> BuiltinResult<()> {
            Ok(())
        }

        async fn evaluate(&mut self, x: f64) -> BuiltinResult<f64> {
            self.evaluations += 1;
            Ok(x * x)
        }

        fn midpoint(&self, a: f64, b: f64) -> f64 {
            super::super::midpoint(a, b)
        }

        fn tolerances(&self) -> (f64, f64) {
            (1.0e-6, 0.0)
        }

        fn max_evaluations(&self) -> usize {
            self.limit
        }

        fn max_depth(&self) -> usize {
            30
        }

        fn evaluation_limit_error(&self) -> RuntimeError {
            crate::build_runtime_error("evaluation limit").build()
        }

        fn convergence_error(&self) -> RuntimeError {
            crate::build_runtime_error("did not converge").build()
        }

        fn nonrepresentable_error(&self) -> RuntimeError {
            crate::build_runtime_error("nonrepresentable estimate").build()
        }

        fn refined(&mut self, _evals: usize, _a: f64, _b: f64, _estimate: f64, _error: f64) {
            self.trace_rows += 1;
        }
    }

    #[test]
    fn scalar_polynomial_oracle_and_refinement_count() {
        let mut evaluator = Polynomial {
            evaluations: 0,
            trace_rows: 0,
            limit: 5,
        };
        let result = futures::executor::block_on(integrate(&mut evaluator, 0.0, 1.0)).unwrap();
        assert!((result.q - 1.0 / 3.0).abs() < 1.0e-15);
        assert_eq!(result.evaluations, 5);
        assert_eq!(evaluator.evaluations, 5);
        assert_eq!(evaluator.trace_rows, 1);
    }

    #[test]
    fn evaluation_limit_fails_before_additional_callback_effects() {
        let mut evaluator = Polynomial {
            evaluations: 0,
            trace_rows: 0,
            limit: 4,
        };
        let error = futures::executor::block_on(integrate(&mut evaluator, 0.0, 1.0)).unwrap_err();
        assert_eq!(error.message(), "evaluation limit");
        assert_eq!(evaluator.evaluations, 3);
        assert_eq!(evaluator.trace_rows, 0);
    }
}
