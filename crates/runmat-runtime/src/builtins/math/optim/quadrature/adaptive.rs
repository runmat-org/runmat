//! Shared adaptive Simpson traversal. Public form and callback admission stay
//! at the owning `integral` or `quad` entrypoint.

use crate::{BuiltinResult, RuntimeError};
#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use super::algebra::ArrayEstimate;
use super::algebra::{AdaptiveEstimate, QuadratureSample};
use super::open_algebra::OpenSample;
use super::open_rule::OpenDomain;
use super::QuadratureArithmeticError;
use super::ScaledEstimate;

pub(crate) trait QuadratureEvaluator {
    type Sample: OpenSample;

    async fn checkpoint(&mut self) -> BuiltinResult<()>;
    async fn charge(&mut self, _units: u64) -> BuiltinResult<()> {
        self.checkpoint().await
    }
    async fn evaluate(&mut self, x: f64) -> BuiltinResult<Self::Sample>;
    async fn evaluate_many<const N: usize>(
        &mut self,
        points: [f64; N],
    ) -> BuiltinResult<[Self::Sample; N]> {
        let mut values: [Option<Self::Sample>; N] = std::array::from_fn(|_| None);
        for (index, point) in points.into_iter().enumerate() {
            values[index] = Some(self.evaluate(point).await?);
        }
        Ok(values.map(|value| value.expect("every evaluated sample fills one slot")))
    }
    async fn evaluate_open_many<const N: usize>(
        &mut self,
        _domain: OpenDomain,
        points: [f64; N],
    ) -> BuiltinResult<[Self::Sample; N]> {
        self.evaluate_many(points).await
    }
    fn midpoint(&self, a: f64, b: f64) -> f64;
    fn tolerances(&self) -> (f64, f64);
    fn max_evaluations(&self) -> usize;
    fn max_intervals(&self) -> Option<usize> {
        None
    }
    fn interval_limit_error(&self) -> RuntimeError {
        self.convergence_error()
    }
    fn max_depth(&self) -> usize;
    fn evaluation_limit_error(&self) -> RuntimeError;
    fn convergence_error(&self) -> RuntimeError;
    fn nonrepresentable_error(&self) -> RuntimeError;
    fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError;
    fn refined(
        &mut self,
        _evals: usize,
        _a: f64,
        _b: f64,
        _estimate: &<Self::Sample as OpenSample>::Result,
        _error: &<Self::Sample as OpenSample>::Error,
    ) -> BuiltinResult<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct AdaptiveSimpsonResult {
    pub(crate) q: f64,
    pub(crate) evaluations: usize,
}

struct Interval<S: QuadratureSample> {
    a: f64,
    b: f64,
    fa: S,
    fm: S,
    fb: S,
    whole: S::Estimate,
    tol: S::Estimate,
    depth: usize,
}

pub(crate) async fn integrate<E: QuadratureEvaluator<Sample = f64>>(
    evaluator: &mut E,
    a: f64,
    b: f64,
) -> BuiltinResult<AdaptiveSimpsonResult> {
    let (q, evaluations) = integrate_scaled(evaluator, a, b).await?;
    let q = q
        .to_f64()
        .ok_or_else(|| evaluator.nonrepresentable_error())?;
    Ok(AdaptiveSimpsonResult { q, evaluations })
}

/// Retain the scaled estimate across adjacent path segments. Publishing each
/// segment as `f64` can overflow even when their combined result is finite.
pub(crate) async fn integrate_scaled<E: QuadratureEvaluator<Sample = f64>>(
    evaluator: &mut E,
    a: f64,
    b: f64,
) -> BuiltinResult<(ScaledEstimate, usize)> {
    integrate_domain(evaluator, a, b).await
}

#[cfg(test)]
pub(crate) async fn integrate_array<E: QuadratureEvaluator<Sample = Arc<Vec<f64>>>>(
    evaluator: &mut E,
    a: f64,
    b: f64,
) -> BuiltinResult<ArrayEstimate> {
    integrate_domain(evaluator, a, b).await.map(|(q, _)| q)
}

async fn integrate_domain<E: QuadratureEvaluator>(
    evaluator: &mut E,
    a: f64,
    b: f64,
) -> BuiltinResult<(<E::Sample as QuadratureSample>::Estimate, usize)>
where
    E::Sample: QuadratureSample
        + OpenSample<
            Result = <E::Sample as QuadratureSample>::Estimate,
            Error = <E::Sample as QuadratureSample>::Estimate,
        >,
{
    evaluator.checkpoint().await?;
    let m = evaluator.midpoint(a, b);
    let [fa, fm, fb] = evaluator.evaluate_many([a, m, b]).await?;
    let whole = E::Sample::simpson(a, b, &fa, &fm, &fb, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let mut evaluations = 3_usize;
    let (absolute_tolerance, relative_tolerance) = evaluator.tolerances();
    let tol = whole
        .tolerance(absolute_tolerance, relative_tolerance, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let depth = evaluator.max_depth();
    let mut intervals = 1usize;
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
        &mut intervals,
    )
    .await?;
    Ok((q, evaluations))
}

async fn adapt<E: QuadratureEvaluator>(
    evaluator: &mut E,
    state: Interval<E::Sample>,
    evaluations: &mut usize,
    intervals: &mut usize,
) -> BuiltinResult<<E::Sample as QuadratureSample>::Estimate>
where
    E::Sample: QuadratureSample
        + OpenSample<
            Result = <E::Sample as QuadratureSample>::Estimate,
            Error = <E::Sample as QuadratureSample>::Estimate,
        >,
{
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

    let left = E::Sample::simpson(state.a, c, &state.fa, &fd, &state.fm, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let right = E::Sample::simpson(c, state.b, &state.fm, &fe, &state.fb, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let refined = left
        .add(&right, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let error = refined
        .subtract(&state.whole, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    evaluator.refined(*evaluations, state.a, state.b, &refined, &error)?;
    let accepted_error = state
        .tol
        .multiply_f64(15.0, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    if error
        .abs_le(&accepted_error, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?
    {
        let correction = error
            .multiply_f64(1.0 / 15.0, evaluator)
            .await
            .map_err(|error| evaluator.arithmetic_error(error))?;
        return refined
            .add(&correction, evaluator)
            .await
            .map_err(|error| evaluator.arithmetic_error(error));
    }
    if state.depth == 0 {
        return Err(evaluator.convergence_error());
    }
    if evaluator
        .max_intervals()
        .is_some_and(|limit| *intervals >= limit)
    {
        return Err(evaluator.interval_limit_error());
    }
    *intervals += 1;

    let child_tolerance = state
        .tol
        .multiply_f64(0.5, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let left_value = Box::pin(adapt(
        evaluator,
        Interval {
            a: state.a,
            b: c,
            fa: state.fa,
            fm: fd,
            fb: state.fm.clone(),
            whole: left,
            tol: child_tolerance.clone(),
            depth: state.depth - 1,
        },
        evaluations,
        intervals,
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
            tol: child_tolerance,
            depth: state.depth - 1,
        },
        evaluations,
        intervals,
    ))
    .await?;
    left_value
        .add(&right_value, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))
}

#[cfg(test)]
#[path = "adaptive/array_tests.rs"]
mod array_tests;

#[cfg(test)]
mod tests {
    use super::*;

    struct Polynomial {
        evaluations: usize,
        trace_rows: usize,
        limit: usize,
    }

    impl QuadratureEvaluator for Polynomial {
        type Sample = f64;

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

        fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError {
            crate::build_runtime_error(format!("estimate error: {error:?}")).build()
        }

        fn refined(
            &mut self,
            _evals: usize,
            _a: f64,
            _b: f64,
            _estimate: &ScaledEstimate,
            _error: &ScaledEstimate,
        ) -> BuiltinResult<()> {
            self.trace_rows += 1;
            Ok(())
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
