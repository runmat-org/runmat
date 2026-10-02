//! A single adaptive error and evaluation budget for an initial interval mesh.
//! Waypoints refine this mesh; they are not independently accepted integrals.

use super::adaptive::QuadratureEvaluator;
use super::algebra::{AdaptiveEstimate, QuadratureSample};
use super::{QuadratureArithmeticError, ScaledEstimate};
use crate::BuiltinResult;

struct Interval<S: QuadratureSample> {
    a: f64,
    b: f64,
    fa: S,
    fd: S,
    fm: S,
    fe: S,
    fb: S,
    corrected: S::Estimate,
    error: S::Estimate,
    magnitude: ScaledEstimate,
    depth: usize,
}

fn allocation_error<E: QuadratureEvaluator>(
    evaluator: &E,
    source: std::collections::TryReserveError,
) -> crate::RuntimeError {
    evaluator.arithmetic_error(QuadratureArithmeticError::Resource(source))
}

async fn build_interval<E: QuadratureEvaluator>(
    evaluator: &mut E,
    a: f64,
    b: f64,
    samples: [E::Sample; 5],
    depth: usize,
) -> BuiltinResult<Interval<E::Sample>> {
    let [fa, fd, fm, fe, fb] = samples;
    let m = evaluator.midpoint(a, b);
    let whole = E::Sample::simpson(a, b, &fa, &fm, &fb, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let left = E::Sample::simpson(a, m, &fa, &fd, &fm, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let right = E::Sample::simpson(m, b, &fm, &fe, &fb, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let fine = left
        .add(&right, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let correction = fine
        .subtract(&whole, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?
        .multiply_f64(1.0 / 15.0, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let corrected = fine
        .add(&correction, evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let error = correction
        .absolute(evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    let magnitude = error
        .magnitude(evaluator)
        .await
        .map_err(|error| evaluator.arithmetic_error(error))?;
    Ok(Interval {
        a,
        b,
        fa,
        fd,
        fm,
        fe,
        fb,
        corrected,
        error,
        magnitude,
        depth,
    })
}

async fn totals<E: QuadratureEvaluator>(
    evaluator: &mut E,
    intervals: &[Interval<E::Sample>],
) -> BuiltinResult<(
    <E::Sample as QuadratureSample>::Estimate,
    <E::Sample as QuadratureSample>::Estimate,
)> {
    let first = intervals.first().expect("nonempty mesh");
    let mut result = first.corrected.clone();
    let mut error = first.error.clone();
    for (index, interval) in intervals[1..].iter().enumerate() {
        result = result
            .add(&interval.corrected, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        error = error
            .add(&interval.error, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        if (index + 1).is_multiple_of(4096) {
            evaluator.charge(4096).await?;
        }
    }
    evaluator
        .charge(((intervals.len() - 1) % 4096) as u64)
        .await?;
    Ok((result, error))
}

fn refinement_points<E: QuadratureEvaluator>(evaluator: &E, a: f64, b: f64) -> Option<[f64; 4]> {
    let m = evaluator.midpoint(a, b);
    let d = evaluator.midpoint(a, m);
    let e = evaluator.midpoint(m, b);
    let q1 = evaluator.midpoint(a, d);
    let q2 = evaluator.midpoint(d, m);
    let q3 = evaluator.midpoint(m, e);
    let q4 = evaluator.midpoint(e, b);
    (q1 > a && q2 < m && q3 > m && q4 < b).then_some([q1, q2, q3, q4])
}

async fn largest_refinable<E: QuadratureEvaluator>(
    evaluator: &mut E,
    intervals: &[Interval<E::Sample>],
) -> BuiltinResult<Option<(usize, [f64; 4])>> {
    let mut selected: Option<(usize, [f64; 4])> = None;
    for (index, interval) in intervals.iter().enumerate() {
        if interval.depth > 0 {
            if let Some(points) = refinement_points(evaluator, interval.a, interval.b) {
                let replace = selected.is_none_or(|(previous, _)| {
                    intervals[previous].magnitude.abs_le(interval.magnitude)
                });
                if replace {
                    selected = Some((index, points));
                }
            }
        }
        if (index + 1).is_multiple_of(4096) {
            evaluator.charge(4096).await?;
        }
    }
    evaluator.charge((intervals.len() % 4096) as u64).await?;
    Ok(selected)
}

pub(crate) async fn integrate_mesh<E: QuadratureEvaluator>(
    evaluator: &mut E,
    segments: &[(f64, f64)],
) -> BuiltinResult<<E::Sample as QuadratureSample>::Estimate> {
    evaluator.checkpoint().await?;
    let required = segments
        .len()
        .checked_mul(5)
        .ok_or_else(|| evaluator.evaluation_limit_error())?;
    if required > evaluator.max_evaluations() {
        return Err(evaluator.evaluation_limit_error());
    }
    let mut intervals = Vec::new();
    intervals
        .try_reserve_exact(segments.len())
        .map_err(|source| allocation_error(evaluator, source))?;
    let mut evaluations = 0usize;
    for &(a, b) in segments {
        evaluator.checkpoint().await?;
        let m = evaluator.midpoint(a, b);
        let d = evaluator.midpoint(a, m);
        let e = evaluator.midpoint(m, b);
        let samples = evaluator.evaluate_many([a, d, m, e, b]).await?;
        evaluations += 5;
        let depth = evaluator.max_depth();
        intervals.push(build_interval(evaluator, a, b, samples, depth).await?);
    }
    let (mut result, mut error) = totals(evaluator, &intervals).await?;
    loop {
        evaluator.checkpoint().await?;
        let (absolute, relative) = evaluator.tolerances();
        let allowed = result
            .tolerance(absolute, relative, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        if error
            .abs_le(&allowed, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?
        {
            // Re-sum the current partition before publication: incremental
            // subtract/add updates must not hide cancellation across segments.
            let (verified_result, verified_error) = totals(evaluator, &intervals).await?;
            let verified_allowed = verified_result
                .tolerance(absolute, relative, evaluator)
                .await
                .map_err(|cause| evaluator.arithmetic_error(cause))?;
            if verified_error
                .abs_le(&verified_allowed, evaluator)
                .await
                .map_err(|cause| evaluator.arithmetic_error(cause))?
            {
                return Ok(verified_result);
            }
            result = verified_result;
            error = verified_error;
        }
        let Some((index, [q1, q2, q3, q4])) = largest_refinable(evaluator, &intervals).await?
        else {
            return Err(evaluator.convergence_error());
        };
        if evaluations
            .checked_add(4)
            .is_none_or(|next| next > evaluator.max_evaluations())
        {
            return Err(evaluator.evaluation_limit_error());
        }
        intervals
            .try_reserve(1)
            .map_err(|source| allocation_error(evaluator, source))?;
        let selected = intervals.swap_remove(index);
        let m = evaluator.midpoint(selected.a, selected.b);
        let [f1, f2, f3, f4] = evaluator.evaluate_many([q1, q2, q3, q4]).await?;
        evaluations += 4;
        let left = build_interval(
            evaluator,
            selected.a,
            m,
            [selected.fa, f1, selected.fd, f2, selected.fm.clone()],
            selected.depth - 1,
        )
        .await?;
        let right = build_interval(
            evaluator,
            m,
            selected.b,
            [selected.fm, f3, selected.fe, f4, selected.fb],
            selected.depth - 1,
        )
        .await?;
        result = result
            .subtract(&selected.corrected, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?
            .add(&left.corrected, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?
            .add(&right.corrected, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        error = error
            .subtract(&selected.error, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?
            .add(&left.error, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?
            .add(&right.error, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        intervals.push(left);
        intervals.push(right);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_runtime_error;
    use crate::RuntimeError;

    struct CancellingSine {
        max_evaluations: usize,
        calls: usize,
    }

    impl QuadratureEvaluator for CancellingSine {
        type Sample = f64;

        async fn checkpoint(&mut self) -> BuiltinResult<()> {
            Ok(())
        }

        async fn evaluate(&mut self, x: f64) -> BuiltinResult<f64> {
            self.calls += 1;
            Ok(1.0e5 * (2.0 * std::f64::consts::PI * x).sin())
        }

        fn midpoint(&self, a: f64, b: f64) -> f64 {
            super::super::midpoint(a, b)
        }

        fn tolerances(&self) -> (f64, f64) {
            (1.0e-3, 1.0)
        }

        fn max_evaluations(&self) -> usize {
            self.max_evaluations
        }

        fn max_depth(&self) -> usize {
            30
        }

        fn evaluation_limit_error(&self) -> RuntimeError {
            build_runtime_error("global mesh evaluation limit").build()
        }

        fn convergence_error(&self) -> RuntimeError {
            build_runtime_error("global mesh did not converge").build()
        }

        fn nonrepresentable_error(&self) -> RuntimeError {
            build_runtime_error("global mesh estimate is nonrepresentable").build()
        }

        fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError {
            build_runtime_error(format!("global mesh arithmetic: {error:?}")).build()
        }
    }

    #[test]
    fn cancellation_between_waypoint_segments_uses_global_error_not_local_relative_error() {
        let segments = [(0.0, 0.5), (0.5, 1.0)];
        let mut insufficient = CancellingSine {
            max_evaluations: 10,
            calls: 0,
        };
        let error = futures::executor::block_on(integrate_mesh(&mut insufficient, &segments))
            .expect_err("large cancelling segments cannot pass on relative error alone");
        assert!(error.message().contains("evaluation limit"));
        assert_eq!(insufficient.calls, 10);

        let mut enough = CancellingSine {
            max_evaluations: 2000,
            calls: 0,
        };
        let value = futures::executor::block_on(integrate_mesh(&mut enough, &segments))
            .expect("global mesh should refine cancelling intervals")
            .to_f64()
            .expect("finite result");
        assert!(value.abs() < 1.0e-4);
        assert!(enough.calls > 10);
    }

    #[test]
    fn exhausted_largest_interval_does_not_hide_another_refinable_interval() {
        let interval = |a, b, depth, error: f64| Interval {
            a,
            b,
            fa: 0.0,
            fd: 0.0,
            fm: 0.0,
            fe: 0.0,
            fb: 0.0,
            corrected: ScaledEstimate::from_f64(0.0),
            error: ScaledEstimate::from_f64(error),
            magnitude: ScaledEstimate::from_f64(error),
            depth,
        };
        let intervals = [interval(0.0, 0.5, 0, 10.0), interval(0.5, 1.0, 1, 1.0)];
        let mut evaluator = CancellingSine {
            max_evaluations: 100,
            calls: 0,
        };
        let selected = futures::executor::block_on(largest_refinable(&mut evaluator, &intervals))
            .expect("selection is cooperative")
            .expect("the second interval can still be refined");
        assert_eq!(selected.0, 1);
    }
}
