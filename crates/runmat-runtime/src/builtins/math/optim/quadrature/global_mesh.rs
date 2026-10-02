//! A single adaptive error and evaluation budget for an initial interval mesh.
//! Waypoints refine this mesh; they are not independently accepted integrals.

use super::adaptive::QuadratureEvaluator;
use super::algebra::{AdaptiveEstimate, QuadratureSample};
use super::ScaledEstimate;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

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

fn allocation_error(source: std::collections::TryReserveError) -> crate::RuntimeError {
    build_runtime_error("integral: could not reserve adaptive waypoint intervals")
        .with_identifier("RunMat:Internal")
        .with_builtin("integral")
        .with_source(source)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
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
    for interval in &intervals[1..] {
        evaluator.checkpoint().await?;
        result = result
            .add(&interval.corrected, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        error = error
            .add(&interval.error, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
    }
    Ok((result, error))
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
        .map_err(allocation_error)?;
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
        if evaluations
            .checked_add(4)
            .is_none_or(|next| next > evaluator.max_evaluations())
        {
            return Err(evaluator.evaluation_limit_error());
        }
        let index = intervals
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                if left.magnitude.abs_le(right.magnitude) && right.magnitude.abs_le(left.magnitude)
                {
                    std::cmp::Ordering::Equal
                } else if left.magnitude.abs_le(right.magnitude) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .map(|(index, _)| index)
            .expect("nonempty mesh");
        let selected = &intervals[index];
        if selected.depth == 0 {
            return Err(evaluator.convergence_error());
        }
        intervals.try_reserve(1).map_err(allocation_error)?;
        let selected = intervals.swap_remove(index);
        let m = evaluator.midpoint(selected.a, selected.b);
        let q1 = evaluator.midpoint(selected.a, evaluator.midpoint(selected.a, m));
        let q2 = evaluator.midpoint(evaluator.midpoint(selected.a, m), m);
        let q3 = evaluator.midpoint(m, evaluator.midpoint(m, selected.b));
        let q4 = evaluator.midpoint(evaluator.midpoint(m, selected.b), selected.b);
        if q1 <= selected.a || q2 >= m || q3 <= m || q4 >= selected.b {
            return Err(evaluator.convergence_error());
        }
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
    use crate::builtins::math::optim::quadrature::QuadratureArithmeticError;
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
}
