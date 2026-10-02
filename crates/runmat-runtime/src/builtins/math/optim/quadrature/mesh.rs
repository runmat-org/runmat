//! One global convergence, interval, evaluation, and cooperation authority for
//! adaptive rules. A numerical rule owns only its interval construction.

use super::adaptive::QuadratureEvaluator;
use super::algebra::AdaptiveEstimate;
use super::geometry::ScaledEstimate;
use super::open_algebra::{OpenResult, OpenSample};
use super::QuadratureArithmeticError;
use crate::BuiltinResult;

pub(crate) trait MeshRule<E: QuadratureEvaluator> {
    type Segment: Copy;
    type Interval;

    const INITIAL_EVALUATIONS: usize;
    const REFINEMENT_EVALUATIONS: usize;

    async fn initial(
        evaluator: &mut E,
        segment: Self::Segment,
        depth: usize,
    ) -> BuiltinResult<Self::Interval>;
    fn result(interval: &Self::Interval) -> &<E::Sample as OpenSample>::Result;
    fn error(interval: &Self::Interval) -> &<E::Sample as OpenSample>::Error;
    fn magnitude(interval: &Self::Interval) -> ScaledEstimate;
    fn can_refine(evaluator: &E, interval: &Self::Interval) -> bool;
    async fn refine(
        evaluator: &mut E,
        interval: Self::Interval,
    ) -> BuiltinResult<[Self::Interval; 2]>;
}

fn allocation_error<E: QuadratureEvaluator>(
    evaluator: &E,
    source: std::collections::TryReserveError,
) -> crate::RuntimeError {
    evaluator.arithmetic_error(QuadratureArithmeticError::Resource(source))
}

async fn totals<E: QuadratureEvaluator, R: MeshRule<E>>(
    evaluator: &mut E,
    intervals: &[R::Interval],
) -> BuiltinResult<(
    <E::Sample as OpenSample>::Result,
    <E::Sample as OpenSample>::Error,
)> {
    let first = intervals.first().expect("nonempty mesh");
    let mut result = (*R::result(first)).clone();
    let mut error = (*R::error(first)).clone();
    for (index, interval) in intervals[1..].iter().enumerate() {
        result = OpenResult::add(&result, R::result(interval), evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        error = AdaptiveEstimate::add(&error, R::error(interval), evaluator)
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

async fn largest_refinable<E: QuadratureEvaluator, R: MeshRule<E>>(
    evaluator: &mut E,
    intervals: &[R::Interval],
) -> BuiltinResult<Option<usize>> {
    let mut selected: Option<usize> = None;
    for (index, interval) in intervals.iter().enumerate() {
        if R::can_refine(evaluator, interval) {
            let replace = selected.is_none_or(|previous| {
                R::magnitude(&intervals[previous]).abs_le(R::magnitude(interval))
            });
            if replace {
                selected = Some(index);
            }
        }
        if (index + 1).is_multiple_of(4096) {
            evaluator.charge(4096).await?;
        }
    }
    evaluator.charge((intervals.len() % 4096) as u64).await?;
    Ok(selected)
}

pub(crate) async fn integrate_partition<E: QuadratureEvaluator, R: MeshRule<E>>(
    evaluator: &mut E,
    segments: &[R::Segment],
) -> BuiltinResult<<E::Sample as OpenSample>::Result> {
    evaluator.checkpoint().await?;
    if segments.is_empty() {
        return Err(evaluator.convergence_error());
    }
    let minimum = segments
        .len()
        .checked_mul(R::INITIAL_EVALUATIONS)
        .ok_or_else(|| evaluator.evaluation_limit_error())?;
    if minimum > evaluator.max_evaluations() {
        return Err(evaluator.evaluation_limit_error());
    }
    if evaluator
        .max_intervals()
        .is_some_and(|limit| segments.len() > limit)
    {
        return Err(evaluator.interval_limit_error());
    }
    let mut intervals = Vec::new();
    intervals
        .try_reserve_exact(segments.len())
        .map_err(|source| allocation_error(evaluator, source))?;
    for &segment in segments {
        evaluator.checkpoint().await?;
        let depth = evaluator.max_depth();
        intervals.push(R::initial(evaluator, segment, depth).await?);
    }
    let mut evaluations = minimum;
    let (mut result, mut error) = totals::<E, R>(evaluator, &intervals).await?;
    loop {
        evaluator.checkpoint().await?;
        let (absolute, relative) = evaluator.tolerances();
        let allowed = result
            .tolerance_error(absolute, relative, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        if error
            .abs_le(&allowed, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?
        {
            // Incremental subtract/add updates may lose cancellation. Re-sum
            // the actual partition before returning the public result.
            let (verified_result, verified_error) = totals::<E, R>(evaluator, &intervals).await?;
            let verified_allowed = verified_result
                .tolerance_error(absolute, relative, evaluator)
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
        let Some(index) = largest_refinable::<E, R>(evaluator, &intervals).await? else {
            return Err(evaluator.convergence_error());
        };
        if evaluations
            .checked_add(R::REFINEMENT_EVALUATIONS)
            .is_none_or(|next| next > evaluator.max_evaluations())
        {
            return Err(evaluator.evaluation_limit_error());
        }
        if evaluator
            .max_intervals()
            .is_some_and(|limit| intervals.len() >= limit)
        {
            return Err(evaluator.interval_limit_error());
        }
        intervals
            .try_reserve(1)
            .map_err(|source| allocation_error(evaluator, source))?;
        let selected = intervals.swap_remove(index);
        let previous_result = (*R::result(&selected)).clone();
        let previous_error = (*R::error(&selected)).clone();
        let [left, right] = R::refine(evaluator, selected).await?;
        evaluations += R::REFINEMENT_EVALUATIONS;
        let without_previous = OpenResult::subtract(&result, &previous_result, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        let with_left = OpenResult::add(&without_previous, R::result(&left), evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        result = OpenResult::add(&with_left, R::result(&right), evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        let without_previous = AdaptiveEstimate::subtract(&error, &previous_error, evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        let with_left = AdaptiveEstimate::add(&without_previous, R::error(&left), evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        error = AdaptiveEstimate::add(&with_left, R::error(&right), evaluator)
            .await
            .map_err(|cause| evaluator.arithmetic_error(cause))?;
        intervals.push(left);
        intervals.push(right);
    }
}
