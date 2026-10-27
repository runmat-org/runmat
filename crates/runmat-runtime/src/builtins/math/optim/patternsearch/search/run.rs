//! Original feasible candidates, original merit and actual selected waves.
use super::super::{
    buffer,
    evaluation::{self, Evaluation, Evaluator, Point},
    invalid,
    linear::Linear,
    poll::{Completion, Improvement, Outcome},
    timing::Deadline,
};
use super::{latin, Window};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

async fn accept(
    candidate: Evaluation,
    score: &mut f64,
    initial_score: f64,
    complete: bool,
    outcome: &mut Outcome,
    merit: &mut impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<bool> {
    let candidate = match candidate {
        Ok(point) => point,
        Err(evaluation::Stop::Cached | evaluation::Stop::Undefined) => return Ok(false),
        Err(evaluation::Stop::Budget) => {
            outcome.completion = Completion::Budget;
            return Ok(true);
        }
    };
    if let Some((point, value)) = evaluation::improving(
        candidate,
        *score,
        initial_score,
        merit,
        evaluator.cooperation,
    )
    .await?
    {
        *score = value;
        // A search point is not a generated poll direction or a stale ordinal.
        outcome.improved = Some(Improvement {
            point,
            direction: None,
            merit: super::super::poll::MeritChange {
                before: initial_score,
                after: value,
            },
        });
        return Ok(!complete);
    }
    Ok(false)
}

async fn wave(
    points: Vec<Vec<f64>>,
    score: &mut f64,
    initial: f64,
    complete: bool,
    outcome: &mut Outcome,
    merit: &mut impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<bool> {
    let expected = points.len();
    let returned = evaluator.evaluate_wave(points).await?;
    if returned.len() != expected {
        return Err(invalid("search candidate/result cardinality changed"));
    }
    for point in returned {
        if accept(point, score, initial, complete, outcome, merit, evaluator).await? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(in crate::builtins::math::optim::patternsearch) async fn run(
    base: &Point,
    geometry: &Linear,
    window: &Window,
    count: usize,
    complete: bool,
    tolerance: f64,
    deadline: &Deadline,
    mut merit: impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    let mut outcome = Outcome {
        improved: None,
        completion: Completion::Finished,
        rotatable: false,
    };
    let Some(initial_score) = merit(base, evaluator.cooperation).await? else {
        outcome.completion = Completion::Numerical;
        return Ok(outcome);
    };
    let candidates =
        latin::generate(&window.lower, &window.upper, count, evaluator.cooperation).await?;
    let mut score = initial_score;
    let direct = evaluator.direct();
    let mut pending = Vec::new();
    for point in candidates {
        evaluator.cooperation.completed(0).await?;
        if deadline.reached() {
            outcome.completion = Completion::Time;
            return Ok(outcome);
        }
        if evaluator.remaining() == 0 {
            outcome.completion = Completion::Budget;
            return Ok(outcome);
        }
        let Some(violation) = geometry.violation(&point, evaluator.cooperation).await? else {
            outcome.completion = Completion::Numerical;
            return Ok(outcome);
        };
        if violation > tolerance {
            continue;
        }
        if direct {
            let candidate = evaluator.evaluate(point).await?;
            if accept(
                candidate,
                &mut score,
                initial_score,
                complete,
                &mut outcome,
                &mut merit,
                evaluator,
            )
            .await?
            {
                return Ok(outcome);
            }
        } else {
            let width = evaluator.wave_width(complete);
            if width == 0 {
                outcome.completion = Completion::Budget;
                return Ok(outcome);
            }
            if pending.is_empty() {
                pending = buffer(width.min(count))?;
            }
            pending.push(point);
            if pending.len() == width
                && wave(
                    std::mem::take(&mut pending),
                    &mut score,
                    initial_score,
                    complete,
                    &mut outcome,
                    &mut merit,
                    evaluator,
                )
                .await?
            {
                return Ok(outcome);
            }
        }
    }
    if !pending.is_empty() {
        wave(
            pending,
            &mut score,
            initial_score,
            complete,
            &mut outcome,
            &mut merit,
            evaluator,
        )
        .await?;
    }
    Ok(outcome)
}
