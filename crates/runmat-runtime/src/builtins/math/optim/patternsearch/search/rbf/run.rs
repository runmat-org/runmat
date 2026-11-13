//! One model proposal, then at most one original callback point.
use super::super::super::{
    augmented::Augmented,
    buffer,
    evaluation::{self, Evaluator, Point},
    invalid,
    linear::Linear,
    poll::{Completion, Improvement, MeritChange, Outcome},
    timing::Deadline,
};
use crate::builtins::math::optim::common::{cubic_rbf, feasible_cone::IndependentAxes};
use crate::BuiltinResult;

pub(in crate::builtins::math::optim::patternsearch) async fn run(
    base: &Point,
    geometry: &Linear,
    augmented: Option<&Augmented>,
    tolerance: f64,
    deadline: &Deadline,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    let mut outcome = Outcome {
        improved: None,
        completion: Completion::Finished,
        rotatable: false,
    };
    if deadline.reached() {
        outcome.completion = Completion::Time;
        return Ok(outcome);
    }
    if evaluator.remaining() == 0 {
        outcome.completion = Completion::Budget;
        return Ok(outcome);
    }
    // Every borrow of training/model/backing ends BEFORE a new real trial can
    // reset the history or change its generation.
    let candidate = {
        let history = evaluator
            .state
            .history
            .as_ref()
            .ok_or_else(|| invalid("selected RBF search lacks its history"))?;
        let bounds = evaluator
            .state
            .original_bounds
            .as_ref()
            .ok_or_else(|| invalid("selected RBF search lacks original bounds"))?;
        let Some(window) = bounds.window(history, evaluator.cooperation).await? else {
            return Ok(outcome);
        };
        let axes = match bounds.axes(geometry, evaluator.cooperation).await? {
            IndependentAxes::Unique | IndependentAxes::Unrepresentable => return Ok(outcome),
            IndependentAxes::Axes(axes) => axes,
        };
        let mut scores = buffer(history.len())?;
        for record in history.records() {
            let score = match augmented {
                Some(augmented) => {
                    augmented
                        .merit_values(
                            record.f,
                            &record.inequality,
                            &record.equality,
                            evaluator.cooperation,
                        )
                        .await?
                }
                None => Some(record.f),
            };
            let Some(score) = score else {
                return Ok(outcome);
            };
            scores.push(score);
            evaluator.cooperation.completed(1).await?;
        }
        let Some(training) = history
            .training(&scores, Some(&axes), evaluator.cooperation)
            .await?
        else {
            return Ok(outcome);
        };
        let model = match cubic_rbf::fit(
            cubic_rbf::Policy {
                identity: super::super::super::IDENTITY,
                invalid: &runmat_builtins::PATTERNSEARCH_ERROR_INVALID_ARGUMENT,
            },
            &training,
            evaluator.cooperation,
        )
        .await?
        {
            cubic_rbf::Fit::Poised(model) => model,
            cubic_rbf::Fit::Unpoised | cubic_rbf::Fit::Undefined | cubic_rbf::Fit::NoFunctions => {
                return Ok(outcome)
            }
        };
        let Some(candidate) = super::model::propose(
            &training,
            &model,
            &window,
            geometry,
            tolerance,
            deadline,
            evaluator.cooperation,
        )
        .await?
        else {
            return Ok(outcome);
        };
        if history.contains(&candidate, evaluator.cooperation).await? {
            return Ok(outcome);
        }
        candidate
    };
    if deadline.reached() {
        outcome.completion = Completion::Time;
        return Ok(outcome);
    }
    let before = match augmented {
        Some(augmented) => augmented.merit(base, evaluator.cooperation).await?,
        None => Some(base.f),
    };
    let Some(before) = before else {
        outcome.completion = Completion::Numerical;
        return Ok(outcome);
    };
    let candidate = match evaluator.evaluate(candidate).await? {
        Ok(candidate) => candidate,
        Err(evaluation::Stop::Budget) => {
            outcome.completion = Completion::Budget;
            return Ok(outcome);
        }
        Err(evaluation::Stop::Undefined | evaluation::Stop::Cached) => return Ok(outcome),
    };
    let improved = evaluation::improving(
        candidate, before, before,
        &mut async |point: &Point, cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation| {
            match augmented {
                Some(augmented) => augmented.merit(point, cooperation).await,
                None => Ok(Some(point.f)),
            }
        },
        evaluator.cooperation,
    ).await?;
    if let Some((candidate, after)) = improved {
        outcome.improved = Some(Improvement {
            point: candidate,
            direction: None,
            merit: MeritChange { before, after },
        });
    }
    Ok(outcome)
}
