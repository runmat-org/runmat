//! Original feasible directions, prior successful ray, and actual callback waves.
use super::{
    evaluation::{Evaluator, Point},
    invalid,
    linear::{Linear, ProposalStop},
    mesh::Mesh,
    scaling::Scaling,
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use runmat_builtins::OptimPatternOrder;
use std::borrow::Cow;
mod frame;
mod ordering;
mod selection;
pub(super) use frame::History;

#[derive(Clone, Copy)]
pub(super) enum Acceptance {
    Strict,
    SufficientDecrease,
    MadsDecrease,
    Nonuniform,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Completion {
    Finished,
    Budget,
    Time,
    Unrepresentable,
    Numerical,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct MeritChange {
    pub before: f64,
    pub after: f64,
}
pub(super) struct Improvement {
    pub point: Point,
    pub direction: Option<History>,
    pub merit: MeritChange,
}
pub(super) struct Outcome {
    pub improved: Option<Improvement>,
    pub completion: Completion,
    pub rotatable: bool,
}
impl Outcome {
    fn numerical() -> Self {
        Self {
            improved: None,
            completion: Completion::Numerical,
            rotatable: false,
        }
    }
}

pub(super) async fn poll(
    base: &Point,
    geometry: &Linear,
    mesh: &Mesh,
    scaling: &Scaling,
    method: runmat_builtins::OptimPatternPoll,
    binding: f64,
    complete: bool,
    tolerance: f64,
    acceptance: Acceptance,
    order: OptimPatternOrder,
    previous: Option<&[f64]>,
    remember: bool,
    refinements: &mut Option<super::mads::Refinements>,
    mut merit: impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    let Some(best_score) = merit(base, evaluator.cooperation).await? else {
        return Ok(Outcome::numerical());
    };
    let Some(frame) = frame::Frame::new(
        base,
        geometry,
        mesh,
        scaling,
        method,
        binding,
        acceptance,
        refinements,
        evaluator.cooperation,
    )
    .await?
    else {
        return Ok(Outcome::numerical());
    };
    execute_frame(
        base, geometry, frame, complete, tolerance, acceptance, order, previous, remember,
        best_score, None, merit, evaluator,
    )
    .await
}

pub(super) async fn nonuniform(
    base: &Point,
    geometry: &Linear,
    mesh: &Mesh,
    prepared: super::nups::Frame<'_>,
    complete: bool,
    tolerance: f64,
    deadline: &super::timing::Deadline,
    mut merit: impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    let Some(best_score) = merit(base, evaluator.cooperation).await? else {
        return Ok(Outcome::numerical());
    };
    let frame = frame::Frame::nonuniform(prepared, mesh.size())?;
    execute_frame(
        base,
        geometry,
        frame,
        complete,
        tolerance,
        Acceptance::Nonuniform,
        OptimPatternOrder::Consecutive,
        None,
        false,
        best_score,
        Some(deadline),
        merit,
        evaluator,
    )
    .await
}

async fn execute_frame(
    base: &Point,
    geometry: &Linear,
    frame: frame::Frame<'_>,
    complete: bool,
    tolerance: f64,
    acceptance: Acceptance,
    order: OptimPatternOrder,
    previous: Option<&[f64]>,
    remember: bool,
    best_score: f64,
    deadline: Option<&super::timing::Deadline>,
    mut merit: impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    let target = match acceptance {
        Acceptance::Strict | Acceptance::Nonuniform => best_score,
        Acceptance::SufficientDecrease | Acceptance::MadsDecrease => {
            best_score - (1e-4 * frame.measure()) * frame.measure()
        }
    };
    let count = frame.count()?;
    let previous = previous.filter(|_| order == OptimPatternOrder::Success);
    let total = count
        .checked_add(usize::from(previous.is_some()))
        .ok_or_else(|| invalid("success poll count overflow"))?;
    let mut selection = selection::Selection {
        frame: &frame,
        complete,
        target,
        best_score,
        initial_score: best_score,
        remember,
        acceptance,
        deadline,
        outcome: Outcome {
            improved: None,
            completion: Completion::Finished,
            rotatable: frame.rotatable(),
        },
    };
    let order = ordering::Order::new(order, count, evaluator.cooperation).await?;
    let direct = evaluator.direct();
    let mut pending = if direct {
        Vec::new()
    } else {
        super::buffer(total.min(evaluator.wave_width(complete)))?
    };
    let mut previous_step = None;
    let mut unrepresentable = false;
    let mut represented = false;
    for ordinal in 0..total {
        evaluator.cooperation.completed(0).await?;
        if selection.expired() {
            return Ok(selection.outcome);
        }
        if evaluator.remaining() == 0 {
            selection.outcome.completion = Completion::Budget;
            return Ok(selection.outcome);
        }
        let id = if previous.is_some() && ordinal == 0 {
            frame::DirectionId::Previous
        } else {
            frame::DirectionId::Generated(order.at(ordinal - usize::from(previous.is_some())))
        };
        let direction = match id {
            frame::DirectionId::Previous => {
                Cow::Borrowed(previous.expect("selected prior direction"))
            }
            frame::DirectionId::Generated(index) => {
                let Some(direction) = frame.direction(index, evaluator.cooperation).await? else {
                    selection.outcome.completion = Completion::Numerical;
                    return Ok(selection.outcome);
                };
                Cow::Owned(direction)
            }
        };
        let prefix = match frame
            .candidate(
                geometry,
                &base.x,
                &direction,
                tolerance,
                evaluator.cooperation,
            )
            .await?
        {
            Ok(prefix) => prefix,
            // The prior ray is a suggestion, not part of the current spanning
            // certificate. An unavailable prior ray never replaces that set.
            Err(_) if matches!(id, frame::DirectionId::Previous) => continue,
            Err(ProposalStop::Blocked) => continue,
            Err(ProposalStop::Unrepresentable) => {
                unrepresentable = true;
                continue;
            }
            Err(ProposalStop::Numerical) => {
                selection.outcome.completion = Completion::Numerical;
                return Ok(selection.outcome);
            }
        };
        represented = true;
        match id {
            frame::DirectionId::Previous => previous_step = Some(prefix.length),
            frame::DirectionId::Generated(_) => {
                if let (Some(ray), Some(length)) = (previous, previous_step) {
                    if frame::same_point(&prefix.point, &base.x, ray, length, evaluator.cooperation)
                        .await?
                    {
                        continue;
                    }
                }
            }
        }
        if direct {
            if selection.expired() {
                return Ok(selection.outcome);
            }
            let candidate = evaluator.evaluate(prefix.point).await?;
            if selection
                .accept(candidate, id, &mut merit, evaluator)
                .await?
            {
                return Ok(selection.outcome);
            }
            continue;
        }
        pending.push(selection::Proposal {
            point: prefix.point,
            direction: id,
        });
        if pending.len() == evaluator.wave_width(complete) {
            let proposals = std::mem::take(&mut pending);
            if selection.wave(proposals, &mut merit, evaluator).await? {
                return Ok(selection.outcome);
            }
            pending = super::buffer(total.min(evaluator.wave_width(complete)))?;
        }
    }
    if !pending.is_empty() && selection.wave(pending, &mut merit, evaluator).await? {
        return Ok(selection.outcome);
    }
    if unrepresentable && !represented {
        selection.outcome.completion = Completion::Unrepresentable;
    }
    Ok(selection.outcome)
}
