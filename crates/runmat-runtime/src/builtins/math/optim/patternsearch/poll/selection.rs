//! One acceptance rule shared by direct and actual batched polling.
use super::super::{
    buffer,
    evaluation::{Evaluation, Evaluator, Point, Stop},
    invalid,
};
use super::{
    frame::{DirectionId, Frame},
    Acceptance, Completion, Improvement, Outcome,
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
pub(super) struct Proposal {
    pub point: Vec<f64>,
    pub direction: DirectionId,
}
pub(super) struct Selection<'a, 'b> {
    pub frame: &'a Frame<'b>,
    pub complete: bool,
    pub target: f64,
    pub best_score: f64,
    pub initial_score: f64,
    pub remember: bool,
    pub acceptance: Acceptance,
    pub deadline: Option<&'a super::super::timing::Deadline>,
    pub outcome: Outcome,
}
impl Selection<'_, '_> {
    pub fn expired(&mut self) -> bool {
        if self
            .deadline
            .is_some_and(super::super::timing::Deadline::reached)
        {
            self.outcome.completion = Completion::Time;
            true
        } else {
            false
        }
    }
    pub async fn accept(
        &mut self,
        candidate: Evaluation,
        direction: DirectionId,
        merit: &mut impl for<'a> std::ops::AsyncFnMut(
            &'a Point,
            &'a mut ScalarCooperation,
        ) -> BuiltinResult<Option<f64>>,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<bool> {
        let candidate = match candidate {
            Ok(point) => point,
            Err(Stop::Cached | Stop::Undefined) => return Ok(false),
            Err(Stop::Budget) => {
                self.outcome.completion = Completion::Budget;
                return Ok(true);
            }
        };
        if let Some((candidate, score)) = super::super::evaluation::improving(
            candidate,
            self.best_score,
            self.target,
            merit,
            evaluator.cooperation,
        )
        .await?
        {
            let direction = if self.remember {
                Some(self.frame.history(direction, evaluator.cooperation).await?)
            } else {
                None
            };
            self.best_score = score;
            self.outcome.improved = Some(Improvement {
                point: candidate,
                direction,
                merit: super::MeritChange {
                    before: self.initial_score,
                    after: score,
                },
            });
            let forcing = !matches!(self.acceptance, Acceptance::Nonuniform)
                || super::super::nups::relative_decrease(self.target, score)
                    .is_some_and(|gain| gain >= super::super::nups::RELATIVE_DECREASE);
            if !self.complete && forcing {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub async fn wave(
        &mut self,
        proposals: Vec<Proposal>,
        merit: &mut impl for<'a> std::ops::AsyncFnMut(
            &'a Point,
            &'a mut ScalarCooperation,
        ) -> BuiltinResult<Option<f64>>,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<bool> {
        if self.expired() {
            return Ok(true);
        }
        let mut directions = buffer(proposals.len())?;
        let mut points = buffer(proposals.len())?;
        for proposal in proposals {
            directions.push(proposal.direction);
            points.push(proposal.point);
        }
        let returned = evaluator.evaluate_wave(points).await?;
        if returned.len() != directions.len() {
            return Err(invalid("poll result/direction cardinality changed"));
        }
        for (point, direction) in returned.into_iter().zip(directions) {
            if self.accept(point, direction, merit, evaluator).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
