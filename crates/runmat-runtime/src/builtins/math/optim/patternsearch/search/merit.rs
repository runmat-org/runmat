//! Borrowed search observations share the enclosing original evaluator. The
//! transaction retains only pending coordinate rows and the best real point.
use super::super::{
    buffer,
    evaluation::{self, Evaluator, Point},
    invalid,
    linear::Linear,
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use std::ops::ControlFlow;

pub(super) struct Best {
    pub point: Point,
    pub score: f64,
}
pub(super) struct Admission<'a> {
    pub base: &'a Point,
    pub initial: f64,
    pub geometry: &'a Linear,
    pub tolerance: f64,
    pub reuse: Reuse,
}
#[derive(Clone, Copy)]
pub(super) enum Reuse {
    KnownBase,
    RequestedMeasurement,
}
pub(super) struct Transaction {
    scores: Vec<f64>,
    pending: Vec<Vec<f64>>,
    selected: Vec<usize>,
}
impl Transaction {
    pub fn new(points: usize) -> BuiltinResult<Self> {
        Ok(Self {
            scores: buffer(points)?,
            pending: buffer(points)?,
            selected: buffer(points)?,
        })
    }
    pub async fn push(
        &mut self,
        x: Vec<f64>,
        admission: &Admission<'_>,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<()> {
        if x.len() != admission.base.x.len() {
            return Err(invalid("borrowed search coordinate extent changed"));
        }
        let mut same = matches!(admission.reuse, Reuse::KnownBase);
        if same {
            for (left, right) in x.chunks(1024).zip(admission.base.x.chunks(1024)) {
                same &= left
                    .iter()
                    .zip(right)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                evaluator.cooperation.completed(left.len() as u64).await?;
            }
        }
        let score = if same {
            admission.initial
        } else if !admission
            .geometry
            .feasible(&x, admission.tolerance, evaluator.cooperation)
            .await?
        {
            f64::NAN
        } else {
            self.selected.push(self.scores.len());
            self.pending.push(x);
            f64::NAN
        };
        self.scores.push(score);
        Ok(())
    }
    pub async fn finish<S>(
        mut self,
        best: &mut Option<Best>,
        initial: f64,
        merit: &mut impl for<'a> std::ops::AsyncFnMut(
            &'a Point,
            &'a mut ScalarCooperation,
        ) -> BuiltinResult<Option<f64>>,
        evaluator: &mut Evaluator<'_>,
        stop: impl Fn() -> Option<S>,
        budget: S,
    ) -> BuiltinResult<ControlFlow<S, Vec<f64>>> {
        evaluator.cooperation.completed(0).await?;
        if let Some(stop) = stop() {
            return Ok(ControlFlow::Break(stop));
        }
        // A vectorized/user wave is indivisible. Refuse it before dispatch
        // rather than exceed the parent's budget or fabricate a partial row.
        if self.pending.len() > evaluator.remaining() {
            return Ok(ControlFlow::Break(budget));
        }
        if evaluator.direct() && self.pending.len() == 1 {
            let x = self.pending.pop().expect("one direct point");
            match evaluator.evaluate(x).await? {
                Ok(point) => {
                    self.scores[self.selected[0]] =
                        observe(point, best, initial, merit, evaluator).await?
                }
                Err(evaluation::Stop::Budget) => return Ok(ControlFlow::Break(budget)),
                Err(evaluation::Stop::Cached | evaluation::Stop::Undefined) => {}
            }
        } else if !self.pending.is_empty() {
            let returned = evaluator.evaluate_wave(self.pending).await?;
            if returned.len() != self.selected.len() {
                return Err(invalid("borrowed merit wave cardinality changed"));
            }
            for (slot, point) in self.selected.into_iter().zip(returned) {
                match point {
                    Ok(point) => {
                        self.scores[slot] = observe(point, best, initial, merit, evaluator).await?
                    }
                    Err(evaluation::Stop::Budget) => return Ok(ControlFlow::Break(budget)),
                    Err(evaluation::Stop::Cached | evaluation::Stop::Undefined) => {}
                }
            }
        }
        Ok(ControlFlow::Continue(self.scores))
    }
}
async fn observe(
    point: Point,
    best: &mut Option<Best>,
    initial: f64,
    merit: &mut impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<f64> {
    let Some(score) = merit(&point, evaluator.cooperation).await? else {
        return Ok(f64::NAN);
    };
    if score < best.as_ref().map_or(initial, |best| best.score) {
        *best = Some(Best { point, score });
    }
    Ok(score)
}
