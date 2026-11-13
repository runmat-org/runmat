//! Surrogate FIFO dispatch on one already selected worker-pool admission.
//! Surplus points are rooted but unsubmitted. Only actual selected-width
//! calls run; resets discard queued points, never accepted work or provenance.
use super::{
    campaign::{Campaign, Provenance, Recorded},
    evaluation::Input,
    invalid, NAME,
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::execution::{CallbackBatch, RootedValueSequence};
use crate::BuiltinResult;
use futures::future::LocalBoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PhaseId(pub(super) u64);
pub(super) use runmat_builtins::SurrogateSampleOrigin as Origin;
pub(super) struct Point {
    pub phase: PhaseId,
    pub origin: Origin,
    pub input: Input,
    pub seed: Option<super::evaluation::Seed>,
}
pub(super) struct Completion {
    pub point: Point,
    pub returned: RootedValueSequence,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct CallId(usize);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Dispatch {
    Running,
    Empty,
    EvaluationLimit,
}
pub(super) struct Queue<'a> {
    selected: &'a CallbackBatch,
    pending: VecDeque<Point>,
    running: FuturesUnordered<LocalBoxFuture<'a, (CallId, BuiltinResult<RootedValueSequence>)>>,
    active: HashMap<CallId, Point>,
    target: usize,
    limit: usize,
    admitted: usize,
    completed: usize,
}
impl<'a> Queue<'a> {
    pub(super) fn new(
        selected: &'a CallbackBatch,
        evaluation_limit: usize,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if selected.width() == 0 || cooperation.runtime().is_none() {
            return Err(invalid("parallel queue lacks its original pool/runtime"));
        }
        let width = selected.width();
        let target = (width / 10)
            .checked_mul(3)
            .and_then(|n| n.checked_add(((width % 10) * 3).div_ceil(10)))
            .and_then(|extra| width.checked_add(extra))
            .ok_or_else(|| invalid("parallel queue extent overflow"))?;
        crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(),
            NAME,
            "finite host-memory execution has no complete surrogate queue resource bound",
        )?;
        Ok(Self {
            selected,
            pending: VecDeque::new(),
            running: FuturesUnordered::new(),
            active: HashMap::new(),
            target,
            limit: evaluation_limit,
            admitted: 0,
            completed: 0,
        })
    }
    pub(super) fn vacancy(&self) -> usize {
        self.target
            .saturating_sub(self.pending.len())
            .saturating_sub(self.running.len())
            .min(
                self.limit
                    .saturating_sub(self.admitted)
                    .saturating_sub(self.pending.len()),
            )
    }
    pub(super) fn remaining(&self) -> usize {
        self.limit
            .saturating_sub(self.admitted)
            .saturating_sub(self.pending.len())
    }
    pub(super) fn enqueue(&mut self, point: Point) -> BuiltinResult<()> {
        if !point.input.scalar() {
            return Err(invalid(
                "worker queue requires an original scalar point invocation",
            ));
        }
        if self.vacancy() == 0 {
            return Err(invalid("surrogate FIFO has no unsubmitted evaluation slot"));
        }
        self.pending.try_reserve(1).map_err(|source| {
            crate::build_runtime_error("surrogateopt: FIFO allocation failed")
                .with_builtin(NAME)
                .with_source(source)
                .build()
        })?;
        self.pending.push_back(point);
        Ok(())
    }
    pub(super) fn dispatch(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Dispatch> {
        while self.running.len() < self.selected.width() && self.admitted < self.limit {
            let Some(point) = self.pending.pop_front() else {
                break;
            };
            self.active.try_reserve(1).map_err(|source| {
                crate::build_runtime_error("surrogateopt: active call metadata allocation failed")
                    .with_builtin(NAME)
                    .with_source(source)
                    .build()
            })?;
            let (_, budget) = cooperation.runtime_budget();
            let future = self.selected.start_call(
                vec![point.input.argument().value().clone()],
                1,
                budget.ok_or_else(|| invalid("parallel dispatch lost its original CPU budget"))?,
            )?;
            // Submission is synchronous and FIFO. The returned future owns
            // PendingBatch already, including if push/unwind drops it unpolled.
            self.admitted += 1;
            let id = CallId(self.admitted);
            self.active.insert(id, point);
            self.running
                .push(Box::pin(async move { (id, future.await) }));
        }
        Ok(if !self.running.is_empty() {
            Dispatch::Running
        } else if self.admitted == self.limit {
            Dispatch::EvaluationLimit
        } else {
            Dispatch::Empty
        })
    }
    pub(super) async fn next(&mut self) -> BuiltinResult<Option<Completion>> {
        // Every live call uses the one cancellation-first await owner and its
        // exact original token; this stream adds no polling/scheduling policy.
        let Some((id, returned)) = self.running.next().await else {
            return Ok(None);
        };
        let point = self
            .active
            .remove(&id)
            .ok_or_else(|| invalid("accepted callback lost its original point owner"))?;
        let returned = returned?;
        self.completed = self
            .completed
            .checked_add(1)
            .ok_or_else(|| invalid("completed evaluation count overflow"))?;
        Ok(Some(Completion { point, returned }))
    }
    /// Construction-site uniqueness includes unsubmitted and authentic
    /// accepted inputs. The active map is their sole point owner, not copies
    /// of a future's hidden workspace; adaptive MinSampleDistance=0 remains
    /// independently controlled by the merit owner.
    pub(super) async fn contains_point(
        &self,
        point: &[f64],
        phase: PhaseId,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        for queued in self.pending.iter().chain(self.active.values()) {
            if queued.phase != phase {
                cooperation.completed(1).await?;
                continue;
            }
            if queued.input.same_point(point, cooperation).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }
    /// Decode exactly the accepted point's returned generation, then transfer
    /// that point/result pair into canonical history with original provenance.
    pub(super) async fn next_recorded(
        &mut self,
        campaign: &mut Campaign,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Recorded>> {
        let Some(Completion { point, returned }) = self.next().await? else {
            return Ok(None);
        };
        let provenance = Provenance {
            phase: point.phase,
            origin: point.origin,
        };
        let mut completed = point.input.finish(returned, tolerance, cooperation).await?;
        let actual = completed
            .pop()
            .ok_or_else(|| invalid("accepted scalar call lost its actual row"))?;
        if !completed.is_empty() {
            return Err(invalid("accepted scalar call returned extra point rows"));
        }
        let actual = actual.with_seed(point.seed, tolerance, cooperation).await?;
        campaign
            .record_callback(actual, provenance, cooperation)
            .await
            .map(Some)
    }

    pub(super) fn reset_unsubmitted(&mut self) {
        self.pending.clear();
    }
    pub(super) fn counts(&self) -> (usize, usize) {
        (self.admitted, self.completed)
    }
    pub(super) fn restore_evaluated(&mut self, evaluated: usize) -> BuiltinResult<()> {
        if self.admitted != 0
            || self.completed != 0
            || !self.pending.is_empty()
            || !self.running.is_empty()
        {
            return Err(invalid("worker count restoration requires an unused queue"));
        }
        self.admitted = evaluated;
        self.completed = evaluated;
        Ok(())
    }
    pub(super) fn planned(&self, phase: PhaseId) -> usize {
        self.pending
            .iter()
            .chain(self.active.values())
            .filter(|point| point.phase == phase)
            .count()
    }
    pub(super) fn in_flight(&self) -> usize {
        self.running.len()
    }
}
