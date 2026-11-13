//! Outer expensive-call route. Route selection is immutable for this solve;
//! callback/provider errors never replay through another execution route.
use super::{
    campaign::{Campaign, Provenance},
    evaluation::{BatchEvaluation, Completed, Evaluation, Evaluator, Input, Seed},
    invalid,
    observation::Form,
    parameters::Parameters,
    queue,
};
use crate::{
    builtins::math::optim::common::ScalarCooperation,
    call::descriptor::RepeatableCallable,
    execution::{CallbackBatch, CallbackParallelMode},
    BuiltinResult,
};
use std::num::NonZeroUsize;
mod batch;
use batch::Batch;

enum Route<'a> {
    Serial(Evaluator<'a>),
    Parallel(queue::Queue<'a>),
}
#[derive(Default)]
pub(super) struct Commit {
    pub points: usize,
    pub updates: usize,
    pub feasibility_reset: bool,
}
pub(super) struct Dispatcher<'a> {
    route: Route<'a>,
    variables: usize,
    interval: NonZeroUsize,
    tolerance: f64,
    vectorized: bool,
    staged: Option<Batch>,
}
impl<'a> Dispatcher<'a> {
    /// The public ingress has already selected BOTH local definition and
    /// optional worker route from the same original rooted source. No worker
    /// capability, implicit context or pool is created by the serial route.
    pub(super) fn new(
        source: &'a RepeatableCallable,
        workers: Option<&'a CallbackBatch>,
        parameters: &Parameters,
        variables: usize,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if workers.is_some() && parameters.parallel == CallbackParallelMode::Off
            || workers.is_none() && parameters.parallel == CallbackParallelMode::On
        {
            return Err(invalid(
                "selected worker route does not match requested parallel policy",
            ));
        }
        let route = match workers {
            Some(workers) => Route::Parallel(queue::Queue::new(
                workers,
                parameters.evaluations,
                cooperation,
            )?),
            None => Route::Serial(Evaluator::new(source, variables, parameters.evaluations)?),
        };
        Ok(Self {
            route,
            variables,
            interval: parameters.batch,
            tolerance: parameters.nonlinear_tolerance,
            vectorized: parameters.vectorized,
            staged: None,
        })
    }
    pub(super) fn counts(&self) -> (usize, usize) {
        match &self.route {
            Route::Serial(evaluator) => (evaluator.evaluated(), evaluator.evaluated()),
            Route::Parallel(queue) => queue.counts(),
        }
    }
    pub(super) fn restore_evaluated(&mut self, evaluated: usize) -> BuiltinResult<()> {
        if self.staged.is_some() {
            return Err(invalid("count restoration requires an unstaged dispatcher"));
        }
        match &mut self.route {
            Route::Serial(evaluator) => evaluator.restore_evaluated(evaluated),
            Route::Parallel(queue) => queue.restore_evaluated(evaluated),
        }
    }
    pub(super) fn vacancy(&self) -> usize {
        match &self.route {
            Route::Parallel(queue) => queue.vacancy(),
            Route::Serial(evaluator) => evaluator
                .remaining()
                .saturating_sub(self.staged.as_ref().map_or(0, Batch::len)),
        }
    }
    pub(super) fn in_flight(&self) -> usize {
        match &self.route {
            Route::Parallel(queue) => queue.in_flight(),
            Route::Serial(_) => 0,
        }
    }
    pub(super) fn parallel(&self) -> bool {
        matches!(&self.route, Route::Parallel(_))
    }
    pub(super) fn remaining(&self) -> usize {
        match &self.route {
            Route::Parallel(queue) => queue.remaining(),
            Route::Serial(evaluator) => evaluator
                .remaining()
                .saturating_sub(self.staged.as_ref().map_or(0, Batch::len)),
        }
    }
    pub(super) fn planned(&self, phase: queue::PhaseId) -> usize {
        match &self.route {
            Route::Parallel(queue) => queue.planned(phase),
            Route::Serial(_) => self.staged.as_ref().map_or(0, |batch| batch.planned(phase)),
        }
    }
    pub(super) async fn contains_point(
        &self,
        point: &[f64],
        phase: queue::PhaseId,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        match &self.route {
            Route::Parallel(queue) => queue.contains_point(point, phase, cooperation).await,
            Route::Serial(_) => match &self.staged {
                Some(batch) if batch.planned(phase) != 0 => {
                    batch.contains_point(point, cooperation).await
                }
                Some(_) => Ok(false),
                None => Ok(false),
            },
        }
    }
    /// A staged vectorized row is not an objective call. A worker submission
    /// starts immediately and owns PendingBatch before this method returns.
    pub(super) async fn submit(
        &mut self,
        point: Vec<f64>,
        provenance: Provenance,
        campaign: &mut Campaign,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Commit> {
        self.submit_seed(point, provenance, None, campaign, cooperation)
            .await
    }
    pub(super) async fn submit_seed(
        &mut self,
        point: Vec<f64>,
        provenance: Provenance,
        seed: Option<Seed>,
        campaign: &mut Campaign,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Commit> {
        if self.vacancy() == 0 {
            return Err(invalid("objective dispatch has no remaining point slot"));
        }
        if let Some(seed) = &seed {
            if !seed.same_point(&point, cooperation).await? {
                return Err(invalid(
                    "supplied measurements require their exact original point",
                ));
            }
        }
        match &mut self.route {
            Route::Parallel(queue) => {
                let input = Input::new(point, self.variables, Form::Point, cooperation).await?;
                queue.enqueue(queue::Point {
                    phase: provenance.phase,
                    origin: provenance.origin,
                    input,
                    seed,
                })?;
                queue.dispatch(cooperation)?;
                Ok(Commit::default())
            }
            Route::Serial(evaluator) if !self.vectorized => {
                let Evaluation::Completed(completed) = evaluator
                    .evaluate(point, self.tolerance, cooperation)
                    .await?
                else {
                    return Err(invalid("admitted serial point lost its evaluation slot"));
                };
                let mut commit = Commit::default();
                let completed = completed
                    .with_seed(seed, self.tolerance, cooperation)
                    .await?;
                record(completed, provenance, campaign, &mut commit, cooperation).await?;
                Ok(commit)
            }
            Route::Serial(evaluator) => {
                if self.staged.is_none() {
                    let target = NonZeroUsize::new(self.interval.get().min(evaluator.remaining()))
                        .ok_or_else(|| {
                            invalid("vectorized callback has no remaining point slot")
                        })?;
                    self.staged = Some(Batch::new(target, self.variables, cooperation).await?);
                }
                let batch = self.staged.as_mut().expect("staged vectorized batch");
                batch.push(point, provenance, seed, cooperation).await?;
                if !batch.full() {
                    return Ok(Commit::default());
                }
                let (coordinates, rows, provenance) =
                    self.staged.take().expect("full staged batch").into_full()?;
                let BatchEvaluation::Completed(completed) = evaluator
                    .evaluate_batch(coordinates, rows, self.tolerance, cooperation)
                    .await?
                else {
                    return Err(invalid(
                        "admitted vectorized batch lost its actual point slots",
                    ));
                };
                if completed.len() != provenance.len() {
                    return Err(invalid("vectorized dispatch lost row provenance"));
                }
                let mut commit = Commit::default();
                for (completed, provenance) in completed.into_iter().zip(provenance) {
                    let completed = completed
                        .with_seed(provenance.seed, self.tolerance, cooperation)
                        .await?;
                    record(
                        completed,
                        provenance.provenance,
                        campaign,
                        &mut commit,
                        cooperation,
                    )
                    .await?;
                }
                Ok(commit)
            }
        }
    }
    pub(super) async fn next(
        &mut self,
        campaign: &mut Campaign,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Commit> {
        match &mut self.route {
            Route::Serial(_) => Ok(Commit::default()),
            Route::Parallel(queue) => {
                let recorded = queue
                    .next_recorded(campaign, self.tolerance, cooperation)
                    .await?;
                Ok(recorded.map_or_else(Commit::default, |recorded| Commit {
                    points: 1,
                    updates: usize::from(recorded.update.is_some()),
                    feasibility_reset: recorded
                        .update
                        .is_some_and(|update| update.feasibility_reset),
                }))
            }
        }
    }
    pub(super) fn reset_unsubmitted(&mut self) {
        self.staged = None;
        if let Route::Parallel(queue) = &mut self.route {
            queue.reset_unsubmitted();
        }
    }
    /// Evaluate feasible final construction rows without padding or replay.
    /// Called only when proposal search ends, not after a time/user stop.
    pub(super) async fn finish_construction(
        &mut self,
        campaign: &mut Campaign,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Commit> {
        let Some(batch) = self.staged.take() else {
            return Ok(Commit::default());
        };
        let Route::Serial(evaluator) = &mut self.route else {
            return Err(invalid("worker route contains serial staging"));
        };
        let (coordinates, rows, provenance) = batch.into_final(cooperation).await?;
        let BatchEvaluation::Completed(completed) = evaluator
            .evaluate_batch(coordinates, rows, self.tolerance, cooperation)
            .await?
        else {
            return Err(invalid("final staged batch lost its original point slots"));
        };
        if completed.len() != provenance.len() {
            return Err(invalid("final batch lost row provenance"));
        }
        let mut commit = Commit::default();
        for (completed, provenance) in completed.into_iter().zip(provenance) {
            let completed = completed
                .with_seed(provenance.seed, self.tolerance, cooperation)
                .await?;
            record(
                completed,
                provenance.provenance,
                campaign,
                &mut commit,
                cooperation,
            )
            .await?;
        }
        if campaign.has_pending() {
            commit.feasibility_reset |= campaign.flush(cooperation).await?.feasibility_reset;
            commit.updates += 1;
        }
        Ok(commit)
    }
    pub(super) async fn unique(
        &mut self,
        point: Vec<f64>,
        provenance: Provenance,
        campaign: &mut Campaign,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Commit> {
        self.unique_seed(point, provenance, None, campaign, cooperation)
            .await
    }
    pub(super) async fn unique_seed(
        &mut self,
        point: Vec<f64>,
        provenance: Provenance,
        seed: Option<Seed>,
        campaign: &mut Campaign,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Commit> {
        if self.in_flight() != 0 || self.staged.is_some() || self.counts().0 != 0 {
            return Err(invalid("unique point requires a fresh objective dispatch"));
        }
        let mut commit = if matches!(self.route, Route::Parallel(_)) || !self.vectorized {
            self.submit_seed(point, provenance, seed, campaign, cooperation)
                .await?
        } else {
            let Route::Serial(evaluator) = &mut self.route else {
                return Err(invalid("unique vectorized point changed route"));
            };
            let BatchEvaluation::Completed(completed) = evaluator
                .evaluate_batch(point, NonZeroUsize::MIN, self.tolerance, cooperation)
                .await?
            else {
                return Err(invalid("unique point lost its objective slot"));
            };
            let mut commit = Commit::default();
            if completed.len() != 1 {
                return Err(invalid("unique vectorized callback lost its sole row"));
            }
            let completed = completed
                .into_iter()
                .next()
                .expect("checked sole row")
                .with_seed(seed, self.tolerance, cooperation)
                .await?;
            record(completed, provenance, campaign, &mut commit, cooperation).await?;
            commit
        };
        if self.in_flight() != 0 {
            let returned = self.next(campaign, cooperation).await?;
            commit.points += returned.points;
            commit.updates += returned.updates;
            commit.feasibility_reset |= returned.feasibility_reset;
        }
        if campaign.has_pending() {
            commit.feasibility_reset |= campaign.flush(cooperation).await?.feasibility_reset;
            commit.updates += 1;
        }
        Ok(commit)
    }
    pub(super) fn pump(&mut self, cooperation: &mut ScalarCooperation) -> BuiltinResult<()> {
        if let Route::Parallel(queue) = &mut self.route {
            queue.dispatch(cooperation)?;
        }
        Ok(())
    }
}
async fn record(
    completed: Completed,
    provenance: Provenance,
    campaign: &mut Campaign,
    commit: &mut Commit,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let recorded = campaign
        .record_callback(completed, provenance, cooperation)
        .await?;
    commit.points += 1;
    commit.updates += usize::from(recorded.update.is_some());
    commit.feasibility_reset |= recorded
        .update
        .is_some_and(|update| update.feasibility_reset);
    Ok(())
}
