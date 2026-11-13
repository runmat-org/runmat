//! Measured completion/phase transitions on the one retained trial history.
//! Dispatch provenance is immutable, including accepted calls completing after
//! a reset. An old completion can improve best without training the new model.
use super::{
    evaluation::Completed,
    history::{History, MeasurementSource, Trial, TrialIndex},
    incumbent::{Tracker, Update},
    invalid,
    merit::{Improvement, Search},
    queue::{Origin, PhaseId},
    schedule::{Applicability, Refinement, Request, Schedule},
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use std::num::NonZeroUsize;
mod checkpoint;
pub(super) use checkpoint::State;

#[derive(Clone, Copy)]
pub(super) struct Provenance {
    pub phase: PhaseId,
    pub origin: Origin,
}
pub(super) struct Recorded {
    pub update: Option<Update>,
}
pub(super) struct Campaign {
    history: History,
    tracker: Tracker,
    phase: PhaseId,
    search: Search,
    dimensions: usize,
    interval: NonZeroUsize,
    pending: Vec<TrialIndex>,
    schedule: Schedule,
}
impl Campaign {
    pub(super) fn new(
        variables: usize,
        dimensions: usize,
        interval: NonZeroUsize,
    ) -> BuiltinResult<Self> {
        if variables == 0 || dimensions > variables {
            return Err(invalid("invalid surrogate campaign dimension"));
        }
        Ok(Self {
            history: History::new(variables),
            tracker: Tracker::default(),
            phase: PhaseId(0),
            search: Search::new(dimensions),
            dimensions,
            interval,
            pending: Vec::new(),
            schedule: Schedule::new(variables, PhaseId(0))?,
        })
    }
    pub(super) fn history(&self) -> &History {
        &self.history
    }
    pub(super) fn tracker(&self) -> &Tracker {
        &self.tracker
    }
    pub(super) fn phase(&self) -> PhaseId {
        self.phase
    }
    pub(super) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
    pub(super) fn search(&self) -> &Search {
        &self.search
    }
    pub(super) fn refinement(
        &self,
        selected: Applicability,
        before_reset: bool,
    ) -> Option<Request> {
        self.schedule.due(selected, before_reset)
    }
    pub(super) fn started_refinement(&mut self, request: Request) -> BuiltinResult<Refinement> {
        let kind = request.kind;
        self.schedule.started(request)?;
        Ok(kind)
    }

    /// Call once per selected adaptive proposal, not per modeled probe or
    /// observer invocation. Submission/actual point budgets remain with the
    /// dispatch owner, so resetting this cycle never grants more evaluations.
    pub(super) fn selected(&mut self) {
        self.search.selected();
    }

    pub(super) fn reset(&mut self) -> BuiltinResult<PhaseId> {
        if !self.pending.is_empty() {
            return Err(invalid("surrogate reset requires a measured batch commit"));
        }
        let next = self
            .phase
            .0
            .checked_add(1)
            .ok_or_else(|| invalid("surrogate phase counter overflow"))?;
        self.phase = PhaseId(next);
        self.schedule.reset(self.phase)?;
        self.tracker.reset(self.history.len());
        self.search = Search::new(self.dimensions);
        Ok(self.phase)
    }

    pub(super) async fn record_callback(
        &mut self,
        completed: Completed,
        provenance: Provenance,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Recorded> {
        self.record(
            completed,
            provenance,
            MeasurementSource::Callback,
            cooperation,
        )
        .await
    }
    pub(super) async fn record_initial(
        &mut self,
        completed: Completed,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Recorded> {
        self.record(
            completed,
            Provenance {
                phase: self.phase,
                origin: Origin::Initial,
            },
            MeasurementSource::InitialValues,
            cooperation,
        )
        .await
    }
    async fn record(
        &mut self,
        completed: Completed,
        provenance: Provenance,
        source: MeasurementSource,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Recorded> {
        if provenance.phase.0 > self.phase.0 {
            return Err(invalid("completion refers to an uncreated surrogate phase"));
        }
        self.pending.try_reserve(1).map_err(|source| {
            crate::build_runtime_error("surrogateopt: pending result-index allocation failed")
                .with_builtin(super::NAME)
                .with_source(source)
                .build()
        })?;
        let index = self
            .history
            .append(
                Trial {
                    point: completed.point,
                    observation: completed.observation,
                    phase: provenance.phase,
                    origin: provenance.origin,
                    source,
                },
                cooperation,
            )
            .await?;
        self.pending.push(index);
        let update = if self.pending.len() >= self.interval.get() {
            Some(self.flush(cooperation).await?)
        } else {
            None
        };
        Ok(Recorded { update })
    }

    /// A terminal/phase transition flushes a partial group once. Until the
    /// update boundary, history already owns every completed actual payload,
    /// but incumbent/scale and the captured model remain unchanged.
    pub(super) async fn flush(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Update> {
        cooperation.completed(0).await?;
        let mut combined = Update::default();
        let mut adaptive = false;
        let mut improved = false;
        for &index in &self.pending {
            let trial = self.history.trial(index);
            let update = self.tracker.record(&self.history, index)?;
            if matches!(trial.source, MeasurementSource::Callback) {
                self.schedule.completed(trial.phase, trial.origin)?;
            }
            combined.incumbent_changed |= update.incumbent_changed;
            combined.best_changed |= update.best_changed;
            combined.became_feasible |= update.became_feasible;
            combined.feasibility_reset |= trial.phase == self.phase
                && trial.observation.schema().objective
                    == super::observation::ObjectiveKind::Absent
                && trial
                    .observation
                    .measurement()
                    .is_some_and(|value| value.nonlinear.violated == 0);
            // Late accepted work retains its phase and may improve lifetime
            // best, but never changes a new phase's adaptive scale statistics.
            if trial.phase == self.phase && trial.origin == Origin::Adaptive {
                adaptive = true;
                improved |= update.incumbent_changed;
            }
            cooperation.completed(1).await?;
        }
        if adaptive {
            self.search.measured(if improved {
                Improvement::Success
            } else {
                Improvement::Failure
            });
        }
        self.pending.clear();
        Ok(combined)
    }

    pub(super) async fn model_rows(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<usize>> {
        self.history.model_rows(self.phase, true, cooperation).await
    }

    #[cfg(test)]
    pub(super) async fn model_snapshot(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<super::history::Snapshot> {
        if !self.pending.is_empty() {
            return Err(invalid("model update requires a measured batch commit"));
        }
        let rows = self.model_rows(cooperation).await?;
        super::history::Snapshot::capture(&self.history, rows, cooperation).await
    }
}
