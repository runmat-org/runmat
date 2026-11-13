//! Current, reset-local incumbent and lifetime best are distinct history
//! indices. Late accepted evaluations keep their original phase provenance.
use super::{
    history::{History, TrialIndex},
    invalid,
    observation::{Measurement, Objective},
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
mod validation;
pub(super) use validation::Validation;

#[derive(Clone, Copy, Default)]
pub(super) struct Update {
    pub incumbent_changed: bool,
    pub best_changed: bool,
    pub became_feasible: bool,
    pub feasibility_reset: bool,
}
#[derive(Default)]
pub(super) struct Tracker {
    current: Option<TrialIndex>,
    incumbent: Option<TrialIndex>,
    best: Option<TrialIndex>,
    feasible: usize,
    incumbent_start: usize,
}

/// Small actual observer-state metadata; all numerical payloads remain in
/// History. These indices are checked against that history on restoration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct State {
    pub current: Option<TrialIndex>,
    pub incumbent: Option<TrialIndex>,
    pub best: Option<TrialIndex>,
    pub feasible: usize,
    pub incumbent_start: usize,
}
#[derive(Clone, Copy)]
enum Rank {
    Incumbent,
    PublicBest,
}
fn better(new: Measurement, old: Measurement, rank: Rank) -> BuiltinResult<bool> {
    let new_feasible = new.nonlinear.violated == 0;
    let old_feasible = old.nonlinear.violated == 0;
    if new_feasible != old_feasible {
        return Ok(new_feasible);
    }
    if !new_feasible {
        if matches!(rank, Rank::PublicBest) {
            // The returned no-feasible solution minimizes maximum violation,
            // distinct from Phase1's fewest-violations incumbent ordering.
            return Ok(new.nonlinear.maximum < old.nonlinear.maximum);
        }
        return Ok(new.nonlinear.violated < old.nonlinear.violated
            || new.nonlinear.violated == old.nonlinear.violated
                && new.nonlinear.maximum < old.nonlinear.maximum);
    }
    match (new.objective, old.objective) {
        (Objective::Measured(new), Objective::Measured(old)) => Ok(new < old),
        (Objective::Absent, Objective::Absent) => Ok(new.nonlinear.maximum < old.nonlinear.maximum),
        _ => Err(invalid("incumbent objective schema changed")),
    }
}
pub(super) fn preferred(new: Measurement, old: Measurement) -> BuiltinResult<bool> {
    better(new, old, Rank::Incumbent)
}
impl Tracker {
    pub(super) fn state(&self) -> State {
        State {
            current: self.current,
            incumbent: self.incumbent,
            best: self.best,
            feasible: self.feasible,
            incumbent_start: self.incumbent_start,
        }
    }

    pub(super) fn validate_state_shape(
        state: State,
        history: &History,
        rows: usize,
    ) -> BuiltinResult<()> {
        if rows > history.len()
            || state.incumbent_start > rows
            || state.feasible > rows
            || state.current.map(|index| index.0) != rows.checked_sub(1)
            || [state.current, state.incumbent, state.best]
                .into_iter()
                .flatten()
                .any(|index| index.0 >= rows)
            || [state.incumbent, state.best]
                .into_iter()
                .flatten()
                .any(|index| history.trial(index).observation.measurement().is_none())
        {
            return Err(invalid(
                "saved observer state differs from its measured history",
            ));
        }
        let feasible_best = state
            .best
            .and_then(|index| history.trial(index).observation.measurement())
            .is_some_and(|value| value.nonlinear.violated == 0);
        if feasible_best != (state.feasible != 0) {
            return Err(invalid(
                "saved feasible count differs from its measured best",
            ));
        }
        Ok(())
    }

    pub(super) async fn from_state(
        state: State,
        history: &History,
        rows: usize,
        phase: u64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut validation = Validation::default();
        validation
            .advance(state, history, rows, phase, cooperation)
            .await?;
        Ok(validation.into_tracker())
    }

    pub(super) fn incumbent(&self) -> Option<TrialIndex> {
        self.incumbent
    }
    pub(super) fn best(&self) -> Option<TrialIndex> {
        self.best
    }

    /// Reset does not discard lifetime best, the latest measured point, or
    /// accepted work. A late old-phase result can improve the current
    /// incumbent, but its original Trial.phase is never rewritten.
    pub(super) fn reset(&mut self, committed: usize) {
        self.incumbent = None;
        self.incumbent_start = committed;
    }

    pub(super) fn record(
        &mut self,
        history: &History,
        measured: TrialIndex,
    ) -> BuiltinResult<Update> {
        let trial = history.trial(measured);
        let Some(new) = trial.observation.measurement() else {
            self.current = Some(measured);
            return Ok(Update::default());
        };
        let old_incumbent = self.incumbent.map(|index| {
            history
                .trial(index)
                .observation
                .measurement()
                .expect("finite incumbent")
        });
        let old_best = self.best.map(|index| {
            history
                .trial(index)
                .observation
                .measurement()
                .expect("finite best")
        });
        // Decide both mutations before publication; schema errors cannot
        // publish only one side of the incumbent/best transition.
        let incumbent_changed = old_incumbent.map_or(Ok(true), |old| preferred(new, old))?;
        let best_changed = old_best.map_or(Ok(true), |old| better(new, old, Rank::PublicBest))?;
        let became_feasible = new.nonlinear.violated == 0
            && old_incumbent.is_none_or(|old| old.nonlinear.violated != 0);
        let feasible = self
            .feasible
            .checked_add(usize::from(new.nonlinear.violated == 0))
            .ok_or_else(|| invalid("feasible observation count overflow"))?;
        self.current = Some(measured);
        self.feasible = feasible;
        if incumbent_changed {
            self.incumbent = Some(measured);
        }
        if best_changed {
            self.best = Some(measured);
        }
        Ok(Update {
            feasibility_reset: false,
            incumbent_changed,
            best_changed,
            became_feasible,
        })
    }
}
