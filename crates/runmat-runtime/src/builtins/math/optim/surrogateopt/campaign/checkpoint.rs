//! Restore measured ownership, not a second normalized history/model cache.
use super::*;

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(in super::super) struct State {
    pub phase: u64,
    pub committed: usize,
    pub tracker: super::super::incumbent::State,
}
impl Campaign {
    pub(in super::super) fn checkpoint_state(&self) -> State {
        State {
            phase: self.phase.0,
            committed: self
                .pending
                .first()
                .map_or(self.history.len(), |index| index.0),
            tracker: self.tracker.state(),
        }
    }

    pub(in super::super) async fn restore_measured(
        &mut self,
        trials: Vec<Trial>,
        state: State,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if self.history.len() != 0 || state.committed > trials.len() {
            return Err(invalid(
                "checkpoint campaign has an invalid measured prefix",
            ));
        }
        for trial in trials {
            if trial.phase.0 > state.phase {
                return Err(invalid("checkpoint trial refers to an uncreated phase"));
            }
            self.history.append(trial, cooperation).await?;
        }
        let tracker = Tracker::from_state(
            state.tracker,
            &self.history,
            state.committed,
            state.phase,
            cooperation,
        )
        .await?;
        let pending = (state.committed..self.history.len()).map(TrialIndex);
        self.pending.try_reserve(pending.len()).map_err(|source| {
            crate::build_runtime_error("surrogateopt: restored pending-index allocation failed")
                .with_builtin(super::super::NAME)
                .with_source(source)
                .build()
        })?;
        self.pending.extend(pending);
        self.tracker = tracker;
        self.phase = PhaseId(state.phase);
        self.schedule = Schedule::new(self.history.variables(), self.phase)?;
        self.search = Search::new(self.dimensions);
        Ok(())
    }
}
