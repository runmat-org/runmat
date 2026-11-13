//! Snapshot cuts drain only accepted work. Unsubmitted FIFO proposals remain
//! with the live continuation and never acquire an evaluation slot here.
use super::*;
use crate::builtins::math::optim::surrogateopt::{
    history::{MeasurementSource, Trial},
    notice::Frame,
};

pub(in super::super) enum Quiescence {
    Iteration,
    Ready,
}
impl Controller<'_> {
    pub(in super::super) fn original_domain(&self) -> &Domain {
        &self.domain
    }
    pub(in super::super) async fn quiesce(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Quiescence> {
        loop {
            cooperation.completed(0).await?;
            if self.dispatcher.in_flight() == 0 {
                let (accepted, completed) = self.counts();
                if accepted != completed {
                    return Err(invalid(
                        "checkpoint cut still owns an incomplete accepted call",
                    ));
                }
                return Ok(Quiescence::Ready);
            }
            let commit = self
                .dispatcher
                .next(&mut self.campaign, cooperation)
                .await?;
            if self.committed(commit, cooperation).await? {
                return Ok(Quiescence::Iteration);
            }
        }
    }

    pub(in super::super) async fn restore_measured(
        &mut self,
        trials: Vec<Trial>,
        state: super::super::campaign::State,
        frame: Frame,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if frame.rows != trials.len()
            || frame.resets != state.phase
            || frame.tracker != state.tracker
        {
            return Err(invalid("checkpoint final frame differs from its campaign"));
        }
        let mut actual = 0usize;
        let mut scratch = super::super::public_values::capacity(self.domain.bounds().0.len())?;
        for trial in &trials {
            scratch.clear();
            for column in 0..trial.point.variables() {
                scratch.push(trial.point.coordinate(column));
                cooperation.completed(1).await?;
            }
            if !self
                .domain
                .classify(&scratch, self.linear_tolerance, cooperation)
                .await?
                .feasible
            {
                return Err(invalid(
                    "saved measured point is outside its original domain",
                ));
            }
            if matches!(trial.source, MeasurementSource::Callback) {
                actual = actual
                    .checked_add(1)
                    .ok_or_else(|| invalid("saved evaluation count overflow"))?;
            }
        }
        if actual != frame.evaluations.0 {
            return Err(invalid(
                "saved point-evaluation count differs from callback provenance",
            ));
        }
        self.campaign
            .restore_measured(trials, state, cooperation)
            .await?;
        if self.initial.is_some() {
            let mut matched = super::super::public_values::capacity(self.campaign.history().len())?;
            matched.resize(self.campaign.history().len(), false);
            self.restored_initial = Some(matched);
        }
        frame.validate_shape(self.campaign.history())?;
        self.dispatcher.restore_evaluated(actual)?;
        self.iteration = frame.updates.0;
        self.reset_since_notice = false;
        self.dirty = true;
        // Models/sampler trajectories are rebuilt from measured data. The
        // lifetime counters, pending measured prefix and original phase stay.
        self.update_anchor(cooperation).await?;
        Ok(())
    }
}
