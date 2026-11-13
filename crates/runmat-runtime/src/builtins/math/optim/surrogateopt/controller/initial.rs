//! Supplied rows precede model construction. Every real missing-lane call
//! uses the already-selected scalar/vectorized/worker route and its original
//! allowance; premeasured rows never increment the source call count.
use super::*;
use crate::builtins::math::optim::surrogateopt::initial::Next;
pub(super) enum Progress {
    Committed,
    Continue,
    Unavailable,
}
impl Controller<'_> {
    pub(super) async fn initial(&mut self, c: &mut ScalarCooperation) -> BuiltinResult<Progress> {
        let Some(initial) = self.initial.as_ref() else {
            return Ok(Progress::Unavailable);
        };
        if self.stop.is_some() {
            return Ok(Progress::Unavailable);
        }
        if initial.done() {
            // Do not discard staged or authentic accepted initial rows merely
            // because enough earlier rows already fit a surrogate.
            self.check_limits();
            if self.stop.is_some() {
                return Ok(Progress::Unavailable);
            }
            let commit = self
                .dispatcher
                .finish_construction(&mut self.campaign, c)
                .await?;
            if commit.updates != 0 && self.committed(commit, c).await? {
                return Ok(Progress::Committed);
            }
            self.dispatcher.pump(c)?;
            if self.dispatcher.in_flight() != 0 {
                let commit = self.dispatcher.next(&mut self.campaign, c).await?;
                return Ok(if self.committed(commit, c).await? {
                    Progress::Committed
                } else {
                    Progress::Continue
                });
            }
            if self.campaign.has_pending() {
                let update = self.campaign.flush(c).await?;
                let commit = Commit {
                    points: 0,
                    updates: 1,
                    feasibility_reset: update.feasibility_reset,
                };
                self.committed(commit, c).await?;
                return Ok(Progress::Committed);
            }
            self.initial = None;
            self.restored_initial = None;
            if self.initial_reset {
                self.initial_reset = false;
                self.reset(c).await?;
            }
            return Ok(Progress::Continue);
        }
        // A complete supplied row can still determine the returned best at
        // MaxFunctionEvaluations=0. No RNG or source grant is invented for it.
        if self.dispatcher.vacancy() == 0 && self.dispatcher.in_flight() != 0 {
            self.check_limits();
            if self.stop.is_some() {
                return Ok(Progress::Unavailable);
            }
            let commit = self.dispatcher.next(&mut self.campaign, c).await?;
            self.dispatcher.pump(c)?;
            return Ok(if self.committed(commit, c).await? {
                Progress::Committed
            } else {
                Progress::Continue
            });
        }
        let next = self
            .initial
            .as_mut()
            .expect("retained initial rows")
            .next(
                self.campaign.history().schema(),
                &mut self.domain,
                self.linear_tolerance,
                if matches!(self.mode, Mode::Unique) {
                    super::super::initial::Discovery::AtUniqueSite
                } else {
                    super::super::initial::Discovery::WaitForNewSite
                },
                c,
            )
            .await?;
        match next {
            Next::Known(completed) => {
                if self
                    .skip_restored_initial(
                        completed.point.variables(),
                        |column| completed.point.coordinate(column),
                        c,
                    )
                    .await?
                {
                    return Ok(Progress::Continue);
                }
                let point = self.campaign.record_initial(completed, c).await?;
                self.dirty = true;
                if let Some(update) = point.update {
                    self.initial_reset |= update.feasibility_reset;
                    // Limits are checked only after all zero-call initial
                    // rows have been considered; normal advance handles the
                    // terminal boundary after the initial sequence finishes.
                    self.update_anchor(c).await?;
                    Ok(Progress::Committed)
                } else {
                    Ok(Progress::Continue)
                }
            }
            Next::Evaluate { point, seed } => {
                if self
                    .skip_restored_initial(point.len(), |column| point[column], c)
                    .await?
                {
                    return Ok(Progress::Continue);
                }
                self.check_limits();
                if self.stop.is_some() {
                    return Ok(Progress::Unavailable);
                }
                let provenance = Provenance {
                    phase: self.campaign.phase(),
                    origin: Origin::Initial,
                };
                let commit = if matches!(self.mode, Mode::Unique) {
                    self.dispatcher
                        .unique_seed(point, provenance, seed, &mut self.campaign, c)
                        .await?
                } else {
                    self.dispatcher
                        .submit_seed(point, provenance, seed, &mut self.campaign, c)
                        .await?
                };
                Ok(if self.committed(commit, c).await? {
                    Progress::Committed
                } else {
                    Progress::Continue
                })
            }
            Next::NeedSchema => Ok(Progress::Unavailable),
            Next::Done => Ok(Progress::Continue),
            Next::LinearInfeasible => {
                self.stop = Some(Stop::LinearInfeasible);
                Ok(Progress::Unavailable)
            }
            Next::TimeLimit => {
                self.stop = Some(Stop::TimeLimit);
                Ok(Progress::Unavailable)
            }
        }
    }
    async fn skip_restored_initial(
        &mut self,
        variables: usize,
        coordinate: impl Fn(usize) -> f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let Some(matched) = &mut self.restored_initial else {
            return Ok(false);
        };
        // Match each saved initial occurrence once. Duplicate initial rows
        // that were still unsubmitted at the saved cut retain their calls.
        for (row, used) in matched.iter_mut().enumerate() {
            let trial = self
                .campaign
                .history()
                .trial(super::super::history::TrialIndex(row));
            if *used || trial.origin != Origin::Initial || trial.point.variables() != variables {
                continue;
            }
            let mut equal = true;
            for column in 0..variables {
                equal &= coordinate(column).to_bits() == trial.point.coordinate(column).to_bits();
                c.completed(1).await?;
                if !equal {
                    break;
                }
            }
            if equal {
                *used = true;
                return Ok(true);
            }
        }
        Ok(false)
    }
}
