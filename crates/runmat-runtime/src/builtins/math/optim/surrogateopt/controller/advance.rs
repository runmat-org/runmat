//! Initial-row processing and search are separate steps in one continuation.
//! A callback from an initial row does not retain the search construction frame.
use super::*;
use std::{future::Future, ops::ControlFlow};

enum Phase {
    Initial,
    Search,
}
type State<'scope, 'source> = (
    &'scope mut Controller<'source>,
    &'scope mut ScalarCooperation,
    Phase,
);
type StepResult<'scope, 'source> = BuiltinResult<ControlFlow<Event, State<'scope, 'source>>>;

#[inline(always)]
fn step<'scope, 'source: 'scope>(
    (controller, cooperation, phase): State<'scope, 'source>,
) -> impl Future<Output = StepResult<'scope, 'source>> + 'scope {
    match phase {
        Phase::Initial => futures::future::Either::Left(controller.advance_initial(cooperation)),
        Phase::Search => futures::future::Either::Right(controller.advance_search(cooperation)),
    }
}

impl<'source> Controller<'source> {
    pub(in super::super) fn advance<'scope>(
        &'scope mut self,
        cooperation: &'scope mut ScalarCooperation,
    ) -> impl Future<Output = BuiltinResult<Event>> + 'scope + use<'scope, 'source> {
        crate::execution::loop_result((self, cooperation, Phase::Initial), step)
    }

    async fn advance_initial<'scope>(
        &'scope mut self,
        cooperation: &'scope mut ScalarCooperation,
    ) -> StepResult<'scope, 'source> {
        cooperation.completed(0).await?;
        self.clock.checked_elapsed()?;
        if !matches!(self.stop, Some(Stop::Observer | Stop::LinearInfeasible))
            && matches!(self.mode, Mode::Unique)
            && self.campaign.history().len() != 0
        {
            self.initial = None;
            self.stop = Some(if self.objective_limit() {
                Stop::ObjectiveLimit
            } else if self.campaign.tracker().best().is_some() {
                Stop::Unique
            } else {
                Stop::TooFewSites
            });
        }
        match self.initial(cooperation).await? {
            initial::Progress::Committed => Ok(ControlFlow::Break(Event::Iteration)),
            initial::Progress::Continue => {
                Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)))
            }
            initial::Progress::Unavailable => {
                Ok(ControlFlow::Continue((self, cooperation, Phase::Search)))
            }
        }
    }

    async fn advance_search<'scope>(
        &'scope mut self,
        cooperation: &'scope mut ScalarCooperation,
    ) -> StepResult<'scope, 'source> {
        self.check_limits();
        if let Some(stop) = self.stop {
            self.dispatcher.reset_unsubmitted();
            if self.dispatcher.in_flight() != 0 {
                let commit = self
                    .dispatcher
                    .next(&mut self.campaign, cooperation)
                    .await?;
                if self.committed(commit, cooperation).await? && stop != Stop::Observer {
                    return Ok(ControlFlow::Break(Event::Iteration));
                }
                return Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)));
            }
            if self.campaign.has_pending() {
                let update = self.campaign.flush(cooperation).await?;
                let commit = Commit {
                    points: 0,
                    updates: 1,
                    feasibility_reset: update.feasibility_reset,
                };
                if self.committed(commit, cooperation).await? && stop != Stop::Observer {
                    return Ok(ControlFlow::Break(Event::Iteration));
                }
                return Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)));
            }
            // Accepted completions can improve final best. A real final
            // objective threshold outranks a time/evaluation boundary,
            // but never overrides an explicit observer stop.
            if stop != Stop::Observer && self.objective_limit() {
                self.stop = Some(Stop::ObjectiveLimit);
            }
            return Ok(ControlFlow::Break(Event::Done(
                self.stop.expect("terminal condition"),
            )));
        }
        if matches!(&self.mode, Mode::Unique) {
            return self.unique(cooperation).await.map(ControlFlow::Break);
        }
        let mut sites = None;
        if !self.campaign.has_pending() && (self.dirty || matches!(&self.mode, Mode::Construct(_)))
        {
            let view = self.geometry.view(&self.domain)?;
            match model_phase::refresh(
                &self.campaign,
                &view,
                &self.axes,
                self.parameters.minimum_surrogate,
                cooperation,
            )
            .await?
            {
                Refresh::Ready(model) => {
                    if matches!(&self.mode, Mode::Construct(_)) {
                        self.dispatcher.reset_unsubmitted();
                    }
                    self.mode = Mode::Search(model);
                    self.dirty = false;
                    self.update_anchor(cooperation).await?;
                }
                Refresh::NeedMoreSites { sites: count } => {
                    if matches!(&self.mode, Mode::Search(_)) {
                        return Err(numerical("updated surrogate lost numerical poisedness"));
                    }
                    sites = Some(count);
                }
                Refresh::Undefined => {
                    return Err(numerical("surrogate factorization is nonfinite"))
                }
                Refresh::NoFunctions => {
                    self.reset(cooperation).await?;
                    return Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)));
                }
            }
        }
        self.check_limits();
        if self.stop.is_some() {
            return Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)));
        }
        self.dispatcher.pump(cooperation)?;
        if self.dispatcher.vacancy() == 0
            || sites.is_some_and(|sites| {
                sites < self.parameters.minimum_surrogate
                    && sites.saturating_add(self.dispatcher.planned(self.campaign.phase()))
                        >= self.parameters.minimum_surrogate
            }) && self.dispatcher.in_flight() != 0
        {
            let commit = self
                .dispatcher
                .next(&mut self.campaign, cooperation)
                .await?;
            if commit.points == 0 {
                self.stop = Some(Stop::FunctionLimit);
                return Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)));
            }
            if self.committed(commit, cooperation).await? {
                return Ok(ControlFlow::Break(Event::Iteration));
            }
            return Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)));
        }
        match &self.mode {
            Mode::Construct(_) => match self.construct(cooperation).await? {
                construction_phase::Next::Point(point) => {
                    self.check_limits();
                    if self.stop.is_some() {
                        return Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)));
                    }
                    if self.anchor.is_none() {
                        self.keep_anchor(&point, cooperation).await?;
                    }
                    let commit = self
                        .dispatcher
                        .submit(
                            point,
                            Provenance {
                                phase: self.campaign.phase(),
                                origin: Origin::Random,
                            },
                            &mut self.campaign,
                            cooperation,
                        )
                        .await?;
                    if self.committed(commit, cooperation).await? {
                        return Ok(ControlFlow::Break(Event::Iteration));
                    }
                }
                construction_phase::Next::NoNewSites => {
                    if self.dispatcher.in_flight() != 0 {
                        let commit = self
                            .dispatcher
                            .next(&mut self.campaign, cooperation)
                            .await?;
                        if self.committed(commit, cooperation).await? {
                            return Ok(ControlFlow::Break(Event::Iteration));
                        }
                    } else {
                        let commit = self
                            .dispatcher
                            .finish_construction(&mut self.campaign, cooperation)
                            .await?;
                        if commit.points != 0 {
                            if self.committed(commit, cooperation).await? {
                                return Ok(ControlFlow::Break(Event::Iteration));
                            }
                        } else {
                            self.stop = Some(Stop::TooFewSites);
                        }
                    }
                }
                construction_phase::Next::LinearInfeasible => {
                    self.stop = Some(Stop::LinearInfeasible)
                }
                construction_phase::Next::TimeLimit => self.stop = Some(Stop::TimeLimit),
            },
            Mode::Search(_) => match self.adaptive(cooperation).await? {
                adaptive::Next::Point(point) => {
                    self.check_limits();
                    if self.stop.is_some() {
                        return Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)));
                    }
                    self.campaign.selected();
                    let commit = self
                        .dispatcher
                        .submit(
                            point,
                            Provenance {
                                phase: self.campaign.phase(),
                                origin: Origin::Adaptive,
                            },
                            &mut self.campaign,
                            cooperation,
                        )
                        .await?;
                    if self.committed(commit, cooperation).await? {
                        return Ok(ControlFlow::Break(Event::Iteration));
                    }
                }
                adaptive::Next::Reset => self.reset(cooperation).await?,
                adaptive::Next::Stop(stop) => self.stop = Some(stop),
            },
            Mode::Unique => return Err(invalid("unique-point controller changed mode")),
        }

        Ok(ControlFlow::Continue((self, cooperation, Phase::Initial)))
    }
}
