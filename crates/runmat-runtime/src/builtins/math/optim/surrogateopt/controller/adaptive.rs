//! Frozen-model selection and real host refinements before one expensive call.
use super::super::{
    merit::{self, Criterion, NoCandidate, Phase, Rank, Selection, SelectionControls},
    oracle::{Controls, ModelView},
    refinement::{self, Session},
    sampler_pool::{self, Generated},
    schedule::{self, Applicability},
    spacing,
};
use super::*;
pub(super) enum Next {
    Point(Vec<f64>),
    Reset,
    Stop(Stop),
}

impl Controller<'_> {
    pub(super) async fn adaptive(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Next> {
        let phase = self.merit_phase()?;
        let anchor = self
            .anchor
            .as_deref()
            .ok_or_else(|| numerical("adaptive model has no actual incumbent"))?;
        let generated = sampler_pool::generate(
            sampler_pool::Request {
                kind: self.campaign.search().sampler(self.problem)?,
                phase,
                anchor,
                scale: self.campaign.search().scale(),
                parameters: &self.parameters,
                clock: &self.clock,
                linear_tolerance: self.linear_tolerance,
            },
            &self.campaign,
            &mut self.domain,
            &self.geometry,
            &self.gss,
            cooperation,
        )
        .await?;
        let selected = match generated {
            Generated::TimeLimit => return Ok(Next::Stop(Stop::TimeLimit)),
            Generated::LinearInfeasible => return Ok(Next::Stop(Stop::LinearInfeasible)),
            Generated::Unrepresentable => {
                return Err(numerical("sampler equality frame is not representable"))
            }
            Generated::Blocked => None,
            Generated::Ready(candidates) => {
                let Mode::Search(frozen) = &self.mode else {
                    return Err(invalid("adaptive sampler lost its frozen model"));
                };
                match frozen
                    .select(
                        &self.campaign,
                        candidates.domain(),
                        &self.axes,
                        &candidates,
                        SelectionControls {
                            phase,
                            weight: self.campaign.search().weight(),
                            constraint_tolerance: self.parameters.nonlinear_tolerance,
                            min_distance: self.parameters.sample_distance,
                        },
                        cooperation,
                    )
                    .await?
                {
                    Selection::Point(selected) => Some((
                        candidates.take(selected.index)?,
                        selected.criterion,
                        selected.rank,
                    )),
                    Selection::Reset(NoCandidate::Undefined) => {
                        return Err(numerical("all adaptive model predictions are undefined"))
                    }
                    Selection::Reset(NoCandidate::TooClose | NoCandidate::NoFeasiblePrediction) => {
                        None
                    }
                }
            }
        };
        let before_reset = selected.is_none();
        let applicable = Applicability {
            problem: self.problem,
            nonlinear: self
                .campaign
                .history()
                .schema()
                .is_some_and(|schema| schema.inequalities != 0),
            linear: self.domain.original_linear().rows() != 0,
            parallel: self.dispatcher.parallel(),
        };
        let Some(request) = self.campaign.refinement(applicable, before_reset) else {
            return Ok(selected.map_or(Next::Reset, |(point, _, _)| Next::Point(point)));
        };
        let kind = self.campaign.started_refinement(request)?;
        let criterion = match &selected {
            Some((_, criterion, _)) => *criterion,
            None => {
                let Mode::Search(frozen) = &self.mode else {
                    return Err(invalid("reset refinement lost its frozen model"));
                };
                let view = self.geometry.view(&self.domain)?;
                let values = frozen
                    .views(&self.campaign, &view, &self.axes, cooperation)
                    .await?;
                merit::reset_criterion(
                    phase,
                    &values.training,
                    self.campaign.search().weight(),
                    cooperation,
                )
                .await?
            }
        };
        let refined = self
            .refine(
                kind,
                criterion,
                selected.as_ref().map(|(_, _, rank)| *rank),
                cooperation,
            )
            .await?;
        if self.parameters.time.reached(self.clock.elapsed()) {
            return Ok(Next::Stop(Stop::TimeLimit));
        }
        Ok(match refined {
            Some(point) => Next::Point(point),
            None => selected.map_or(Next::Reset, |(point, _, _)| Next::Point(point)),
        })
    }
    fn merit_phase(&self) -> BuiltinResult<Phase> {
        let schema = self
            .campaign
            .history()
            .schema()
            .ok_or_else(|| numerical("adaptive model has no actual observation schema"))?;
        let inequality_start = usize::from(schema.objective == ObjectiveKind::Present);
        let feasible = self
            .campaign
            .tracker()
            .incumbent()
            .and_then(|index| {
                self.campaign
                    .history()
                    .trial(index)
                    .observation
                    .measurement()
            })
            .is_some_and(|value| value.nonlinear.violated == 0);
        if schema.objective == ObjectiveKind::Present && feasible {
            Ok(Phase::Objective {
                objective_lane: 0,
                inequality_start,
            })
        } else if schema.inequalities != 0 {
            Ok(Phase::Feasibility { inequality_start })
        } else {
            Err(numerical(
                "function-free feasibility observation requires a reset",
            ))
        }
    }
    async fn refine(
        &mut self,
        kind: schedule::Refinement,
        criterion: Criterion,
        preceding: Option<Rank>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        let Mode::Search(frozen) = &self.mode else {
            return Err(invalid("host refinement requires a frozen search model"));
        };
        let result = {
            let view = self.geometry.view(&self.domain)?;
            let values = frozen
                .views(&self.campaign, &view, &self.axes, cooperation)
                .await?;
            let schema = self
                .campaign
                .history()
                .schema()
                .ok_or_else(|| numerical("host refinement has no measured schema"))?;
            let mut session = Session::new(
                &self.domain,
                self.source,
                ModelView {
                    model: frozen.model(),
                    training: &values.training,
                    evaluated: &values.evaluated,
                    domain: self.geometry.view(&self.domain)?,
                    criterion,
                    inequality_start: usize::from(schema.objective == ObjectiveKind::Present),
                },
                Controls {
                    tolerance: self.parameters.nonlinear_tolerance,
                    clock: &self.clock,
                    time_limit: self.parameters.time,
                },
                self.limits,
                cooperation,
            )
            .await?;
            let anchor = self
                .anchor
                .as_deref()
                .ok_or_else(|| numerical("host refinement has no actual anchor"))?;
            let Some(result) = refinement::run(
                kind,
                self.geometry.view(&self.domain)?,
                anchor,
                self.campaign.search().scale(),
                &mut session,
                cooperation,
            )
            .await?
            else {
                return Ok(None);
            };
            if preceding.is_some_and(|rank| {
                !session.beats(&result, rank, self.parameters.nonlinear_tolerance)
            }) {
                return Ok(None);
            }
            result
        };
        // Preserve the exact modeled point/result pair. A failed original
        // membership check discards this optional refinement; do not repair
        // it and then transfer the old merit to changed coordinates.
        if !self
            .domain
            .classify(&result.point, self.linear_tolerance, cooperation)
            .await?
            .feasible
        {
            return Ok(None);
        }
        let view = self.geometry.view(&self.domain)?;
        let evaluated = frozen
            .evaluated_view(&self.campaign, &view, &self.axes, cooperation)
            .await?;
        if !spacing::admissible(
            &result.point,
            &view,
            &evaluated,
            self.parameters.sample_distance,
            cooperation,
        )
        .await?
        {
            return Ok(None);
        }
        Ok(Some(result.point))
    }
}
