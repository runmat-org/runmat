//! Measured events and phase/unique-point transitions; no source replay.
use super::*;
use crate::builtins::math::optim::surrogateopt::proposal::{self, Prepared};

impl Controller<'_> {
    pub(super) async fn committed(
        &mut self,
        commit: Commit,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if commit.updates == 0 {
            return Ok(false);
        }
        self.iteration = self
            .iteration
            .checked_add(1)
            .ok_or_else(|| invalid("surrogate iteration extent overflow"))?;
        self.dirty = true;
        self.update_anchor(cooperation).await?;
        self.check_limits();
        if commit.feasibility_reset {
            if self.initial.is_some() {
                self.initial_reset = true;
            } else if self.stop.is_none() {
                self.reset(cooperation).await?;
            }
        }
        Ok(true)
    }
    pub(super) async fn construct(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<construction_phase::Next> {
        let Mode::Construct(construction) = &mut self.mode else {
            return Err(invalid("construction entered outside its phase"));
        };
        let remaining = self.dispatcher.remaining();
        construction
            .next(
                construction_phase::Owners {
                    campaign: &self.campaign,
                    dispatcher: &self.dispatcher,
                    domain: &mut self.domain,
                    geometry: &self.geometry,
                    gss: &self.gss,
                },
                construction_phase::Controls {
                    parameters: &self.parameters,
                    clock: &self.clock,
                    remaining,
                    anchor: self.anchor.as_deref(),
                    linear_tolerance: self.linear_tolerance,
                },
                cooperation,
            )
            .await
    }
    pub(super) async fn reset(&mut self, cooperation: &mut ScalarCooperation) -> BuiltinResult<()> {
        self.campaign.flush(cooperation).await?;
        self.dispatcher.reset_unsubmitted();
        self.campaign.reset()?;
        self.reset_since_notice = true;
        self.mode = Mode::Construct(Construction::new(self.parameters.minimum_surrogate)?);
        self.dirty = false;
        Ok(())
    }
    pub(super) async fn keep_anchor(
        &mut self,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if point.len() != self.domain.bounds().0.len() {
            return Err(invalid("working anchor changed original extent"));
        }
        if self.anchor.is_none() {
            self.anchor = Some(runmat_value::exact_capacity_vec(point.len()).map_err(allocation)?);
        }
        let anchor = self.anchor.as_mut().expect("working anchor storage");
        anchor.clear();
        for &value in point {
            anchor.push(value);
            cooperation.completed(1).await?;
        }
        Ok(())
    }
    pub(super) async fn update_anchor(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let Some(index) = self.campaign.tracker().incumbent() else {
            return Ok(());
        };
        let point = &self.campaign.history().trial(index).point;
        if self.anchor.is_none() {
            self.anchor =
                Some(runmat_value::exact_capacity_vec(point.variables()).map_err(allocation)?);
        }
        let anchor = self.anchor.as_mut().expect("working anchor storage");
        anchor.clear();
        for column in 0..point.variables() {
            anchor.push(point.coordinate(column));
            cooperation.completed(1).await?;
        }
        Ok(())
    }
    pub(super) async fn unique(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Event> {
        let mut point =
            runmat_value::exact_capacity_vec(self.domain.bounds().0.len()).map_err(allocation)?;
        {
            let view = self.geometry.view(&self.domain)?;
            let mut free = 0;
            for column in 0..view.variables() {
                let (lower, upper) = view.effective_bounds(column);
                let value = if view.free_columns().get(free) == Some(&column) {
                    let value = view.interpolate_coordinate(free, 0.5)?;
                    free += 1;
                    if view.integers().binary_search(&column).is_ok() {
                        value.round().max(lower).min(upper)
                    } else {
                        value
                    }
                } else {
                    lower
                };
                point.push(value);
                cooperation.completed(1).await?;
            }
        }
        let point =
            match proposal::prepare(point, &mut self.domain, self.linear_tolerance, cooperation)
                .await?
            {
                Prepared::Point(point) => point,
                Prepared::LinearInfeasible => {
                    self.stop = Some(Stop::LinearInfeasible);
                    return Ok(Event::Done(Stop::LinearInfeasible));
                }
                Prepared::TimeLimit => {
                    self.stop = Some(Stop::TimeLimit);
                    return Ok(Event::Done(Stop::TimeLimit));
                }
                Prepared::Unverified => {
                    return Err(numerical(
                        "unique original-domain point could not be verified",
                    ))
                }
            };
        let commit = self
            .dispatcher
            .unique(
                point,
                Provenance {
                    phase: self.campaign.phase(),
                    origin: Origin::Random,
                },
                &mut self.campaign,
                cooperation,
            )
            .await?;
        self.stop = Some(if self.campaign.tracker().best().is_some() {
            Stop::Unique
        } else {
            Stop::TooFewSites
        });
        if self.committed(commit, cooperation).await? {
            Ok(Event::Iteration)
        } else {
            Ok(Event::Done(self.stop.expect("unique completion")))
        }
    }
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> crate::RuntimeError {
    crate::build_runtime_error("surrogateopt: working anchor allocation failed")
        .with_builtin(super::super::NAME)
        .with_source(source)
        .build()
}
