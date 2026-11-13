//! Completed NUPS polls share original evaluations and accepted generations.
use super::*;
impl Engine {
    pub(super) fn poll_state(controls: &Controls, mesh: f64) -> BuiltinResult<PollState> {
        Ok(match controls.polling {
            super::super::controls::Polling::Classic(_) => PollState::Classic { refinements: None },
            super::super::controls::Polling::Nonuniform(algorithm) => {
                PollState::Nonuniform(Box::new(super::super::nups::Driver::new(
                    algorithm,
                    controls.equalities,
                    mesh,
                    controls.binding,
                    controls.constraint_tolerance,
                )?))
            }
        })
    }
    pub(super) async fn advance_nonuniform(
        &mut self,
        geometry: &Linear,
        controls: &Controls,
        evaluator: &mut Evaluator<'_>,
        deadline: &Deadline,
    ) -> BuiltinResult<Event> {
        let PollState::Nonuniform(driver) = &mut self.polling else {
            return Err(invalid("missing nonuniform driver"));
        };
        let Some(frame) = driver
            .frame(
                &self.point,
                geometry,
                self.mesh.size(),
                &self.scaling,
                controls.binding,
                evaluator.cooperation,
            )
            .await?
        else {
            return Ok(Event::Stopped(Stop::Numerical));
        };
        let started = evaluator.state.objective_calls;
        let augmented = &self.augmented;
        let outcome = poll::nonuniform(
            &self.point,
            geometry,
            &self.mesh,
            frame,
            controls.complete_poll,
            controls.constraint_tolerance,
            deadline,
            async |point, cooperation| match augmented {
                Some(augmented) => augmented.merit(point, cooperation).await,
                None => Ok(Some(point.f)),
            },
            evaluator,
        )
        .await?;
        let event = self
            .transition_nonuniform(
                outcome,
                false,
                evaluator.state.objective_calls - started,
                geometry,
                controls,
                evaluator,
            )
            .await?;
        self.record(event)
    }
    pub(super) async fn transition_nonuniform(
        &mut self,
        outcome: poll::Outcome,
        search: bool,
        points: usize,
        geometry: &Linear,
        controls: &Controls,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<Event> {
        let improved = outcome
            .improved
            .as_ref()
            .map(|improvement| improvement.merit);
        if let Some(improvement) = outcome.improved {
            self.accept(improvement.point, Some(improvement.merit), evaluator)
                .await?;
            self.method = if search { Method::Search } else { Method::Poll };
        }
        match outcome.completion {
            Completion::Budget => return Ok(Event::Stopped(Stop::Limit)),
            Completion::Time => return Ok(Event::Stopped(Stop::Time)),
            Completion::Numerical => return Ok(Event::Stopped(Stop::Numerical)),
            Completion::Finished | Completion::Unrepresentable => {}
        }
        let PollState::Nonuniform(driver) = &mut self.polling else {
            return Err(invalid("missing nonuniform transition"));
        };
        let update = if search {
            driver.searched()?;
            // A successful auxiliary search does not attribute progress or
            // change polarity in either NUPS polling phase.
            if improved.is_some() {
                self.mesh.nonuniform_update(2.0, false)
            } else {
                mesh::Update::Continue
            }
        } else if outcome.completion == Completion::Unrepresentable && improved.is_none() {
            if driver.unrepresentable()? {
                mesh::Update::Unrepresentable
            } else {
                mesh::Update::Continue
            }
        } else {
            driver.completed(improved, points, &mut self.mesh)?
        };
        if improved.is_none() {
            self.method = Method::Refine;
        }
        let covered = driver.covered();
        match update {
            mesh::Update::Numerical => return Ok(Event::Stopped(Stop::Numerical)),
            mesh::Update::Unrepresentable => {
                if improved.is_some() {
                    return Ok(Event::Stopped(Stop::Numerical));
                }
                return Ok(Event::Stopped(
                    if self.feasible(geometry, controls, evaluator).await? {
                        Stop::Machine
                    } else {
                        Stop::Numerical
                    },
                ));
            }
            _ => {}
        }
        if self.augmented.is_some() {
            return self.finish_inner(update, controls, evaluator).await;
        }
        self.finish_plain(geometry, controls, evaluator).await?;
        if improved.is_none()
            && update == mesh::Update::BelowTolerance
            && self.feasible(geometry, controls, evaluator).await?
        {
            return Ok(Event::Stopped(Stop::Mesh));
        }
        if improved.is_none()
            && covered
            && self.mesh.size() < controls.step_tolerance
            && self.feasible(geometry, controls, evaluator).await?
        {
            if self
                .change
                .is_some_and(|change| change.x < controls.step_tolerance)
            {
                return Ok(Event::Stopped(Stop::Step));
            }
            if self
                .change
                .is_some_and(|change| change.f < controls.function_tolerance)
            {
                return Ok(Event::Stopped(Stop::Function));
            }
        }
        Ok(Event::Iteration)
    }
}
