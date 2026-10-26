//! Accepted-point moves and completed subproblem transitions are separate
//! from poll generation. Partial polls cannot authorize contraction/stopping.
use super::*;

pub(super) fn parameters(controls: &Controls) -> augmented::Parameters {
    augmented::Parameters {
        initial_penalty: controls.initial_penalty,
        penalty_factor: controls.penalty_factor,
        mesh_tolerance: controls.mesh.tolerance,
        constraint_tolerance: controls.constraint_tolerance,
    }
}
impl Engine {
    pub(super) async fn accept(
        &mut self,
        point: Point,
        merit: Option<poll::MeritChange>,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<()> {
        let old = std::mem::replace(&mut self.point, point);
        if let Some(augmented) = &self.augmented {
            let mut change =
                state::measure(&old.x, old.f, &self.point, evaluator.cooperation).await?;
            change.f = if let Some(merit) = merit {
                (merit.after - merit.before).abs()
            } else {
                match (
                    augmented.merit(&old, evaluator.cooperation).await?,
                    augmented.merit(&self.point, evaluator.cooperation).await?,
                ) {
                    (Some(old), Some(new)) => (new - old).abs(),
                    _ => f64::INFINITY,
                }
            };
            self.inner_change = Some(change);
            if self.anchor.is_none() {
                self.anchor = Some(Anchor::take(old));
            }
        } else {
            self.change =
                Some(state::measure(&old.x, old.f, &self.point, evaluator.cooperation).await?);
        }
        Ok(())
    }

    pub(super) async fn finish_plain(
        &mut self,
        geometry: &Linear,
        controls: &Controls,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<Event> {
        self.iterations = self
            .iterations
            .checked_add(1)
            .ok_or_else(|| invalid("iteration count overflow"))?;
        // An unconstrained logarithmic scale is fixed at the initial point.
        // Linear constraints recompute only at accepted major generations.
        if controls.scale_mesh
            && geometry.matrix.m > 0
            && matches!(self.method, Method::Search | Method::Poll)
        {
            self.scaling = Scaling::logarithmic(&self.point.x, evaluator.cooperation).await?;
        }
        Ok(Event::Iteration)
    }

    pub(super) async fn transition(
        &mut self,
        outcome: poll::Outcome,
        search: bool,
        points: usize,
        geometry: &Linear,
        controls: &Controls,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<Event> {
        if matches!(self.polling, PollState::Nonuniform(_)) {
            return self
                .transition_nonuniform(outcome, search, points, geometry, controls, evaluator)
                .await;
        }
        let improved = outcome.improved.is_some();
        if let Some(improvement) = outcome.improved {
            match improvement.direction {
                Some(poll::History::New(direction)) => self.previous_direction = Some(direction),
                Some(poll::History::Previous) => {}
                None => self.previous_direction = None,
            }
            self.accept(improvement.point, None, evaluator).await?;
            self.method = if search { Method::Search } else { Method::Poll };
        }
        match outcome.completion {
            Completion::Budget => return Ok(Event::Stopped(Stop::Limit)),
            Completion::Time => return Ok(Event::Stopped(Stop::Time)),
            Completion::Numerical => return Ok(Event::Stopped(Stop::Numerical)),
            Completion::Unrepresentable => {
                return Ok(Event::Stopped(
                    if self.feasible(geometry, controls, evaluator).await? {
                        Stop::Machine
                    } else {
                        Stop::Numerical
                    },
                ));
            }
            Completion::Finished => {}
        }
        if improved {
            let update = if let Some(method) = controls
                .search
                .filter(|_| search)
                .and_then(super::super::search::Search::poll)
            {
                self.mesh.successful_search(method)
            } else {
                self.mesh.successful()
            };
            if matches!(
                update,
                mesh::Update::Unrepresentable | mesh::Update::Numerical
            ) {
                return Ok(Event::Stopped(Stop::Numerical));
            }
            return if self.augmented.is_some() {
                Ok(Event::Inner)
            } else {
                self.finish_plain(geometry, controls, evaluator).await
            };
        }
        self.method = Method::Refine;
        let orientation = self.mesh.sign();
        let update = self.mesh.unsuccessful(
            Controls::basis(
                controls
                    .polling
                    .method()
                    .ok_or_else(|| invalid("missing classic poll method"))?,
            ),
            geometry.equalities > 0 || !outcome.rotatable,
        );
        if update == mesh::Update::Numerical {
            return Ok(Event::Stopped(Stop::Numerical));
        }
        if orientation != self.mesh.sign() {
            self.method = Method::Rotate;
        }
        if update == mesh::Update::Unrepresentable {
            return Ok(Event::Stopped(
                if self.feasible(geometry, controls, evaluator).await? {
                    Stop::Machine
                } else {
                    Stop::Numerical
                },
            ));
        }
        if self.augmented.is_some() {
            return self.finish_inner(update, controls, evaluator).await;
        }
        // Keep the measured change between the two most recent accepted
        // best points. An unsuccessful poll does not manufacture an earlier
        // best point, nor overwrite a genuine measured displacement with 0.
        self.finish_plain(geometry, controls, evaluator).await?;
        if self.feasible(geometry, controls, evaluator).await? {
            if update == mesh::Update::BelowTolerance {
                return Ok(Event::Stopped(Stop::Mesh));
            }
            if !self.mesh.adaptive() && self.mesh.size() < controls.step_tolerance {
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
        }
        Ok(Event::Iteration)
    }

    pub(super) async fn finish_inner(
        &mut self,
        update: mesh::Update,
        controls: &Controls,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<Event> {
        if update != mesh::Update::BelowTolerance {
            return Ok(Event::Inner);
        }
        let moved = self.anchor.is_some();
        self.change = Some(match self.anchor.take() {
            Some(anchor) => anchor.change(&self.point, evaluator.cooperation).await?,
            None => Change::default(),
        });
        let augmented = self
            .augmented
            .as_mut()
            .expect("selected nonlinear controller");
        let result = augmented
            .completed_inner(&self.point, evaluator.cooperation)
            .await?;
        self.iterations = self
            .iterations
            .checked_add(1)
            .ok_or_else(|| invalid("iteration count overflow"))?;
        let Some(result) = result else {
            return Ok(Event::Stopped(Stop::Numerical));
        };
        self.method = match result {
            augmented::Update::Converged => return Ok(Event::Stopped(Stop::Mesh)),
            augmented::Update::Multipliers => Method::Multipliers,
            augmented::Update::Penalty => Method::Penalty,
        };
        // Freeze logarithmic scaling during each AL subproblem; refreshing it
        // at an outer transition does not change any evaluated original x.
        if controls.scale_mesh && moved {
            self.scaling = Scaling::logarithmic(&self.point.x, evaluator.cooperation).await?;
        } else if !controls.scale_mesh {
            self.scaling = Scaling::unit(self.point.x.len());
        }
        let mut parameters = controls.mesh;
        parameters.tolerance = augmented.inner_tolerance();
        self.mesh = match controls.polling.method() {
            Some(method) => Mesh::for_method(parameters, method, self.point.x.len()),
            None => Mesh::nonuniform(parameters),
        }
        .ok_or_else(|| invalid("invalid nonlinear mesh parameters"))?;
        if matches!(self.polling, PollState::Nonuniform(_)) {
            self.polling = Self::poll_state(controls, self.mesh.size())?;
        }
        self.inner_iterations = 0;
        self.inner_change = None;
        Ok(Event::Iteration)
    }
}
