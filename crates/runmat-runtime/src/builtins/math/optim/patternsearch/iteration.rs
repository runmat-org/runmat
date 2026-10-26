//! Classic feasible search/poll controller. Nonlinear major iterations are
//! completed ALPS subproblems; inner polls consume the same objective budget.
use super::{
    augmented::{self, Augmented},
    controls::Controls,
    evaluation::{Evaluator, Point},
    invalid,
    linear::Linear,
    mesh::{self, Mesh},
    poll::{self, Completion},
    scaling::Scaling,
    state::{self, Anchor, Change, Stop},
    timing::Deadline,
};
use crate::BuiltinResult;
use runmat_builtins::OptimPatternPoll;
mod transition;
mod auxiliary;
pub(super) mod selected_search;
mod nonuniform;
enum PollState {
    Classic {
        refinements: Option<super::mads::Refinements>,
    },
    Nonuniform(Box<super::nups::Driver>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Method {
    Initial,
    Search,
    Poll,
    Refine,
    Rotate,
    Multipliers,
    Penalty,
}
impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Self::Initial => "Initial point",
            Self::Search => "Successful Search",
            Self::Poll => "Successful Poll",
            Self::Refine => "Refine Mesh",
            Self::Rotate => "Refine Mesh\\Rotate",
            Self::Multipliers => "Update multipliers",
            Self::Penalty => "Increase penalty",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Event {
    /// Completed an inner nonlinear poll, not a completed outer iteration.
    Inner,
    Iteration,
    Stopped(Stop),
}
pub(super) struct Engine {
    pub point: Point,
    pub iterations: usize,
    pub inner_iterations: usize,
    pub method: Method,
    pub change: Option<Change>,
    pub inner_change: Option<Change>,
    pub mesh: Mesh,
    completed_passes: usize,
    search_window: Option<super::search::Window>,
    scaling: Scaling,
    scaled: bool,
    augmented: Option<Augmented>,
    anchor: Option<Anchor>,
    previous_direction: Option<Vec<f64>>,
    polling: PollState,
}
impl Engine {
    pub fn into_point(self) -> Point {
        self.point
    }
    pub async fn new(
        point: Point,
        controls: &Controls,
        nonlinear: bool,
        search_window: Option<super::search::Window>,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<Self> {
        let augmented = if nonlinear {
            Some(
                Augmented::new(
                    &point,
                    transition::parameters(controls),
                    evaluator.cooperation,
                )
                .await?,
            )
        } else {
            None
        };
        let mut parameters = controls.mesh;
        if let Some(augmented) = &augmented {
            parameters.tolerance = augmented.inner_tolerance();
        }
        let mesh = match controls.polling.method() {
            Some(method) => Mesh::for_method(parameters, method, point.x.len()),
            None => Mesh::nonuniform(parameters),
        }
        .ok_or_else(|| invalid("invalid mesh parameters"))?;
        let polling = Self::poll_state(controls, mesh.size())?;
        let scaling = if controls.scale_mesh {
            Scaling::logarithmic(&point.x, evaluator.cooperation).await?
        } else {
            Scaling::unit(point.x.len())
        };
        Ok(Self {
            point,
            iterations: 0,
            inner_iterations: 0,
            method: Method::Initial,
            completed_passes: 0,
            search_window,
            change: None,
            inner_change: None,
            mesh,
            scaling,
            scaled: controls.scale_mesh,
            augmented,
            anchor: None,
            previous_direction: None,
            polling,
        })
    }

    /// An output callback can change effective tolerances and transition
    /// factors. InitialMeshSize/InitialPenalty remain initialization controls.
    pub async fn reconfigure(
        &mut self,
        controls: &Controls,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<()> {
        if !controls.remember_direction {
            self.previous_direction = None;
        }
        let scale_changed = controls.scale_mesh != self.scaled;
        if scale_changed {
            self.scaling = if controls.scale_mesh {
                Scaling::logarithmic(&self.point.x, evaluator.cooperation).await?
            } else {
                Scaling::unit(self.point.x.len())
            };
            self.scaled = controls.scale_mesh;
        }
        if let Some(augmented) = &mut self.augmented {
            augmented.reconfigure(transition::parameters(controls))?;
        }
        let mut parameters = controls.mesh;
        if let Some(augmented) = &self.augmented {
            parameters.tolerance = augmented.inner_tolerance();
        }
        match controls.polling.method() {
            Some(method) => self
                .mesh
                .reconfigure_for(parameters, method, self.point.x.len())?,
            None => self.mesh.reconfigure_nonuniform(parameters)?,
        }
        match (&mut self.polling, controls.polling) {
            (PollState::Classic { .. }, super::controls::Polling::Classic(_)) => {}
            (PollState::Nonuniform(driver), super::controls::Polling::Nonuniform(algorithm))
                if driver.algorithm() == algorithm =>
            {
                driver.configure(
                    controls.binding,
                    controls.constraint_tolerance,
                    scale_changed,
                )
            }
            _ => self.polling = Self::poll_state(controls, self.mesh.size())?,
        }
        Ok(())
    }

    pub async fn advance(
        &mut self,
        geometry: &Linear,
        options: &super::options::Options,
        input: &super::input::Input<'_>,
        evaluator: &mut Evaluator<'_>,
        deadline: &Deadline,
    ) -> BuiltinResult<Event> {
        let controls = &options.controls;
        evaluator.cooperation.completed(0).await?;
        if self.iterations >= controls.iterations || evaluator.remaining() == 0 {
            return Ok(Event::Stopped(Stop::Limit));
        }
        if deadline.reached() {
            return Ok(Event::Stopped(Stop::Time));
        }
        let started = evaluator.state.objective_calls;
        // A separate poll-method search precedes polling. Identical methods
        // are not run twice, and unsuccessful search never contracts a mesh.
        if let Some(selected) = controls.search.filter(|search| {
            !controls
                .polling
                .method()
                .is_some_and(|method| search.redundant(method))
        }) {
            let outcome = match selected {
                super::search::Search::Poll(method) => Some(
                    self.run_poll(
                        geometry,
                        controls,
                        method,
                        controls.complete_search,
                        evaluator,
                    )
                    .await?,
                ),
                super::search::Search::Latin { iterations, points }
                    if self.completed_passes < iterations =>
                {
                    Some(
                        self.run_latin(geometry, controls, points, deadline, evaluator)
                            .await?,
                    )
                }
                super::search::Search::Latin { .. } => None,
                super::search::Search::Nelder { iterations }
                    if self.completed_passes < iterations =>
                {
                    if geometry.matrix.m > 0 || evaluator.context.constraints.is_some() {
                        return Err(invalid(
                            "searchneldermead supports unconstrained problems only",
                        ));
                    }
                    Some(
                        selected_search::admit(evaluator.cooperation.runtime().cloned(), || {
                            super::search::run_nelder(&self.point, options, deadline, evaluator)
                        })?
                        .await?,
                    )
                }
                super::search::Search::Nelder { .. } => None,
                super::search::Search::Genetic { iterations }
                    if self.completed_passes < iterations =>
                {
                    Some(
                        selected_search::admit(evaluator.cooperation.runtime().cloned(), || {
                            super::search::run_genetic(
                                &self.point,
                                options,
                                input,
                                geometry,
                                self.augmented.as_ref(),
                                deadline,
                                evaluator,
                            )
                        })?
                        .await?,
                    )
                }
                super::search::Search::Genetic { .. } => None,
                super::search::Search::Rbf => Some(
                    selected_search::admit(evaluator.cooperation.runtime().cloned(), || {
                        super::search::rbf::run(
                            &self.point,
                            geometry,
                            self.augmented.as_ref(),
                            controls.constraint_tolerance,
                            deadline,
                            evaluator,
                        )
                    })?
                    .await?,
                ),
                super::search::Search::Custom => Some(
                    selected_search::admit(evaluator.cooperation.runtime().cloned(), || {
                        self.run_custom(geometry, options, input, deadline, evaluator)
                    })?
                    .await?,
                ),
            };
            if let Some(outcome) = outcome {
                if outcome.improved.is_some() || outcome.completion != Completion::Finished {
                    let event = self
                        .transition(
                            outcome,
                            true,
                            evaluator.state.objective_calls - started,
                            geometry,
                            controls,
                            evaluator,
                        )
                        .await?;
                    return self.record(event);
                }
                if evaluator.remaining() == 0 {
                    return Ok(Event::Stopped(Stop::Limit));
                }
                if deadline.reached() {
                    return Ok(Event::Stopped(Stop::Time));
                }
            }
        }
        if matches!(self.polling, PollState::Nonuniform(_)) {
            return self
                .advance_nonuniform(geometry, controls, evaluator, deadline)
                .await;
        }
        let method = controls
            .polling
            .method()
            .ok_or_else(|| invalid("missing classic method"))?;
        let outcome = self
            .run_poll(
                geometry,
                controls,
                method,
                controls.complete_poll,
                evaluator,
            )
            .await?;
        let event = self
            .transition(
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

    fn record(&mut self, event: Event) -> BuiltinResult<Event> {
        if matches!(event, Event::Inner | Event::Iteration) {
            self.completed_passes = self
                .completed_passes
                .checked_add(1)
                .ok_or_else(|| invalid("search pass count overflow"))?;
        }
        if event == Event::Inner {
            self.inner_iterations = self
                .inner_iterations
                .checked_add(1)
                .ok_or_else(|| invalid("inner iteration count overflow"))?;
        }
        Ok(event)
    }

    pub async fn inner_objective(
        &self,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<Option<f64>> {
        match &self.augmented {
            Some(augmented) => augmented.merit(&self.point, evaluator.cooperation).await,
            None => Ok(Some(self.point.f)),
        }
    }

    async fn run_poll(
        &mut self,
        geometry: &Linear,
        controls: &Controls,
        method: OptimPatternPoll,
        complete: bool,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<poll::Outcome> {
        let augmented = &self.augmented;
        poll::poll(
            &self.point,
            geometry,
            &self.mesh,
            &self.scaling,
            method,
            controls.binding,
            complete,
            controls.constraint_tolerance,
            Controls::acceptance(method, augmented.is_some()),
            if method.mesh_adaptive() {
                runmat_builtins::OptimPatternOrder::Consecutive
            } else {
                controls
                    .polling
                    .classic()
                    .map_or(runmat_builtins::OptimPatternOrder::Consecutive, |classic| {
                        classic.order
                    })
            },
            self.previous_direction.as_deref(),
            controls.remember_direction,
            match &mut self.polling {
                PollState::Classic { refinements } => refinements,
                PollState::Nonuniform(_) => {
                    return Err(invalid("nonuniform controller cannot run a classic poll"))
                }
            },
            async |point, cooperation| match augmented {
                Some(augmented) => augmented.merit(point, cooperation).await,
                None => Ok(Some(point.f)),
            },
            evaluator,
        )
        .await
    }

    async fn feasible(
        &self,
        geometry: &Linear,
        controls: &Controls,
        evaluator: &mut Evaluator<'_>,
    ) -> BuiltinResult<bool> {
        Ok(geometry
            .feasible(
                &self.point.x,
                controls.constraint_tolerance,
                evaluator.cooperation,
            )
            .await?
            && state::nonlinear_violation(&self.point, evaluator.cooperation).await?
                <= controls.constraint_tolerance)
    }
}
