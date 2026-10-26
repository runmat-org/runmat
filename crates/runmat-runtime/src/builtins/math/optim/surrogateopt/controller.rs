//! Original-owner construction/search/reset controller. Public observers run
//! between returned measured-batch events, never inside numerical probes.
use super::{
    affine::{self, Coordinates},
    campaign::{Campaign, Provenance},
    construction_phase::{self, Construction},
    dispatch::{Commit, Dispatcher},
    domain::BoxGeometry,
    gss::Geometry,
    invalid,
    model_phase::{self, Frozen, Refresh},
    observation::{Objective, ObjectiveKind},
    parameters::Parameters,
    queue::Origin,
    sampling::Problem,
};
use crate::{
    builtins::math::optim::{
        common::{solver_timing::SolverTiming, ScalarCooperation},
        fmincon::internal::Limits,
        intlinprog::feasibility::Domain,
    },
    call::descriptor::RepeatableCallable,
    execution::{CallbackBatch, RootedValue},
    BuiltinResult,
};
mod adaptive;
mod checkpoint;
pub(super) use checkpoint::Quiescence;
mod initial;
mod transitions;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Stop {
    FunctionLimit,
    TimeLimit,
    ObjectiveLimit,
    Observer,
    TooFewSites,
    LinearInfeasible,
    Unique,
}
pub(super) enum Event {
    Iteration,
    Done(Stop),
}
enum Mode {
    Unique,
    Construct(Construction),
    Search(crate::builtins::math::optim::common::retained_control::Retained<Frozen>),
}
pub(super) struct Setup<'a> {
    pub domain: Domain,
    pub source: &'a RootedValue,
    pub selected: &'a RepeatableCallable,
    pub workers: Option<&'a CallbackBatch>,
    pub clock: SolverTiming,
    pub parameters: Parameters,
    pub linear_tolerance: f64,
    pub initial: Option<super::initial::Initial>,
}
pub(super) struct Controller<'a> {
    domain: Domain,
    geometry: BoxGeometry,
    axes: Vec<usize>,
    gss: Geometry,
    problem: Problem,
    source: &'a RootedValue,
    pub(super) clock: SolverTiming,
    parameters: Parameters,
    pub(super) campaign: Campaign,
    dispatcher: Dispatcher<'a>,
    mode: Mode,
    anchor: Option<Vec<f64>>,
    dirty: bool,
    stop: Option<Stop>,
    pub(super) iteration: usize,
    reset_since_notice: bool,
    linear_tolerance: f64,
    limits: Limits,
    initial: Option<super::initial::Initial>,
    initial_reset: bool,
    restored_initial: Option<Vec<bool>>,
}
impl<'a> Controller<'a> {
    pub(super) async fn new(
        setup: Setup<'a>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(),
            super::NAME,
            "finite host-memory execution has no complete surrogate controller workspace bound",
        )?;
        let Setup {
            domain,
            source,
            selected,
            workers,
            clock,
            parameters,
            linear_tolerance,
            initial,
        } = setup;
        if !linear_tolerance.is_finite() || linear_tolerance <= 0.0 {
            return Err(invalid("invalid original linear feasibility tolerance"));
        }
        let geometry = BoxGeometry::new(&domain, cooperation).await?;
        let view = geometry.view(&domain)?;
        let variables = view.variables();
        let dimensions = view.dimensions();
        let problem = Problem::classify(&view, cooperation).await?;
        let (axes, mode) = match affine::coordinates(&view, cooperation).await? {
            Coordinates::Unique => (Vec::new(), Mode::Unique),
            Coordinates::Axes(axes) => (
                axes,
                Mode::Construct(Construction::new(parameters.minimum_surrogate)?),
            ),
            Coordinates::Unrepresentable => {
                return Err(numerical(
                    "original equality nullspace is not representable",
                ))
            }
        };
        let gss = Geometry::new(&view, cooperation).await?;
        let campaign = Campaign::new(variables, dimensions, parameters.batch)?;
        let dispatcher = Dispatcher::new(selected, workers, &parameters, variables, cooperation)?;
        let limits = Limits {
            iterations: 100,
            evaluations: variables
                .checked_mul(1000)
                .ok_or_else(|| invalid("internal refinement extent overflow"))?,
            feasibility: linear_tolerance,
            optimality: 1e-6,
            step: 1e-10,
        };
        Ok(Self {
            domain,
            geometry,
            axes,
            gss,
            problem,
            source,
            clock,
            parameters,
            campaign,
            dispatcher,
            mode,
            anchor: None,
            dirty: false,
            stop: None,
            iteration: 0,
            reset_since_notice: false,
            linear_tolerance,
            limits,
            initial,
            initial_reset: false,
            restored_initial: None,
        })
    }
    pub(super) fn counts(&self) -> (usize, usize) {
        self.dispatcher.counts()
    }
    pub(super) fn reset_since_notice(&self) -> bool {
        self.reset_since_notice
    }
    pub(super) fn acknowledge_notice(&mut self) {
        self.reset_since_notice = false;
    }
    pub(super) fn constraint_tolerance(&self) -> f64 {
        self.parameters.nonlinear_tolerance
    }
    pub(super) fn request_stop(&mut self) {
        self.stop = Some(Stop::Observer);
    }
    pub(super) async fn advance(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Event> {
        loop {
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
                initial::Progress::Committed => return Ok(Event::Iteration),
                initial::Progress::Continue => continue,
                initial::Progress::Unavailable => {}
            }
            self.check_limits();
            if let Some(stop) = self.stop {
                self.dispatcher.reset_unsubmitted();
                if self.dispatcher.in_flight() != 0 {
                    let commit = self
                        .dispatcher
                        .next(&mut self.campaign, cooperation)
                        .await?;
                    if self.committed(commit, cooperation).await? && stop != Stop::Observer {
                        return Ok(Event::Iteration);
                    }
                    continue;
                }
                if self.campaign.has_pending() {
                    let update = self.campaign.flush(cooperation).await?;
                    let commit = Commit {
                        points: 0,
                        updates: 1,
                        feasibility_reset: update.feasibility_reset,
                    };
                    if self.committed(commit, cooperation).await? && stop != Stop::Observer {
                        return Ok(Event::Iteration);
                    }
                    continue;
                }
                // Accepted completions can improve final best. A real final
                // objective threshold outranks a time/evaluation boundary,
                // but never overrides an explicit observer stop.
                if stop != Stop::Observer && self.objective_limit() {
                    self.stop = Some(Stop::ObjectiveLimit);
                }
                return Ok(Event::Done(self.stop.expect("terminal condition")));
            }
            if matches!(&self.mode, Mode::Unique) {
                return self.unique(cooperation).await;
            }
            let mut sites = None;
            if !self.campaign.has_pending()
                && (self.dirty || matches!(&self.mode, Mode::Construct(_)))
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
                        continue;
                    }
                }
            }
            self.check_limits();
            if self.stop.is_some() {
                continue;
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
                    continue;
                }
                if self.committed(commit, cooperation).await? {
                    return Ok(Event::Iteration);
                }
                continue;
            }
            match &self.mode {
                Mode::Construct(_) => match self.construct(cooperation).await? {
                    construction_phase::Next::Point(point) => {
                        self.check_limits();
                        if self.stop.is_some() {
                            continue;
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
                            return Ok(Event::Iteration);
                        }
                    }
                    construction_phase::Next::NoNewSites => {
                        if self.dispatcher.in_flight() != 0 {
                            let commit = self
                                .dispatcher
                                .next(&mut self.campaign, cooperation)
                                .await?;
                            if self.committed(commit, cooperation).await? {
                                return Ok(Event::Iteration);
                            }
                        } else {
                            let commit = self
                                .dispatcher
                                .finish_construction(&mut self.campaign, cooperation)
                                .await?;
                            if commit.points != 0 {
                                if self.committed(commit, cooperation).await? {
                                    return Ok(Event::Iteration);
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
                            continue;
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
                            return Ok(Event::Iteration);
                        }
                    }
                    adaptive::Next::Reset => self.reset(cooperation).await?,
                    adaptive::Next::Stop(stop) => self.stop = Some(stop),
                },
                Mode::Unique => return Err(invalid("unique-point controller changed mode")),
            }
        }
    }
    fn check_limits(&mut self) {
        if self.stop.is_some() {
            return;
        }
        self.stop = if self.objective_limit() {
            Some(Stop::ObjectiveLimit)
        } else if self.counts().1 >= self.parameters.evaluations {
            Some(Stop::FunctionLimit)
        } else if self.parameters.time.reached(self.clock.elapsed()) {
            Some(Stop::TimeLimit)
        } else {
            None
        };
    }
    fn objective_limit(&self) -> bool {
        self.campaign.tracker().best().and_then(|index| self.campaign.history().trial(index).observation.measurement())
            .is_some_and(|value| value.nonlinear.violated == 0 && matches!(value.objective, Objective::Measured(value) if value < self.parameters.objective_limit))
    }
}
fn numerical(message: &str) -> crate::RuntimeError {
    crate::build_runtime_error(format!("surrogateopt: {message}"))
        .with_builtin(super::NAME)
        .with_identifier("RunMat:surrogateopt:NumericalFailure")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
