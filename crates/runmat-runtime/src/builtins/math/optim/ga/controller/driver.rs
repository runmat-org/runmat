//! Init, completed generations, AL interrupts and one final notification all
//! use the same search, timing origin and rooted observer generation.
use super::super::callbacks::Event;
use super::super::{
    common::solver_timing::SolverTiming, invalid, outcome::Termination, ScalarCooperation,
};
use super::super::{creation::factory::Factory, evaluation::source::Source, linear::Geometry};
use super::{
    observing::Observing,
    progress::Subproblem,
    reporting,
    search::{InitialStop, Retained, Search, Step},
};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::OptimGeneticObserverState as Phase;
use std::ops::ControlFlow;
mod setup;
#[cfg(test)]
mod tests;

pub(in crate::builtins::math::optim::ga) struct Finished<
    G = super::super::linear::Domain,
    R = Termination,
> {
    pub(super) search: Option<Retained<G>>,
    pub(super) initial: Option<InitialStop>,
    pub reason: R,
    pub rng: RootedValue,
    pub(super) hybrid: Option<super::polishing::Report>,
    pub elapsed: std::time::Duration,
}
pub(super) fn run<'a>(
    ready: super::super::initialization::Prepared,
    timing: SolverTiming,
    sources: super::super::callbacks::Sources,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<Finished>> + 'a {
    use futures::FutureExt;
    run_with(
        ready,
        timing,
        sources,
        super::super::creation::factory::Owned,
        super::super::callbacks::Direct,
        cooperation,
    )
    .map(|returned| {
        returned.map(|finished| Finished {
            search: finished.search,
            initial: finished.initial,
            reason: super::super::callbacks::take_direct(finished.reason),
            rng: finished.rng,
            hybrid: finished.hybrid,
            elapsed: finished.elapsed,
        })
    })
}
pub(in crate::builtins::math::optim::ga) async fn run_with<F: Factory, S: Source>(
    ready: super::super::initialization::Prepared,
    timing: SolverTiming,
    sources: super::super::callbacks::Sources,
    factory: F,
    source: S,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Finished<F::Geometry, ControlFlow<S::Stop, Termination>>> {
    let runtime = super::continuation::original(cooperation)?;
    let original = &mut *cooperation;
    let prepared = super::continuation::run(&runtime, move || {
        prepare(ready, timing, sources, factory, source, original)
    })?
    .await?;
    super::continuation::run(&runtime, move || active(prepared, cooperation))?.await
}

struct PreparedRun<G, S: Source> {
    started: ControlFlow<S::Stop, super::search::Start<G>>,
    rng: RootedValue,
    observers: Observing,
    display: runmat_builtins::OptimDisplay,
    original_clock: SolverTiming,
    source: S,
}

async fn prepare<F: Factory, S: Source>(
    mut ready: super::super::initialization::Prepared,
    timing: SolverTiming,
    sources: super::super::callbacks::Sources,
    factory: F,
    mut source: S,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<super::super::control::Retained<PreparedRun<F::Geometry, S>>> {
    let runtime = super::continuation::original(cooperation)?;
    let original = &mut *cooperation;
    let rngstate = ready.parts_mut().0.rng.take();
    let setup::Initialized { rng, observers } = super::continuation::run(&runtime, || {
        setup::initialize(rngstate, ready.options(), &timing, original)
    })?
    .await?;
    reporting::begin(ready.problem(), ready.options(), cooperation)?;
    let display = ready.options().display;
    let original_clock = timing.clone();
    let started =
        Search::start_with(ready, timing, sources, factory, &mut source, cooperation).await?;
    super::super::control::Retained::new(
        PreparedRun {
            started,
            rng,
            observers,
            display,
            original_clock,
            source,
        },
        cooperation,
        super::super::control::POLICY,
    )
}

async fn active<G: Geometry, S: Source>(
    prepared: super::super::control::Retained<PreparedRun<G, S>>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Finished<G, ControlFlow<S::Stop, Termination>>> {
    let PreparedRun {
        started,
        rng,
        mut observers,
        display,
        original_clock,
        mut source,
    } = prepared.into_inner();
    let started = match started {
        ControlFlow::Continue(started) => started,
        ControlFlow::Break(stop) => {
            return Ok(Finished {
                search: None,
                initial: None,
                reason: ControlFlow::Break(stop),
                rng,
                hybrid: None,
                elapsed: original_clock.elapsed(),
            })
        }
    };
    let Some(mut search) = started.ready else {
        let initial = started.stopped.ok_or_else(|| {
            invalid("initial search has neither a generation nor a stopping reason")
        })?;
        reporting::stopped(display, initial.reason, cooperation)?;
        return Ok(Finished {
            search: None,
            reason: ControlFlow::Continue(initial.reason),
            initial: Some(initial),
            rng,
            hybrid: None,
            elapsed: original_clock.elapsed(),
        });
    };
    let mut reason = observe_reason(
        notify(
            &mut search,
            &mut observers,
            Phase::Init,
            &mut source,
            cooperation,
        )
        .await?,
    );
    reporting::iteration(&search, cooperation)?;
    while reason.is_none() {
        cooperation.completed(0).await?;
        reason = source
            .stop(Event::Work)
            .map(ControlFlow::Break)
            .or_else(|| {
                search
                    .progress
                    .hard_stop(&search.options, &search.timing)
                    .or_else(|| {
                        search
                            .progress
                            .ordinary_stop(&search.options, search.best.score())
                    })
                    .map(ControlFlow::Continue)
            });
        if reason.is_some() {
            break;
        }
        if let Subproblem::Completed { solved } = search.progress.subproblem(&search.options) {
            let solved_reason = search.complete_subproblem(solved)?;
            let observed = notify(
                &mut search,
                &mut observers,
                Phase::Iter,
                &mut source,
                cooperation,
            )
            .await?;
            reporting::iteration(&search, cooperation)?;
            let observed = match observed {
                ControlFlow::Continue(observed) => observed,
                ControlFlow::Break(stop) => {
                    reason = Some(ControlFlow::Break(stop));
                    break;
                }
            };
            reason = if observed.stopped {
                Some(ControlFlow::Continue(Termination::Observer))
            } else {
                search
                    .progress
                    .hard_stop(&search.options, &search.timing)
                    .map(ControlFlow::Continue)
            };
            if reason.is_none() && !observed.changed {
                reason = solved_reason.map(ControlFlow::Continue);
            }
            if reason.is_none() {
                reason = search
                    .progress
                    .ordinary_stop(&search.options, search.best.score())
                    .map(ControlFlow::Continue);
            }
            if reason.is_some() {
                break;
            }
            reason = search
                .update_merit(solved && !observed.changed, cooperation)
                .await?
                .map(ControlFlow::Continue);
            observers.clear_expectation();
            continue;
        }
        let step = search
            .step_with(
                observers.state().map(RootedValue::value),
                observers.controls().map(RootedValue::value),
                observers.needed(),
                &mut source,
                cooperation,
            )
            .await?;
        let step = match step {
            ControlFlow::Continue(step) => step,
            ControlFlow::Break(stop) => {
                reason = Some(ControlFlow::Break(stop));
                break;
            }
        };
        match step {
            Step::Stopped(stop) => reason = Some(ControlFlow::Continue(stop)),
            Step::Completed(record) => {
                observers.record(record);
                let phase = if search.merit.is_some() {
                    Phase::Interrupt
                } else {
                    Phase::Iter
                };
                reason = observe_reason(
                    notify(&mut search, &mut observers, phase, &mut source, cooperation).await?,
                );
                if phase == Phase::Iter {
                    reporting::iteration(&search, cooperation)?;
                }
            }
        }
    }
    let mut reason = reason.ok_or_else(|| invalid("search ended without a stopping reason"))?;
    let hybrid = if matches!(&reason, ControlFlow::Continue(reason) if reason.may_run_hybrid())
        && search.best.point.is_some()
        && search
            .progress
            .hard_stop(&search.options, &search.timing)
            .is_none()
    {
        match super::polishing::run_with(&mut search, &mut source, cooperation).await? {
            ControlFlow::Continue(report) => report,
            ControlFlow::Break(stop) => {
                reason = ControlFlow::Break(stop);
                None
            }
        }
    } else {
        None
    };
    if hybrid.is_some()
        && search.options.maximum.reached(search.timing.elapsed())
        && matches!(reason, ControlFlow::Continue(_))
    {
        reason = ControlFlow::Continue(Termination::Time);
    }
    if search.best.point.is_none()
        && matches!(&reason, ControlFlow::Continue(reason) if reason.may_run_hybrid())
    {
        reason = ControlFlow::Continue(Termination::NoFeasiblePoint);
    }
    // Done is a final notification, not another search transition: it cannot
    // add a generation, replay a fitness call or retroactively change limits.
    counts(&mut search, &source)?;
    if let ControlFlow::Break(stop) = observers
        .event_with(&search, Phase::Done, &mut source, cooperation)
        .await?
    {
        if matches!(reason, ControlFlow::Continue(_)) {
            reason = ControlFlow::Break(stop);
        }
    }
    counts(&mut search, &source)?;
    if let ControlFlow::Continue(reason) = &reason {
        reporting::finish(&search, *reason, cooperation)?;
    }
    let elapsed = search.timing.elapsed();
    Ok(Finished {
        search: Some(search),
        initial: None,
        reason,
        rng,
        hybrid,
        elapsed,
    })
}
struct Observed {
    stopped: bool,
    changed: bool,
}
fn counts<G, S: Source>(search: &mut Search<G>, source: &S) -> BuiltinResult<()> {
    source.refresh_counts(&mut search.objective_counts, &mut search.nonlinear_counts)
}
async fn notify<G: Geometry, S: Source>(
    search: &mut Search<G>,
    observers: &mut Observing,
    phase: Phase,
    source: &mut S,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<S::Stop, Observed>> {
    let runtime = super::continuation::original(cooperation)?;
    super::continuation::run(&runtime, move || {
        notify_phase(search, observers, phase, source, cooperation)
    })?
    .await
}

async fn notify_phase<G: Geometry, S: Source>(
    search: &mut Search<G>,
    observers: &mut Observing,
    phase: Phase,
    source: &mut S,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<S::Stop, Observed>> {
    source.refresh_counts(&mut search.objective_counts, &mut search.nonlinear_counts)?;
    let event = match observers
        .event_with(search, phase, source, cooperation)
        .await?
    {
        ControlFlow::Continue(event) => event,
        ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
    };
    source.refresh_counts(&mut search.objective_counts, &mut search.nonlinear_counts)?;
    if !event.called {
        return Ok(ControlFlow::Continue(Observed {
            stopped: event.stopped,
            changed: false,
        }));
    }
    let adopted = search
        .adopt_with(
            observers
                .state()
                .ok_or_else(|| invalid("observer returned no state"))?,
            if event.options_changed {
                observers.controls()
            } else {
                None
            },
            source,
            cooperation,
        )
        .await?;
    source.refresh_counts(&mut search.objective_counts, &mut search.nonlinear_counts)?;
    let adopted = match adopted {
        ControlFlow::Continue(adopted) => adopted,
        ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
    };
    if adopted.population {
        observers
            .population_changed(search.options.layout.total(), cooperation)
            .await?;
    }
    observers.adopt_history(cooperation).await?;
    if event.options_changed {
        observers.reconfigure(&search.options, cooperation).await?;
    }
    Ok(ControlFlow::Continue(Observed {
        stopped: event.stopped,
        changed: adopted.changed,
    }))
}

fn observe_reason<S>(observed: ControlFlow<S, Observed>) -> Option<ControlFlow<S, Termination>> {
    match observed {
        ControlFlow::Break(stop) => Some(ControlFlow::Break(stop)),
        ControlFlow::Continue(observed) if observed.stopped => {
            Some(ControlFlow::Continue(Termination::Observer))
        }
        ControlFlow::Continue(_) => None,
    }
}
