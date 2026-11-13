//! Initial population uses the same creation, source and geometry owners.
use super::super::super::evaluation::source::Source;
use super::*;
use futures::TryFutureExt;
use std::ops::ControlFlow;

#[cfg(test)]
impl Search {
    pub async fn start(
        problem: Problem,
        options: Options,
        timing: SolverTiming,
        sources: super::super::super::callbacks::Sources,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Start> {
        let ready = super::super::super::initialization::Ready::new(problem, options, cooperation)?;
        Self::start_with(
            ready,
            timing,
            sources,
            creation::factory::Owned,
            &mut super::super::super::evaluation::source::Direct,
            cooperation,
        )
        .await
        .map(super::super::super::callbacks::take_direct)
    }
}
impl<G: super::super::super::linear::Geometry> Search<G> {
    pub fn start_with<'a, F, S>(
        ready: super::super::super::initialization::Prepared,
        timing: SolverTiming,
        sources: super::super::super::callbacks::Sources,
        factory: F,
        source: &'a mut S,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<S::Stop, Start<G>>>> + 'a
    where
        G: 'a,
        F: creation::factory::Factory<Geometry = G> + 'a,
        S: Source + 'a,
    {
        futures::future::lazy(move |_| {
            let runtime = super::super::continuation::original(cooperation)?;
            super::super::continuation::run(&runtime, move || {
                Self::start_staged(ready, timing, sources, factory, source, cooperation)
            })
        })
        .try_flatten()
    }
    async fn start_staged<F, S>(
        ready: super::super::super::initialization::Prepared,
        timing: SolverTiming,
        sources: super::super::super::callbacks::Sources,
        factory: F,
        source: &mut S,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<S::Stop, Start<G>>>
    where
        F: creation::factory::Factory<Geometry = G>,
        S: Source,
    {
        let runtime = super::super::continuation::original(cooperation)?;
        let original_source = &mut *source;
        let original = &mut *cooperation;
        let prepared = super::super::continuation::run(&runtime, move || {
            Self::start_phase(ready, timing, sources, factory, original_source, original)
        })?
        .await?;
        match prepared {
            ControlFlow::Break(stop) => Ok(ControlFlow::Break(stop)),
            ControlFlow::Continue(Prepared::Stopped(stopped)) => Ok(ControlFlow::Continue(stopped)),
            ControlFlow::Continue(Prepared::Created(created)) => {
                super::super::continuation::run(&runtime, move || {
                    finish::complete(created, source, cooperation)
                })?
                .await
            }
        }
    }
    async fn start_phase<F, S>(
        mut ready: super::super::super::initialization::Prepared,
        timing: SolverTiming,
        sources: super::super::super::callbacks::Sources,
        factory: F,
        source: &mut S,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<S::Stop, Prepared<G>>>
    where
        F: creation::factory::Factory<Geometry = G>,
        S: Source,
    {
        cooperation.completed(0).await?;
        let (mut problem, mut options) = ready.parts_mut();
        source.retune(options.maximum);
        if sources.nonlinear.is_some() != problem.nonlinear.is_some() {
            return Err(super::super::super::invalid(
                "selected constraint source differs from the active problem role",
            ));
        }
        if options.maximum.reached(timing.elapsed()) {
            return Ok(ControlFlow::Continue(Prepared::Stopped(Start {
                ready: None,
                stopped: Some(InitialStop {
                    reason: Termination::Time,
                    nonlinear_calls: 0,
                    nonlinear_points: 0,
                }),
            })));
        }
        // On's real pool/source matching fails before creator/objective calls.
        // Once selected, no callback failure replays through a serial route.
        options.bind_generation_callbacks(cooperation)?;
        let hybrid = super::super::super::hybrid::select(
            options.field(runmat_builtins::OptimOptionField::HybridFcn),
            problem,
            options.population,
            cooperation,
        )?;
        let routes = source.select_routes(
            &problem.objective,
            problem.nonlinear.as_ref(),
            options.vectorized,
            options.parallel,
            cooperation,
        )?;
        let objective_parallel = routes.objective;
        let nonlinear_parallel = routes.nonlinear;
        let runtime = super::super::continuation::original(cooperation)?;
        let original_problem = &mut problem;
        let original_options = &mut options;
        let original_timing = &timing;
        let original_sources = &sources;
        let original_source = &mut *source;
        let original = &mut *cooperation;
        let initial = super::super::continuation::run(&runtime, move || {
            creation::create_with(
                original_problem,
                original_options,
                original_timing,
                Some(original_sources),
                factory,
                original_source,
                original,
            )
        })?
        .await?;
        let initial = match initial {
            ControlFlow::Continue(initial) => initial,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        Ok(ControlFlow::Continue(Prepared::Created(
            super::super::super::control::Retained::new(
                Created {
                    ready,
                    timing,
                    sources,
                    initial,
                    hybrid,
                    objective_parallel,
                    nonlinear_parallel,
                },
                cooperation,
                super::super::super::control::POLICY,
            )?,
        )))
    }
}

mod finish;
struct Created<G> {
    ready: super::super::super::initialization::Prepared,
    timing: SolverTiming,
    sources: super::super::super::callbacks::Sources,
    initial: creation::Initial<G>,
    hybrid: Option<super::super::super::hybrid::Selected>,
    objective_parallel: Option<CallbackBatch>,
    nonlinear_parallel: Option<CallbackBatch>,
}
enum Prepared<G> {
    Stopped(Start<G>),
    Created(super::super::super::control::Retained<Created<G>>),
}
