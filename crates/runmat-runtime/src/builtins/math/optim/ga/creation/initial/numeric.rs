//! Numeric creation owns separate admitted seed, nonlinear and geometry phases.
//! Completed phase state retires before its selected successor is polled.
use super::*;
use crate::builtins::math::optim::ga::{self, creation};
mod finish;
mod seed;

pub(super) async fn create<F: creation::factory::Factory, I: Invocation>(
    problem: &mut Problem,
    options: &mut options::Options,
    timing: &SolverTiming,
    sources: Option<&ga::callbacks::Sources>,
    factory: F,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, Initial<F::Geometry>>> {
    let runtime = ga::controller::continuation::original(cooperation)?;
    let original_problem = &*problem;
    let original_options = &mut *options;
    let original_invoker = &mut *invoker;
    let original_factory = &factory;
    let original = &mut *cooperation;
    let seeded = ga::controller::continuation::run(&runtime, move || {
        seed::prepare(
            original_problem,
            original_options,
            original_factory,
            original_invoker,
            original,
        )
    })?
    .await?;
    let mut seed = match seeded {
        ControlFlow::Continue(seed) => seed,
        ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
    };
    let counts = if options.creation.kind() == Some(OptimGeneticCreation::NonlinearFeasible)
        && !seed.draft.missing().is_empty()
    {
        let original_draft = &mut seed.draft;
        let original_problem = &mut *problem;
        let original_options = &mut *options;
        let original = &mut *cooperation;
        let source = sources.and_then(|sources| sources.nonlinear.as_ref());
        let selected = ga::controller::continuation::run(&runtime, move || {
            creation::nonlinear::prepare(
                original_draft,
                original_problem,
                original_options,
                timing,
                source,
                original,
            )
        })?
        .await?;
        // The preparation continuation is gone before the selected kernel polls.
        selected.await?
    } else {
        creation::nonlinear::Counts::default()
    };
    let original = &mut *cooperation;
    ga::controller::continuation::run(&runtime, move || {
        finish::finish::<F, I>(seed, problem, options, timing, factory, counts, original)
    })?
    .await
}
