//! The public GA controller composes one search lifecycle with its mutable
//! observations, hybrid transition and requested-output publication.
mod best;
mod frame;
mod linear;
mod nonlinear;
mod augmented;
mod progress;
mod search;
mod supplied;
mod state_input;
mod observing;
use super::score_value;
mod adopt;
mod driver;
pub(super) mod continuation;
mod reporting;
mod publication;
mod polishing;
mod migration;

pub(super) async fn solve(
    ready: super::initialization::Prepared,
    timing: super::common::solver_timing::SolverTiming,
    cooperation: &mut super::ScalarCooperation,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    let sources = super::callbacks::Sources::select(ready.problem(), cooperation)?;
    let runtime = continuation::original(cooperation)?;
    let original_cooperation = &mut *cooperation;
    let finished = continuation::run(&runtime, move || {
        driver::run(ready, timing, sources, original_cooperation)
    })?
    .await?;
    publication::finalize(finished, cooperation)
}

pub(super) use driver::Finished;

/// Completion of a borrowed search, including its enclosing oracle's stop.
type BorrowedResult<G, S> =
    crate::BuiltinResult<Finished<G, std::ops::ControlFlow<S, super::outcome::Termination>>>;
pub(super) fn run_borrowed<'a, F, S>(
    ready: super::initialization::Prepared,
    timing: super::common::solver_timing::SolverTiming,
    sources: super::callbacks::Sources,
    factory: F,
    source: &'a mut S,
    cooperation: &'a mut super::ScalarCooperation,
) -> impl std::future::Future<Output = BorrowedResult<F::Geometry, S::Stop>> + 'a
where
    F: super::creation::factory::Factory + 'a,
    F::Geometry: 'a,
    S: super::evaluation::source::Source + 'a,
    S::Stop: 'a,
{
    use futures::TryFutureExt;
    futures::future::lazy(move |_| {
        let runtime = continuation::original(cooperation)?;
        continuation::run(&runtime, move || {
            driver::run_with(ready, timing, sources, factory, source, cooperation)
        })
    })
    .try_flatten()
}
