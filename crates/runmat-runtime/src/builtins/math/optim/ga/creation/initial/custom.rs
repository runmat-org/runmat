//! Only the selected custom-population creator retains its public controls.
use super::*;
use crate::builtins::math::optim::ga::{self, creation};
pub(super) async fn create<F: creation::factory::Factory, I: Invocation>(
    problem: &mut Problem,
    options: &mut options::Options,
    _factory: F,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, Initial<F::Geometry>>> {
    let (initial, changed) = super::prepare(problem, options, cooperation)?;
    let controls = options.public(cooperation).await?;
    let runtime = ga::controller::continuation::original(cooperation)?;
    let members = options.layout.total();
    let variables = problem.variables;
    let creator = &mut options.creation;
    let public_problem = &problem.public;
    let public_controls = controls.value();
    let original_invoker = &mut *invoker;
    let original = &mut *cooperation;
    let created = ga::controller::continuation::run(&runtime, move || {
        creation::custom::create_with(
            initial,
            members,
            variables,
            creator,
            public_problem,
            public_controls,
            original_invoker,
            original,
        )
    })?
    .await?;
    let (generation, supplied) = match created {
        ControlFlow::Continue(created) => created,
        ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
    };
    Ok(ControlFlow::Continue(Initial {
        generation,
        domain: None,
        admission: creation::Admission::Complete,
        constraint_calls: 0,
        constraint_points: 0,
        extents: None,
        supplied,
        invalidated_scores: changed.rows,
    }))
}

pub(super) async fn numeric<I: Invocation>(
    draft: creation::Draft,
    problem: &Problem,
    options: &options::Options,
    bitstring: bool,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, creation::Draft>> {
    let controls = options.public(cooperation).await?;
    creation::custom::numeric(
        draft,
        &options.creation,
        &problem.public,
        controls.value(),
        bitstring,
        invoker,
        cooperation,
    )
    .await
}
