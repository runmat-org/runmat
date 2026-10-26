//! Custom search borrows one original evaluator through a revocable callable.
mod arguments;
pub(super) mod measurement;
mod result;
use super::super::{
    evaluation::{Evaluator, Point},
    input::Input,
    invalid,
    linear::Linear,
    options::Options,
    poll::{Completion, Outcome},
    scaling::Scaling,
    timing::Deadline,
};
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::{execution::scoped_callable, BuiltinResult};
use runmat_builtins::OptimOptionField as Field;
use runmat_value::Value;
use std::ops::ControlFlow;

use super::merit::Best;
pub(in crate::builtins::math::optim::patternsearch) async fn run(
    base: &Point,
    geometry: &Linear,
    options: &Options,
    input: &Input<'_>,
    iteration: usize,
    mesh: f64,
    method: &'static str,
    scaling: &Scaling,
    deadline: &Deadline,
    mut merit: impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    evaluator.cooperation.completed(0).await?;
    let initial_score = merit(base, evaluator.cooperation)
        .await?
        .ok_or_else(|| invalid("custom search initial merit is undefined"))?;
    let target = super::spec::custom_target(
        options
            .callback(Field::SearchFcn)
            .ok_or_else(|| invalid("custom SearchFcn is missing"))?,
    )?;
    let mut arguments = arguments::prepare(
        base,
        input,
        options,
        iteration,
        mesh,
        method,
        scaling,
        initial_score,
        evaluator,
    )
    .await?;
    let runtime = evaluator.cooperation.runtime().cloned();
    let token = runtime
        .as_ref()
        .map(crate::context::RuntimeContext::cancellation);
    let started = evaluator.state.objective_calls;
    let mut best = None;
    let driven = scoped_callable::drive(
        runtime.as_ref(),
        token.as_ref(),
        |handle, original| {
            arguments.insert(0, Value::ScopedFunctionHandle(handle));
            Ok(async move {
                common::call_function_sequence_in_context(original.as_ref(), target, arguments, 4)
                    .await
            })
        },
        async |request| {
            let point_axis = evaluator.point_axis();
            measurement::measure(
                request,
                point_axis,
                base,
                initial_score,
                super::merit::Reuse::KnownBase,
                geometry,
                options.controls.constraint_tolerance,
                || deadline.reached().then_some(Completion::Time),
                Completion::Budget,
                &mut best,
                &mut merit,
                evaluator,
            )
            .await
        },
    )
    .await?;
    let completion = match driven {
        ControlFlow::Break(completion) => completion,
        ControlFlow::Continue(returned) => {
            result::validate(
                returned,
                evaluator.state.objective_calls - started,
                base,
                initial_score,
                &mut best,
                evaluator,
            )
            .await?;
            Completion::Finished
        }
    };
    Ok(Outcome {
        improved: best.filter(|best| best.score < initial_score).map(|best| {
            super::super::poll::Improvement {
                point: best.point,
                direction: None,
                merit: super::super::poll::MeritChange {
                    before: initial_score,
                    after: best.score,
                },
            }
        }),
        completion,
        rotatable: false,
    })
}
