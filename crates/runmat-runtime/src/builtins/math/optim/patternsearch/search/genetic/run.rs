//! One borrowed genetic invocation, subordinate to the enclosing Pattern
//! deadline and actual point budget. Only retained physical observations may
//! supply an outer improvement, independently of user-supplied GA scores.
use super::super::super::{
    augmented::Augmented,
    evaluation::{Evaluator, Point},
    input::Input,
    invalid,
    linear::Linear,
    options::Options,
    poll::{Completion, Improvement, MeritChange, Outcome},
    timing::Deadline,
};
use super::{
    callbacks::Callbacks,
    geometry::Geometry,
    oracle::{CountsAtEntry, Oracle},
    Stop,
};
use crate::builtins::math::optim::ga::auxiliary;
use crate::{execution::scoped_callable::Scope, BuiltinResult};
use runmat_builtins::OptimOptionField as Field;
use runmat_value::Value;
mod active;

pub(in crate::builtins::math::optim::patternsearch) async fn run(
    base: &Point,
    options: &Options,
    input: &Input<'_>,
    linear: &Linear,
    augmented: Option<&Augmented>,
    deadline: &Deadline,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    evaluator.cooperation.completed(0).await?;
    if deadline.reached() {
        return Ok(empty(Completion::Time));
    }
    if evaluator.remaining() == 0 {
        return Ok(empty(Completion::Budget));
    }
    let bounds = evaluator
        .state
        .original_bounds
        .take()
        .ok_or_else(|| invalid("genetic search lacks its retained original bounds"))?;
    let result = run_inner(
        base, options, input, linear, augmented, deadline, &bounds, evaluator,
    )
    .await;
    // This is the same moved bound owner, including on an ordinary error. A
    // cancelled/dropped invocation never publishes a replacement generation.
    evaluator.state.original_bounds = Some(bounds);
    result
}
fn run_inner<'a, 'e>(
    base: &'a Point,
    options: &'a Options,
    input: &'a Input<'_>,
    linear: &'a Linear,
    augmented: Option<&'a Augmented>,
    deadline: &'a Deadline,
    bounds: &'a super::super::Bounds,
    evaluator: &'a mut Evaluator<'e>,
) -> impl std::future::Future<Output = BuiltinResult<Outcome>> + 'a + use<'a, 'e>
where
    'e: 'a,
{
    crate::execution::after_result(
        async move {
            let prepared = prepare(
                base, options, input, linear, augmented, deadline, bounds, evaluator,
            )
            .await?;
            Ok((prepared, evaluator))
        },
        move |(prepared, evaluator)| {
            let Some(prepared) = prepared else {
                return futures::future::Either::Left(std::future::ready(Ok(empty(
                    Completion::Numerical,
                ))));
            };
            match super::super::super::iteration::selected_search::admit(
                evaluator.cooperation.runtime().cloned(),
                move || {
                    active::run(
                        base, options, linear, augmented, deadline, bounds, evaluator, prepared,
                    )
                },
            ) {
                Ok(future) => futures::future::Either::Right(future),
                Err(error) => futures::future::Either::Left(std::future::ready(Err(error))),
            }
        },
    )
}
async fn prepare<'deadline>(
    base: &Point,
    options: &Options,
    input: &Input<'_>,
    linear: &Linear,
    augmented: Option<&Augmented>,
    deadline: &'deadline Deadline,
    bounds: &super::super::Bounds,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Option<active::Prepared<'deadline>>> {
    let source = options
        .callback(Field::SearchFcn)
        .ok_or_else(|| invalid("missing genetic search generation"))?;
    let explicit = super::super::spec::genetic_options(source)?;
    let timing = deadline.auxiliary()?;
    let initial = match augmented {
        Some(augmented) => augmented.merit(base, evaluator.cooperation).await?,
        None => Some(base.f),
    };
    let Some(initial) = initial else {
        return Ok(None);
    };
    let runtime = evaluator.cooperation.runtime().cloned();
    let token = runtime
        .as_ref()
        .map(crate::context::RuntimeContext::cancellation);
    let scope = Scope::new(runtime.as_ref(), token.as_ref())?;
    let controls = auxiliary::Controls::prepare(
        Value::ScopedFunctionHandle(scope.handle()),
        &base.x,
        explicit,
        (options.controls.vectorized, options.controls.parallel),
        auxiliary::PublicLinear {
            inequalities: input.inequality,
            equalities: input.equality,
            lower: input.lower,
            upper: input.upper,
        },
        bounds.original(),
        linear.matrix.m != 0,
        evaluator.cooperation,
    )
    .await?;
    let prepared = active::Prepared {
        scope,
        controls,
        timing,
        initial,
    };

    Ok(Some(prepared))
}

fn empty(completion: Completion) -> Outcome {
    Outcome {
        improved: None,
        completion,
        rotatable: false,
    }
}
